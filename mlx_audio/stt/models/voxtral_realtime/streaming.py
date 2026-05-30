"""Streaming audio ingestion pipeline for Voxtral Realtime.

Mirrors the batch mel/conv_stem path but accepts audio incrementally via
.append(samples) and .close(). Frames are produced as soon as enough
raw audio has arrived. At close() the residual tail is flushed using
right-reflect padding so the stream output matches the batch output
sample-for-sample for any audio that is fully fed then closed.
"""

from __future__ import annotations

import math
import queue
import threading
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import RAW_AUDIO_LENGTH_PER_TOK, _num_delay_tokens


class StreamingAudioSource:
    """Thread-safe blocking queue of raw audio samples.

    Producer side calls .append(np.ndarray) as audio arrives and .close()
    when the stream ends. Consumer side calls .read() which blocks until
    some samples are available OR the source is closed (whichever first).
    .read() returns (samples, closed_flag). After close, subsequent reads
    drain any buffered audio and then return (empty, True).
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False

    def append(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self._q.put(samples.reshape(-1).copy())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._q.put(None)  # sentinel

    def read(self, timeout: Optional[float] = None) -> tuple[np.ndarray, bool]:
        """Block until samples are available or the source closes.

        Returns (samples, closed). ``samples`` can be empty only when
        ``closed`` is True. Drains any buffered chunks in a single call
        so multiple small appends are coalesced before decoding runs.
        """
        try:
            first = self._q.get(timeout=timeout)
        except queue.Empty:
            return np.zeros(0, dtype=np.float32), False
        if first is None:
            return np.zeros(0, dtype=np.float32), True

        chunks: list[np.ndarray] = [first]
        closed = False
        while True:
            try:
                nxt = self._q.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                closed = True
                break
            chunks.append(nxt)
        return np.concatenate(chunks), closed


class StreamingMel:
    """Incrementally compute log-mel spectrogram frames.

    Parity contract with compute_mel_spectrogram(audio_all):
        Feed the same samples through .append (in any chunking),
        then .close(). Concatenating the returned per-call outputs
        yields the same [mel_bins, frames] spectrogram (up to fp
        rounding) as the batch call.
    """

    def __init__(
        self,
        mel_filters: mx.array,
        window_size: int = 400,
        hop_length: int = 160,
        global_log_mel_max: float = 1.5,
    ):
        self.window_size = window_size
        self.hop_length = hop_length
        self.pad_size = window_size // 2
        self.global_log_mel_max = global_log_mel_max
        self.mel_filters = mel_filters  # [freq_bins, mel_bins]

        n = mx.arange(window_size, dtype=mx.float32)
        self._window = 0.5 * (1.0 - mx.cos(2.0 * math.pi * n / window_size))
        mx.eval(self._window)

        # Raw audio ring: _buf[0] corresponds to raw index _buf_start
        self._buf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._buf_start: int = 0
        self._n_received: int = 0
        self._next_k: int = 0
        self._closed: bool = False

    def append(self, samples: np.ndarray) -> Optional[mx.array]:
        """Append samples; return [mel_bins, n_new_frames] or None if no frame ready."""
        if self._closed:
            raise RuntimeError("StreamingMel is closed")
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        self._buf = np.concatenate([self._buf, samples])
        self._n_received += len(samples)
        return self._drain(final=False)

    def close(self) -> Optional[mx.array]:
        """Close stream; flush trailing frames with right-reflect pad, apply drop-last."""
        if self._closed:
            return None
        self._closed = True
        return self._drain(final=True)

    def _extract_windows(self, k_start: int, k_end: int) -> Optional[np.ndarray]:
        """Vectorized window extraction for a contiguous range of frames.

        Builds ``[n_frames, window_size]`` directly with numpy advanced indexing
        — the previous per-frame Python loop over window_size (400) was 400
        iterations per mel frame, costing single-digit milliseconds per 80 ms
        realtime chunk and dominating the CPU side of the streaming pipeline.

        Returns None if any frame in the range can't be resolved yet (e.g. the
        right-reflect region isn't legal because we haven't closed).
        """
        N = self._n_received
        n_frames = k_end - k_start
        if n_frames <= 0:
            return None

        # Global raw indices for every (frame, sample) pair
        frame_starts = np.arange(k_start, k_end) * self.hop_length - self.pad_size
        offsets = np.arange(self.window_size)
        r = frame_starts[:, None] + offsets[None, :]  # [n_frames, window_size]

        # Reflect padding: left uses src = -r, right uses src = 2N-2-r
        left_mask = r < 0
        right_mask = r >= N

        if right_mask.any() and not self._closed:
            return None  # caller should wait for more audio / close()

        src = np.where(left_mask, -r, np.where(right_mask, 2 * N - 2 - r, r))

        # Any index still out of [0, N) signals a buffer gap (should not happen
        # in normal use, but matches the conservative behavior of the previous
        # scalar loop).
        if ((src < 0) | (src >= N)).any():
            return None

        buf_idx = src - self._buf_start
        if (buf_idx < 0).any() or (buf_idx >= len(self._buf)).any():
            return None

        return self._buf[buf_idx]  # [n_frames, window_size], float32 view/copy

    def _drain(self, *, final: bool) -> Optional[mx.array]:
        N = self._n_received
        if final:
            # Match batch: n_frames_raw = 1 + N//hop, then drop last -> N//hop frames.
            max_k_inclusive = N // self.hop_length - 1
        else:
            # No right-reflect allowed yet. Right-boundary rule:
            #   frame k needs raw[k*hop + pad - 1] in-range -> k*hop + pad <= N.
            max_k_inclusive = (N - self.pad_size) // self.hop_length

        if self._next_k > max_k_inclusive:
            return None

        frames_np = self._extract_windows(self._next_k, max_k_inclusive + 1)
        if frames_np is None:
            return None
        self._next_k = max_k_inclusive + 1
        frames_mx = mx.array(frames_np, dtype=mx.float32) * self._window[None, :]
        spectrum = mx.fft.rfft(frames_mx, n=self.window_size, axis=-1)
        magnitudes = mx.abs(spectrum) ** 2  # [n_new, freq_bins]
        # Drop-last (the batch path's magnitudes[:-1, :] over the TIME axis)
        # is applied globally via max_k_inclusive when final=True, not here.
        mel_spec = magnitudes @ self.mel_filters  # [n_new, mel_bins]
        log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))
        min_val = self.global_log_mel_max - 8.0
        log_spec = mx.maximum(log_spec, min_val)
        log_spec = (log_spec + 4.0) / 4.0
        out = log_spec.T  # [mel_bins, n_new]
        mx.eval(out)
        return out

    def trim(self, keep_from_raw_idx: int) -> None:
        """Drop buffered samples with global index < keep_from_raw_idx.

        Call sparingly; streaming only needs roughly window_size samples of
        history at steady state, but right-reflect at close needs the tail.
        """
        keep_from_raw_idx = max(self._buf_start, keep_from_raw_idx)
        drop = keep_from_raw_idx - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start += drop


class StreamingCausalConv1d:
    """Incremental causal Conv1d wrapper over an existing CausalConv1d module.

    Reuses the weights of the wrapped module. Manages edge state between
    calls so the concatenation of outputs equals the batch output for the
    concatenation of inputs.

    Usage:
        sc = StreamingCausalConv1d(existing_causal_conv)
        y0 = sc.step(x0)  # [n0_out, C_out]
        y1 = sc.step(x1)
        ...  # concat(y*) == existing_causal_conv(concat(x*))
    """

    def __init__(self, causal_conv):
        self.conv = causal_conv  # CausalConv1d
        self.kernel_size = causal_conv.kernel_size
        self.stride = causal_conv.stride
        self.left_pad = causal_conv.padding  # kernel - stride
        # keep = (kernel - stride) is the state size we carry between calls.
        self._keep = self.kernel_size - self.stride
        self._state: Optional[mx.array] = None  # [n_state, C_in]
        self._initialized = False

    def step(self, x_new: mx.array) -> mx.array:
        """Feed new input [n_new, C_in]; return output [n_out, C_out].

        On the first call, the required left-pad of (kernel - stride) zeros
        is prepended (matching CausalConv1d behavior). Subsequent calls reuse
        the cached tail of prior inputs as left context.
        """
        if x_new.shape[0] == 0:
            return x_new[:0]  # empty passthrough
        if not self._initialized:
            if self._keep > 0:
                pad = mx.zeros((self._keep, x_new.shape[-1]), dtype=x_new.dtype)
                context = mx.concatenate([pad, x_new], axis=0)
            else:
                context = x_new
            self._initialized = True
        else:
            context = (
                mx.concatenate([self._state, x_new], axis=0)
                if self._state is not None
                else x_new
            )

        # If we do not yet have enough timesteps to cover one kernel window,
        # keep buffering until a later call. This happens in realtime mode when
        # the client drips very short audio chunks and the first conv produces
        # only 1 frame, which is still too short for the second conv (k=3,s=2).
        if context.shape[0] < self.kernel_size:
            self._state = context
            return mx.zeros((0, self.conv.conv.weight.shape[0]), dtype=x_new.dtype)

        # Call the inner nn.Conv1d directly to avoid the CausalConv1d wrapper
        # re-adding its own left-pad.
        out = self.conv.conv(context[None, :, :]).squeeze(0)  # [n_out, C_out]
        n_out = out.shape[0]

        # Save the last (kernel - stride) inputs as state. After this call the
        # next kernel window will start exactly at position `n_out * stride`
        # within the current context, leaving `context.shape[0] - n_out*stride`
        # samples as "leftover", which must be (kernel - stride) by construction
        # when the input stream has been packed with no gaps.
        if self._keep > 0:
            # Guard: if leftover < keep (early edge), save all.
            leftover = context.shape[0] - n_out * self.stride
            if leftover <= 0:
                self._state = None
            elif leftover >= self._keep:
                self._state = context[-self._keep :]
            else:
                self._state = context[-leftover:]
        else:
            self._state = None

        return out


class StreamingConvStem:
    """Streaming version of AudioEncoder.conv_stem.

    Reuses the conv layer weights of an existing AudioEncoder. Feeds mel
    frames in as they arrive, returns the corresponding conv_out frames.

    Parity contract with AudioEncoder.conv_stem(mel_all):
        Concatenating the per-call outputs of step(mel_chunk) equals
        conv_stem(full_mel) *before* the `trunc` truncation (which is
        applied to the total at the caller level when alignment is known).

    Note: the front-trunc by downsample_factor in conv_stem is NOT applied
    here because streaming can't know the final count. Callers are expected
    to arrange input sizes so the total is already aligned (the standard
    Voxtral padding aligns audio to 1280 samples = 8 mel frames, which
    makes conv_stem output naturally divisible by 4).
    """

    def __init__(self, encoder):
        # Bypass the CausalConv1d wrappers' internal left-pad; manage it ourselves.
        self._c0 = StreamingCausalConv1d(encoder.conv_layers_0_conv)
        self._c1 = StreamingCausalConv1d(encoder.conv_layers_1_conv)

    def step(self, mel_chunk: mx.array) -> mx.array:
        """Process a mel chunk. mel_chunk: [mel_bins, n_frames]. Returns [n_out, dim]."""
        # Cast mel to conv weight dtype (bf16) so the entire encoder/projection
        # pipeline runs in bf16 instead of fp32. Keeping fp32 propagates all
        # the way to the decoder input and forces the decoder forward to run
        # in fp32 (~2x slower than bf16 on Apple Silicon).
        target_dtype = self._c0.conv.conv.weight.dtype
        if mel_chunk.shape[1] == 0:
            return mx.zeros((0, self._c0.conv.conv.weight.shape[0]), dtype=target_dtype)
        # Match batch: conv_stem takes mel.T[None, :, :] ([1, frames, 128])
        x = mel_chunk.T.astype(target_dtype)  # [frames, 128]
        x = self._c0.step(x)
        x = nn.gelu(x)
        x = self._c1.step(x)
        x = nn.gelu(x)
        return x  # [out_frames, dim]


class StreamingEncoder:
    """Streaming wrapper over AudioEncoder transformer layers.

    Reuses encoder weights; maintains a RotatingKVCache (max_size=sw) per
    layer and a running RoPE position. Each step() call processes a chunk
    of conv_out and returns the corresponding post-norm encoder output.

    Parity contract with AudioEncoder.encode_chunks(conv_out_all):
        Concatenating the per-call outputs of step(conv_chunk) equals
        concatenating the yields of encode_chunks(full_conv_out), for any
        chunking of the inputs (including chunk sizes smaller than the
        sliding window).
    """

    def __init__(self, encoder):
        from mlx_lm.models.cache import RotatingKVCache

        self.encoder = encoder
        self._sw = encoder.config.sliding_window
        self._caches = [
            RotatingKVCache(max_size=self._sw, keep=0)
            for _ in range(len(encoder.transformer_layers))
        ]
        self._pos = 0  # global position for RoPE

    def step(self, conv_chunk: mx.array) -> mx.array:
        """conv_chunk: [n_frames, dim] -> encoded: [n_frames, dim]."""
        chunk_len = conv_chunk.shape[0]
        if chunk_len == 0:
            return conv_chunk

        # The mask only depends on chunk_len + the current KV offset, so build
        # it once from cache[0] and share it across all 32 layers. Must be a
        # materialized array: under streaming, K_len = Q_len + cache_size and
        # SDPA's "causal" string shortcut is only valid when Q_len == K_len.
        mask = self._caches[0].make_mask(
            chunk_len, window_size=self._sw, return_array=True
        )
        x = conv_chunk
        for i, layer in enumerate(self.encoder.transformer_layers):
            x = layer(x, self._pos, mask, cache=self._caches[i])
        out = self.encoder.transformer_norm(x)
        self._pos += chunk_len
        return out


class VoxtralStreamingSession:
    """Stateful streaming transcription session.

    Usage pattern:
        sess = model.create_streaming_session(...)
        # producer thread / async task
        sess.feed(samples_np)          # cheap, thread-safe
        sess.feed(more_samples)
        sess.close()                   # end-of-stream

        # consumer thread (MLX executor)
        while not sess.done:
            deltas = sess.step(max_decode_tokens=4)
            for d in deltas: emit(d)

    The split between ``feed`` (just queues raw samples) and ``step``
    (does MLX work and returns a bounded number of deltas) lets the
    caller release the MLX executor between ``step`` calls, so other
    MLX-bound work can be interleaved.

    Parameters:
        max_tokens: decode stop cap (total tokens per utterance).
        temperature: sampling temperature (0 = greedy).
        transcription_delay_ms: target decoder lag behind audio, i.e.
            how much audio the encoder accumulates BEFORE the decoder
            emits its first token. This is the core latency/quality
            knob:
              - Smaller (e.g. 160, 320 ms): deltas appear sooner but
                the decoder has less acoustic context, so WER rises
                and partial hypotheses flip more often.
              - Larger (e.g. 640, 960 ms): more stable transcripts,
                but each delta arrives that many ms after the word
                was spoken.
              - 2400 ms: per Mistral's recommended-settings table
                (see ``config.py`` link), this is the high-latency
                preset whose WER is within a fraction of a point of
                the offline (non-streaming) mode — use it when you
                need offline-grade quality but still want the
                incremental-delta API (e.g. live captioning of
                pre-recorded audio, or tolerant real-time flows).
            Defaults to ``config.transcription_delay_ms`` (480 ms,
            Mistral's sweet spot between latency and quality). This
            also shifts the prompt length
            (``1 + n_left + n_delay``) and the right pad injected at
            ``close()``.
    """

    def __init__(
        self,
        model,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        transcription_delay_ms: Optional[int] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        cfg = model.config
        delay_ms = transcription_delay_ms or cfg.transcription_delay_ms
        self._n_delay = _num_delay_tokens(delay_ms)
        self._n_left = cfg.n_left_pad_tokens
        self._prompt_len = 1 + self._n_left + self._n_delay
        self._raw_tok = RAW_AUDIO_LENGTH_PER_TOK

        model._ensure_ada_scales(transcription_delay_ms)

        aec = cfg.audio_encoding_args
        mel_filters = model._ensure_mel_filters()
        self._smel = StreamingMel(
            mel_filters,
            window_size=aec.window_size,
            hop_length=aec.hop_length,
            global_log_mel_max=aec.global_log_mel_max,
        )
        self._mel_filters = mel_filters
        self._aec = aec
        self.input_sample_rate: int = int(aec.sampling_rate)
        self._sconv = StreamingConvStem(model.encoder)
        self._senc = StreamingEncoder(model.encoder)
        self._sproj = StreamingDownsampler(model.encoder)

        self._audio_q: list[np.ndarray] = []
        self._audio_lock = threading.Lock()
        self._audio_closed = False
        self._flushed_close = False

        self._adapter_frames: list[mx.array] = []
        self._prefilled = False
        self._cache = None
        self._next_tok: Optional[mx.array] = None
        self._pos = self._prompt_len
        self.generated: list[int] = []
        self._prev_text = ""
        self._trailing_after_close = 0
        self._done = False
        self._left_pad_seeded = False

    @property
    def done(self) -> bool:
        return self._done

    def feed(self, samples: np.ndarray) -> None:
        """Queue raw audio samples; cheap and thread-safe.

        No MLX work happens here — samples are just buffered. The actual
        mel/encoder work runs inside ``step()``.
        """
        if samples is None:
            return
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.float32)
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if samples.size == 0:
            return
        with self._audio_lock:
            self._audio_q.append(samples.reshape(-1))

    def close(self) -> None:
        """Signal end-of-audio; safe to call from any thread."""
        with self._audio_lock:
            self._audio_closed = True

    def step(self, *, max_decode_tokens: int = 4) -> list[str]:
        """Run one unit of MLX work. Call from the MLX executor thread.

        Drains pending audio into the encoder pipeline, runs prefill if
        ready, then decodes up to ``max_decode_tokens`` tokens (stopping
        early if we catch up to the available audio). Returns a list of
        text deltas emitted during this call.

        ``max_decode_tokens`` is the yield granularity, not a free
        speedup: each call decodes at most that many tokens and then
        returns so the caller can release the MLX executor for other
        work. Smaller values (4-8) interleave more finely but pay
        per-call overhead; larger values (32-64) amortize overhead but
        hold the executor longer. Pick based on how often other MLX
        tasks need to run; if decoding is the bottleneck, a higher
        value is usually fine because interleaving can't create
        compute that isn't there.
        """
        if self._done:
            return []

        self._ingest_pending()

        if not self._prefilled:
            if self._n_adapter() < self._prompt_len:
                if self._flushed_close:
                    # Closed before we could prefill anything.
                    self._done = True
                return []
            self._do_prefill()
            self._prefilled = True

        return self._decode_some(max_decode_tokens)

    def _ingest_pending(self) -> None:
        """Drain audio queue; also seed left-pad and handle close flush."""
        if not self._left_pad_seeded:
            left_pad = np.zeros(self._n_left * self._raw_tok, dtype=np.float32)
            self._ingest_mel(self._smel.append(left_pad))
            self._left_pad_seeded = True

        while True:
            with self._audio_lock:
                if not self._audio_q:
                    closed = self._audio_closed
                    break
                chunk = self._audio_q.pop(0)
            self._ingest_mel(self._smel.append(chunk))

        if closed and not self._flushed_close:
            self._flushed_close = True
            right_pad_toks = (self._n_delay + 1) + 10
            right_pad = np.zeros(right_pad_toks * self._raw_tok, dtype=np.float32)
            self._ingest_mel(self._smel.append(right_pad))
            self._ingest_mel(self._smel.close())

    def _ingest_mel(self, mel_chunk: Optional[mx.array]) -> None:
        if mel_chunk is None or mel_chunk.shape[1] == 0:
            return
        conv_out = self._sconv.step(mel_chunk)
        if conv_out.shape[0] == 0:
            return
        encoded = self._senc.step(conv_out)
        adapter = self._sproj.step(encoded)
        if adapter.shape[0] > 0:
            mx.eval(adapter)
            self._adapter_frames.append(adapter)

    def _n_adapter(self) -> int:
        return sum(a.shape[0] for a in self._adapter_frames)

    def _coalesce_adapter(self) -> mx.array:
        if len(self._adapter_frames) > 1:
            merged = mx.concatenate(self._adapter_frames, axis=0)
            mx.eval(merged)
            self._adapter_frames = [merged]
        return self._adapter_frames[0]

    def _adapter_at(self, pos: int) -> mx.array:
        if len(self._adapter_frames) > 8:
            self._coalesce_adapter()
        offset = 0
        for piece in self._adapter_frames:
            if pos < offset + piece.shape[0]:
                return piece[pos - offset]
            offset += piece.shape[0]
        raise IndexError(f"pos={pos} out of adapter range (have {offset})")

    def _do_prefill(self) -> None:
        adapter_concat = self._coalesce_adapter()
        prompt_ids = [self.model.config.bos_token_id] + [
            self.model.config.streaming_pad_token_id
        ] * (self._n_left + self._n_delay)
        prompt_ids_mx = mx.array(prompt_ids)
        prompt_text_embeds = self.model.decoder.embed_tokens(prompt_ids_mx)
        prefix_embeds = adapter_concat[: self._prompt_len] + prompt_text_embeds

        h, self._cache = self.model.decoder.forward(prefix_embeds, start_pos=0)
        logits = self.model.decoder.logits(h[-1])
        cache_arrays = [
            a for c in self._cache for a in (c.keys, c.values) if a is not None
        ]
        mx.eval(logits, *cache_arrays)
        self._next_tok = self.model._next_token_mx(logits, self.temperature)
        mx.async_eval(self._next_tok)

    def _decode_some(self, max_decode_tokens: int) -> list[str]:
        deltas: list[str] = []
        eos = self.model.config.eos_token_id
        tok_emb = self.model.decoder.tok_embeddings

        for _ in range(max_decode_tokens):
            # Before consuming the pending token, make sure we'll be able
            # to run a forward pass for it (which needs adapter[pos] while
            # audio is still flowing). If neither adapter nor close is
            # ready, return and let the caller feed more audio.
            if self._n_adapter() <= self._pos and not self._flushed_close:
                return deltas

            if self._n_adapter() <= self._pos:
                # Closed and out of audio: the right-pad (silence) tokens we
                # appended at close() should already have let the model emit
                # EOS. Flush the last pending token without further padding —
                # matches voxmlx's finalize() behavior.
                token = int(self._next_tok.item())
                self.generated.append(token)
                text_so_far = self.model._tokenizer.decode(
                    [t for t in self.generated if t != eos]
                )
                if text_so_far != self._prev_text:
                    deltas.append(text_so_far[len(self._prev_text) :])
                    self._prev_text = text_so_far
                self._done = True
                return deltas

            # Dispatch the current forward BEFORE the .item() sync so the
            # previous step's eval overlaps with the current step's compute
            # queueing — this is the pipelining pattern voxmlx uses. We pass
            # ``self._next_tok`` (an mx.array living on the GPU) directly to
            # the embedding lookup instead of round-tripping via
            # ``mx.array([int(token)])``, which would force a CPU→GPU sync
            # on every step.
            prev_tok_mx = self._next_tok  # shape [], argmax result
            token_embed = tok_emb(prev_tok_mx.reshape(1))[0]
            embed = self._adapter_at(self._pos) + token_embed

            h, self._cache = self.model.decoder.forward(
                embed[None, :], start_pos=self._pos, cache=self._cache
            )
            logits = self.model.decoder.logits(h.squeeze(0))
            new_next_tok = self.model._next_token_mx(logits, self.temperature)
            mx.async_eval(new_next_tok)

            # Now read the PREVIOUS step's token from the GPU. This .item()
            # only waits for the argmax from the prior iteration — the
            # current iteration's forward is already queued.
            token = int(prev_tok_mx.item())
            self.generated.append(token)

            text_so_far = self.model._tokenizer.decode(
                [t for t in self.generated if t != eos]
            )
            if text_so_far != self._prev_text:
                deltas.append(text_so_far[len(self._prev_text) :])
                self._prev_text = text_so_far

            if token == eos or len(self.generated) > self.max_tokens:
                self._done = True
                return deltas

            self._next_tok = new_next_tok
            self._pos += 1
            if len(self.generated) % 256 == 0:
                mx.clear_cache()

        return deltas


class StreamingDownsampler:
    """Buffers encoder output frames and emits adapter-projected frames in
    downsample_factor-aligned groups.

    Reuses audio_language_projection weights of an AudioEncoder.
    """

    def __init__(self, encoder):
        self.encoder = encoder
        self._ds = encoder.config.downsample_factor
        self._dim = encoder.config.dim
        self._buf: Optional[mx.array] = None  # [n, dim] pending frames

    def step(self, encoded_chunk: mx.array) -> mx.array:
        """encoded_chunk: [n_frames, dim]. Returns adapter frames [n_out, decoder_dim]."""
        if encoded_chunk.shape[0] == 0:
            return self._empty_adapter(encoded_chunk.dtype)

        if self._buf is not None and self._buf.shape[0] > 0:
            x = mx.concatenate([self._buf, encoded_chunk], axis=0)
        else:
            x = encoded_chunk

        n = x.shape[0]
        usable = n - (n % self._ds)
        if usable == 0:
            self._buf = x
            return self._empty_adapter(encoded_chunk.dtype)
        to_project = x[:usable]
        self._buf = x[usable:] if usable < n else None

        projected = self.encoder.downsample_and_project(to_project)
        return projected

    def _empty_adapter(self, dtype) -> mx.array:
        """Empty return that matches the adapter output dim (decoder dim, 3072).

        Callers check ``shape[0] > 0`` so the second dim is never read today,
        but keeping both zero-return paths consistent avoids a latent bug if
        anything ever reshapes or concatenates these empties.
        """
        proj_out_dim = self.encoder.audio_language_projection_2.weight.shape[0]
        return mx.zeros((0, proj_out_dim), dtype=dtype)
