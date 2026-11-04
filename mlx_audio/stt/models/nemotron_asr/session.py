"""Cooperative live-input adapter for the realtime STT server."""

from collections import deque
from threading import Lock

import mlx.core as mx
import numpy as np

from . import tokenizer
from .audio import StreamingLogMelSpectrogram
from .buffered import BufferedContext, BufferedEncoderState
from .streaming import ConformerStreamingState


class NemotronStreamingSession:
    """Single-consumer decoder; feed/close may run on another thread.

    Queued input is limited to 30 seconds. Overflow is explicit; audio is never
    silently dropped. For Nemotron 3.5, step budgets joint evaluations (including
    blanks), not just visible text, so silence also yields cooperatively. A Nemotron
    Unified (buffered) step ingests up to a batch of chunks and decodes all of it;
    undecoded frames count against the 30-second limit.
    """

    def __init__(
        self,
        model,
        *,
        temperature=0.0,
        language=None,
        transcription_delay_ms=None,
        att_context_size=None,
    ):
        if temperature != 0.0:
            raise ValueError("Nemotron streaming supports greedy temperature=0 only")
        self.model = model
        self.language = language or model.default_language
        model._resolve_prompt_index(self.language)
        self.att_context_size = model.default_att_context_size
        self._context = None
        if model.buffered:
            self._context = BufferedContext.from_att_context_size(
                model.default_att_context_size
            )
            if att_context_size is not None:
                self._context = BufferedContext.from_att_context_size(att_context_size)
            elif transcription_delay_ms is not None:
                self._context = BufferedContext.from_delay_ms(
                    transcription_delay_ms, left=self._context.left
                )
            self.att_context_size = self._context.as_list()
        elif att_context_size is not None:
            raise ValueError("att_context_size needs a buffered (Unified) checkpoint")
        elif (
            transcription_delay_ms is not None
            and model.model_type == "nemotron_asr_streaming"
        ):
            lookahead = {80: 0, 160: 1, 560: 6, 1120: 13}.get(transcription_delay_ms)
            if lookahead is None:
                raise ValueError(
                    f"Unsupported transcription_delay_ms={transcription_delay_ms}; "
                    "expected one of [80, 160, 560, 1120]"
                )
            self.att_context_size = [model.default_att_context_size[0], lookahead]
        self.input_sample_rate = model.preprocessor_config.sample_rate
        self._lock = Lock()
        self.reset()

    def reset(self):
        """Reset from the decoder thread, with no concurrent step in progress."""
        with self._lock:
            self._audio = deque()
            self._queued = 0
            self._closed = False
            self._done = False
        self._encoded = deque()
        # Encoded frames not yet decoded; feed() counts them against the input limit.
        self._pending_frames = 0
        self._frame = 0
        self._symbols = 0
        self._last_token = self.model.blank_id
        self._hidden = None
        self._prediction = None
        self._flushed = False
        self._has_text = False
        if self._context is not None:
            self._buffered = BufferedEncoderState(
                self.model, self._context, self.language
            )
            return
        self._frontend = StreamingLogMelSpectrogram(self.model.preprocessor_config)
        first_chunk_mel = None
        if self.model.model_type == "nemotron_asr_streaming":
            factor = self.model.encoder_config.subsampling_factor
            first_chunk_mel = 1 + factor * self.att_context_size[1]
        self._encoder = ConformerStreamingState(
            self.model.encoder,
            att_context_size=self.att_context_size,
            first_chunk_mel=first_chunk_mel,
        )

    @property
    def done(self) -> bool:
        return self._done

    def feed(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("expected finite mono PCM samples")
        with self._lock:
            if self._closed:
                raise RuntimeError("streaming input is closed")
            backlog = self._queued + samples.size
            if self._context is not None:
                backlog += self._pending_frames * self._buffered.frame_samples
            if backlog > 30 * self.input_sample_rate:
                raise BufferError("Nemotron streaming input queue exceeds 30 seconds")
            if samples.size:
                self._audio.append(samples.copy())
                self._queued += samples.size

    def close(self) -> None:
        """Signal end-of-input; step drains and flushes it exactly once."""
        with self._lock:
            self._closed = True

    def cancel(self):
        """Discard a session on the decoder thread without producing a final."""
        self.reset()
        with self._lock:
            self._closed = True
            self._done = True

    def _ingest(self):
        # At most one native encoder chunk of audio per cooperative step; a
        # buffered session takes up to a batch of chunks, so clients that send
        # more than one chunk per message keep up.
        if self._context is not None:
            limit = self._buffered.chunk_samples * self._buffered.MAX_BATCH
        else:
            limit = self._encoder.chunk_mel * self.model.preprocessor_config.hop_length
        parts = []
        with self._lock:
            while self._audio and limit:
                head = self._audio.popleft()
                count = min(limit, head.size)
                parts.append(head[:count])
                if count < head.size:
                    self._audio.appendleft(head[count:])
                self._queued -= count
                limit -= count
            final = self._closed and not self._audio
        if not parts and not final:
            return
        samples = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        if self._context is not None:
            chunks = self._buffered.push(samples, final=final)
            self._encoded.extend(chunks)
            self._pending_frames += sum(chunk.shape[1] for chunk in chunks)
            self._flushed = final
            return
        # Empty streams should not manufacture a reflected silence frame.
        if final and not samples.size and not self._frontend.total_samples:
            self._flushed = True
            return
        mel = self._frontend.push(mx.array(samples), final=final)
        for encoded in self._encoder.push(mel, final=final):
            prompted = self.model.apply_prompt(encoded, self.language)
            self._encoder.materialize(prompted)
            self._encoded.append(prompted)
        self._flushed = final

    def step(self, *, max_decode_tokens: int = 4) -> list[str]:
        if max_decode_tokens <= 0:
            raise ValueError("max_decode_tokens must be positive")
        if self.done:
            return []
        if self._context is not None:
            # Buffered: ingest every step, so the audio queue drains even while
            # decoding catches up on frames already encoded.
            if not self._flushed:
                self._ingest()
        elif not self._encoded and not self._flushed:
            self._ingest()
        deltas = []
        # The realtime server steps once per message with a small budget. A buffered
        # session decodes everything it ingested (at most a batch of chunks, each frame
        # at most max_symbols tokens), so it keeps up at any message size up to that
        # batch; Nemotron 3.5 keeps the budget as joint evaluations.
        evaluations = 0
        while self._encoded:
            if self._context is None and evaluations == max_decode_tokens:
                break
            evaluations += 1
            feature = self._encoded[0][:, self._frame : self._frame + 1]
            if self._prediction is None:
                token = (
                    mx.array([[self._last_token]], dtype=mx.int32)
                    if self._last_token != self.model.blank_id
                    else None
                )
                self._prediction = self.model.decoder(token, self._hidden)
            output, (h, c) = self._prediction
            hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
            prediction = int(
                mx.argmax(self.model.joint(feature, output.astype(feature.dtype)))
            )
            if prediction != self.model.blank_id:
                self._last_token = prediction
                self._hidden = hidden
                self._prediction = None
                mx.eval(*hidden)
                text = tokenizer.decode([prediction], self.model.vocabulary)
                if not self._has_text:
                    text = text.lstrip()
                if text:
                    self._has_text = True
                    deltas.append(text)
                self._symbols += 1
            if prediction == self.model.blank_id or (
                self.model.max_symbols is not None
                and self._symbols >= self.model.max_symbols
            ):
                self._frame += 1
                self._symbols = 0
                if self._context is not None:
                    self._pending_frames -= 1
                if self._frame == self._encoded[0].shape[1]:
                    self._encoded.popleft()
                    self._frame = 0
        if self._flushed and not self._encoded:
            self._done = True
        return deltas
