"""Streaming inferencer and session API for MOSS-TTS-Realtime."""

from __future__ import annotations

import inspect
import re
from typing import Any, Iterable, Iterator, Optional, Sequence

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.moss_tts.sampling import resolve_channel_sampling_configs

from .config import ModelConfig
from .processor import MossTTSRealtimeProcessor


def _stack_left_padded_turns(
    turns: Sequence[mx.array],
    *,
    text_pad: int,
    audio_pad_token: int,
    rvq: int,
) -> mx.array:
    if not turns:
        raise ValueError("At least one turn input is required")

    channels = 1 + rvq
    max_len = max(int(turn.shape[0]) for turn in turns)
    padded: list[mx.array] = []
    for turn in turns:
        if turn.ndim != 2 or int(turn.shape[1]) != channels:
            raise ValueError(
                f"Each turn must have shape (T, {channels}), got {turn.shape}"
            )
        pad_len = max_len - int(turn.shape[0])
        if pad_len > 0:
            pad = mx.full((pad_len, channels), audio_pad_token, dtype=mx.int32)
            pad[:, 0] = text_pad
            turn = mx.concatenate([pad, turn.astype(mx.int32)], axis=0)
        padded.append(turn.astype(mx.int32))
    return mx.stack(padded, axis=0)


def _sanitize_audio_tokens(
    tokens: mx.array,
    *,
    audio_vocab_size: int,
    audio_eos_token: int,
) -> tuple[mx.array, bool]:
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    if tokens.ndim != 2:
        raise ValueError(f"Expected [T, RVQ] tokens, got {tokens.shape}")
    if int(tokens.shape[0]) == 0:
        return tokens.astype(mx.int32), False

    tokens_np = np.array(tokens)
    eos_rows = np.where(tokens_np[:, 0] == int(audio_eos_token))[0]
    invalid_rows = np.where(
        np.logical_or(tokens_np < 0, tokens_np >= int(audio_vocab_size)).any(axis=1)
    )[0]

    stop_idx = None
    if eos_rows.size > 0:
        stop_idx = int(eos_rows[0])
    if invalid_rows.size > 0:
        invalid_idx = int(invalid_rows[0])
        stop_idx = invalid_idx if stop_idx is None else min(stop_idx, invalid_idx)

    if stop_idx is not None:
        if stop_idx <= 0:
            return mx.zeros((0, int(tokens.shape[1])), dtype=mx.int32), True
        return mx.array(tokens_np[:stop_idx], dtype=mx.int32), True

    return tokens.astype(mx.int32), False


class MossTTSRealtimeInference:
    """Step-wise inferencer with explicit prefill/step/finish lifecycle."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        config: ModelConfig,
        max_length: int = 1000,
        max_context_tokens: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = int(max_length)
        self.max_context_tokens = (
            int(config.max_context_tokens)
            if max_context_tokens is None
            else int(max_context_tokens)
        )

        self.cache = None
        self.generated_tokens: list[mx.array] = []
        self.last_audio_tokens: Optional[mx.array] = None
        self.is_stopping: Optional[mx.array] = None
        self.step_idx = 0
        self.has_prefilled = False

    @property
    def is_finished(self) -> bool:
        return self.is_stopping is not None and bool(np.all(np.array(self.is_stopping)))

    def _normalize_input_ids(self, input_ids: Any) -> list[mx.array]:
        if isinstance(input_ids, mx.array):
            if input_ids.ndim == 2:
                return [input_ids.astype(mx.int32)]
            if input_ids.ndim == 3:
                return [input_ids[idx].astype(mx.int32) for idx in range(int(input_ids.shape[0]))]
        if isinstance(input_ids, (list, tuple)):
            normalized: list[mx.array] = []
            for entry in input_ids:
                arr = mx.array(entry, dtype=mx.int32)
                if arr.ndim != 2:
                    raise ValueError("Each input_ids entry must be 2D")
                normalized.append(arr)
            if normalized:
                return normalized
        raise ValueError(
            "input_ids must be shape [T, C], [B, T, C], or a list of [T, C] arrays"
        )

    def _normalize_text_prefix_ids(
        self,
        text_prefix_ids: Any,
        *,
        batch_size: int,
    ) -> list[list[int]]:
        if text_prefix_ids is None:
            raise ValueError("text_prefix_ids is required")

        if isinstance(text_prefix_ids, mx.array):
            text_prefix_ids = np.array(text_prefix_ids).tolist()
        elif isinstance(text_prefix_ids, np.ndarray):
            text_prefix_ids = text_prefix_ids.tolist()

        if isinstance(text_prefix_ids, list):
            if not text_prefix_ids:
                return [[] for _ in range(batch_size)]
            if isinstance(text_prefix_ids[0], (int, np.integer)):
                prefix = [int(token) for token in text_prefix_ids]
                if batch_size > 1:
                    return [list(prefix) for _ in range(batch_size)]
                return [prefix]
            if len(text_prefix_ids) == 1 and batch_size > 1:
                return [list(int(token) for token in text_prefix_ids[0]) for _ in range(batch_size)]
            if len(text_prefix_ids) != batch_size:
                raise ValueError(
                    f"text_prefix_ids batch mismatch: {len(text_prefix_ids)} vs {batch_size}"
                )
            return [list(int(token) for token in row) for row in text_prefix_ids]

        raise ValueError("text_prefix_ids must be list-like or tensor-like")

    def _build_sampling_cfg(
        self,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: float,
    ):
        return resolve_channel_sampling_configs(
            self.config.rvq,
            default_temperature=float(temperature),
            default_top_p=float(top_p),
            default_top_k=int(top_k),
            default_repetition_penalty=float(repetition_penalty),
            do_samples=[bool(do_sample)] * self.config.rvq,
        )

    def _ensure_cache_capacity(self, next_tokens: int):
        if self.cache is None:
            return
        if not self.cache:
            return
        current_offset = int(getattr(self.cache[0], "offset", 0))
        if current_offset + int(next_tokens) <= self.max_context_tokens:
            return
        # Fail-safe bound for long-running sessions: drop stale cache once the cap is hit.
        self.cache = self.model.make_cache()

    def reset_generation_state(self, keep_cache: bool = True):
        if not keep_cache:
            self.cache = None
        self.generated_tokens = []
        self.last_audio_tokens = None
        self.is_stopping = None
        self.step_idx = 0
        self.has_prefilled = False

    def reset_turn(
        self,
        user_text: Optional[str],
        user_audio_tokens: Optional[mx.array],
        include_system_prompt: bool,
        reset_cache: bool,
    ):
        del user_text, user_audio_tokens, include_system_prompt
        self.reset_generation_state(keep_cache=not bool(reset_cache))

    def prefill(
        self,
        *,
        input_ids: Any,
        text_prefix_ids: Any,
        max_prefill_len: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        repetition_window: Optional[int] = 50,
    ) -> mx.array:
        turns = self._normalize_input_ids(input_ids)
        batch_size = len(turns)
        prefixes = self._normalize_text_prefix_ids(
            text_prefix_ids,
            batch_size=batch_size,
        )

        merged_turns: list[mx.array] = []
        for turn, prefix in zip(turns, prefixes):
            current_prefix = list(prefix)
            if max_prefill_len is not None:
                current_prefix = current_prefix[: int(max_prefill_len)]
            if not current_prefix:
                raise ValueError("Prefill requires at least one text token")

            prefix_rows = mx.full(
                (len(current_prefix), self.config.channels),
                self.config.audio_pad_token,
                dtype=mx.int32,
            )
            prefix_rows[:, 0] = mx.array(current_prefix, dtype=mx.int32)
            prefix_rows[-1, 1] = int(self.config.audio_bos_token)
            merged_turns.append(mx.concatenate([turn, prefix_rows], axis=0))

        stacked_turns = _stack_left_padded_turns(
            merged_turns,
            text_pad=self.config.text_pad,
            audio_pad_token=self.config.audio_pad_token,
            rvq=self.config.rvq,
        )

        if self.cache is None:
            self.cache = self.model.make_cache()
        self._ensure_cache_capacity(int(stacked_turns.shape[1]))

        hidden_states = self.model(stacked_turns, cache=self.cache)
        global_hidden = hidden_states[:, -1, :]

        sampling_cfg = self._build_sampling_cfg(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        audio_tokens = self.model.sample_next_audio_tokens(
            global_hidden,
            generated_history=None,
            channel_sampling=sampling_cfg,
            repetition_window=repetition_window,
        )

        self.generated_tokens = [audio_tokens]
        self.last_audio_tokens = audio_tokens
        self.is_stopping = audio_tokens[:, 0] == int(self.config.audio_eos_token)
        self.step_idx = 1
        self.has_prefilled = True
        return audio_tokens

    def step(
        self,
        text_token: Optional[Iterable[int] | mx.array | np.ndarray | int],
        *,
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        repetition_window: Optional[int] = 50,
    ) -> mx.array:
        if not self.has_prefilled or self.last_audio_tokens is None:
            raise ValueError("prefill() must be called before step()")
        if self.is_finished:
            return self.last_audio_tokens

        batch_size = int(self.last_audio_tokens.shape[0])
        if text_token is None:
            text_tokens = [self.config.text_pad] * batch_size
        elif isinstance(text_token, mx.array):
            text_tokens = np.array(text_token).astype(np.int32).tolist()
        elif isinstance(text_token, np.ndarray):
            text_tokens = text_token.astype(np.int32).tolist()
        elif isinstance(text_token, (list, tuple)):
            text_tokens = [int(token) for token in text_token]
        else:
            text_tokens = [int(text_token)]

        if len(text_tokens) != batch_size:
            raise ValueError(
                f"text_token batch mismatch: {len(text_tokens)} vs {batch_size}"
            )

        step_ids = mx.full(
            (batch_size, self.config.channels),
            self.config.audio_pad_token,
            dtype=mx.int32,
        )
        step_ids[:, 0] = mx.array(text_tokens, dtype=mx.int32)
        step_ids[:, 1 : 1 + self.config.rvq] = self.last_audio_tokens[:, : self.config.rvq]

        if self.cache is None:
            self.cache = self.model.make_cache()
        self._ensure_cache_capacity(1)

        hidden_states = self.model(step_ids[:, None, :], cache=self.cache)
        global_hidden = hidden_states[:, -1, :]

        sampling_cfg = self._build_sampling_cfg(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        history = (
            mx.stack(self.generated_tokens, axis=1)
            if self.generated_tokens
            else None
        )
        audio_tokens = self.model.sample_next_audio_tokens(
            global_hidden,
            generated_history=history,
            channel_sampling=sampling_cfg,
            repetition_window=repetition_window,
        )

        self.generated_tokens.append(audio_tokens)
        self.last_audio_tokens = audio_tokens
        if self.is_stopping is None:
            self.is_stopping = audio_tokens[:, 0] == int(self.config.audio_eos_token)
        else:
            self.is_stopping = self.is_stopping | (
                audio_tokens[:, 0] == int(self.config.audio_eos_token)
            )
        self.step_idx += 1
        return audio_tokens

    def finish(
        self,
        *,
        max_steps: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        repetition_window: Optional[int] = 50,
    ) -> list[mx.array]:
        outputs: list[mx.array] = []
        steps_left = self.max_length if max_steps is None else int(max_steps)
        while steps_left > 0 and not self.is_finished:
            outputs.append(
                self.step(
                    None,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    repetition_window=repetition_window,
                )
            )
            steps_left -= 1
        return outputs


class AudioStreamDecoder:
    """Decode realtime audio token rows into waveform chunks."""

    def __init__(
        self,
        *,
        processor: MossTTSRealtimeProcessor,
        chunk_frames: int = 40,
        overlap_frames: int = 4,
        decode_kwargs: Optional[dict[str, Any]] = None,
        max_pending_frames: int = 4096,
    ):
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if overlap_frames < 0:
            raise ValueError("overlap_frames must be non-negative")
        if overlap_frames >= chunk_frames:
            raise ValueError("overlap_frames must be smaller than chunk_frames")
        if max_pending_frames <= 0:
            raise ValueError("max_pending_frames must be positive")

        self.processor = processor
        self.chunk_frames = int(chunk_frames)
        self.overlap_frames = int(overlap_frames)
        self.decode_kwargs = {} if decode_kwargs is None else dict(decode_kwargs)
        self.max_pending_frames = int(max_pending_frames)

        self._buffer: list[mx.array] = []
        self._buffer_len = 0
        self._previous_tail: Optional[mx.array] = None

    @property
    def pending_frames(self) -> int:
        return int(self._buffer_len)

    def reset(self):
        self._buffer = []
        self._buffer_len = 0
        self._previous_tail = None

    def push_tokens(self, audio_tokens: mx.array):
        tokens = audio_tokens
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        if tokens.ndim == 3 and int(tokens.shape[0]) == 1:
            tokens = tokens[0]
        if tokens.ndim != 2:
            raise ValueError(f"Expected audio tokens shape [T, RVQ], got {tokens.shape}")
        if int(tokens.shape[1]) != self.processor.model_config.rvq:
            raise ValueError(
                "Unexpected RVQ width in pushed tokens: "
                f"{tokens.shape[1]} vs {self.processor.model_config.rvq}"
            )

        next_len = self._buffer_len + int(tokens.shape[0])
        if next_len > self.max_pending_frames:
            raise RuntimeError(
                "Pending realtime audio-token buffer overflow: "
                f"{next_len} > {self.max_pending_frames}"
            )

        self._buffer.append(tokens.astype(mx.int32))
        self._buffer_len = next_len

    def audio_chunks(self) -> list[mx.array]:
        chunks: list[mx.array] = []
        while self._buffer_len >= self.chunk_frames:
            chunk_tokens = self._consume_frames(self.chunk_frames)
            decoded = self._decode(chunk_tokens, chunk_duration=0.32)
            chunks.append(self._apply_crossfade(decoded, final_chunk=False))
        return chunks

    def flush(self) -> Optional[mx.array]:
        if self._buffer_len <= 0:
            return None
        chunk_tokens = self._consume_frames(self._buffer_len)
        decoded = self._decode(chunk_tokens, chunk_duration=None)
        return self._apply_crossfade(decoded, final_chunk=True)

    def _consume_frames(self, num_frames: int) -> mx.array:
        remaining = int(num_frames)
        chunks: list[mx.array] = []
        while remaining > 0 and self._buffer:
            head = self._buffer[0]
            head_len = int(head.shape[0])
            if head_len <= remaining:
                chunks.append(head)
                remaining -= head_len
                self._buffer.pop(0)
            else:
                chunks.append(head[:remaining, :])
                self._buffer[0] = head[remaining:, :]
                remaining = 0

        consumed = int(num_frames) - remaining
        self._buffer_len -= consumed
        if consumed <= 0:
            return mx.zeros((0, self.processor.model_config.rvq), dtype=mx.int32)
        return mx.concatenate(chunks, axis=0)

    def _decode(self, tokens: mx.array, chunk_duration: Optional[float]) -> mx.array:
        decode_kwargs = dict(self.decode_kwargs)
        chunk_override = decode_kwargs.pop("chunk_duration", None)
        resolved_chunk_duration = chunk_duration
        if chunk_override is not None:
            override = float(chunk_override)
            resolved_chunk_duration = None if override <= 0 else override

        return self.processor.decode_audio_codes(
            tokens.astype(mx.int32),
            chunk_duration=resolved_chunk_duration,
            decode_kwargs=decode_kwargs,
        )

    def _apply_crossfade(self, wav: mx.array, *, final_chunk: bool) -> mx.array:
        if self.overlap_frames <= 0:
            return wav

        overlap = self._overlap_samples(wav)
        if overlap <= 0:
            return wav

        if self._previous_tail is None:
            if not final_chunk:
                self._previous_tail = self._copy_tail(wav, overlap)
            return wav

        prev_np = np.array(self._previous_tail)
        wav_np = np.array(wav)
        overlap = min(overlap, prev_np.shape[0], wav_np.shape[0])
        if overlap <= 0:
            if not final_chunk:
                self._previous_tail = self._copy_tail(
                    wav,
                    self._overlap_samples(wav),
                )
            else:
                self._previous_tail = None
            return wav

        fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        fade_in = 1.0 - fade_out
        cross = prev_np[-overlap:] * fade_out + wav_np[:overlap] * fade_in
        # Only emit samples aligned to the current decode; previous tail provides
        # blend context but must not be prepended or we duplicate stale audio.
        merged = np.concatenate([cross, wav_np[overlap:]], axis=0)

        if final_chunk:
            self._previous_tail = None
        else:
            self._previous_tail = self._copy_tail(wav, overlap)
        return mx.array(merged, dtype=wav.dtype)

    def _copy_tail(self, wav: mx.array, overlap: int) -> mx.array:
        tail_size = max(0, int(overlap))
        if tail_size <= 0:
            return mx.zeros((0,), dtype=wav.dtype)
        # MLX arrays do not implement `.copy()`.
        return mx.array(np.array(wav[-tail_size:], copy=True), dtype=wav.dtype)

    def _overlap_samples(self, wav: mx.array) -> int:
        if self.chunk_frames <= 0:
            return 0
        return int(int(wav.shape[0]) * (self.overlap_frames / self.chunk_frames))


class TextDeltaTokenizer:
    """Stateful delta->token bridge that preserves tokenizer consistency."""

    def __init__(self, tokenizer: Any, *, hold_back: int = 0):
        self.tokenizer = tokenizer
        self.hold_back = max(0, int(hold_back))
        self._text = ""
        self._all_ids: list[int] = []
        self._emitted_count = 0

    def push_delta(self, delta: str) -> list[int]:
        if not delta:
            return []
        self._text += str(delta)
        try:
            self._all_ids = list(
                self.tokenizer.encode(self._text, add_special_tokens=False)
            )
        except TypeError:
            self._all_ids = list(self.tokenizer.encode(self._text))

        stable_count = max(self._emitted_count, len(self._all_ids) - self.hold_back)
        new_ids = self._all_ids[self._emitted_count : stable_count]
        self._emitted_count = stable_count
        return [int(token) for token in new_ids]

    def flush(self) -> list[int]:
        try:
            self._all_ids = list(
                self.tokenizer.encode(self._text, add_special_tokens=False)
            )
        except TypeError:
            self._all_ids = list(self.tokenizer.encode(self._text))

        remaining = self._all_ids[self._emitted_count :]
        self._emitted_count = len(self._all_ids)
        return [int(token) for token in remaining]


class RealtimeTextDeltaBridge:
    """Explicit text-delta bridge for realtime ingestion parity tests."""

    def __init__(self, session: "RealtimeSession", *, hold_back: int = 0):
        self.session = session
        self.delta_tokenizer = TextDeltaTokenizer(
            session.processor.tokenizer,
            hold_back=hold_back,
        )

    def push_text_delta(self, delta: str) -> list[mx.array]:
        token_ids = self.delta_tokenizer.push_delta(delta)
        return self.session.push_text_tokens(token_ids)

    def push_text_tokens(self, token_ids: Sequence[int]) -> list[mx.array]:
        return self.session.push_text_tokens(token_ids)

    def end_text(self) -> list[mx.array]:
        token_ids = self.delta_tokenizer.flush()
        chunks: list[mx.array] = []
        if token_ids:
            chunks.extend(self.session.push_text_tokens(token_ids))
        chunks.extend(self.session.end_text())
        return chunks

    def drain(self, *, max_steps: Optional[int] = None) -> list[mx.array]:
        return self.session.drain(max_steps=max_steps)


class RealtimeSession:
    """Single-conversation realtime session with explicit lifecycle."""

    _split_pattern = re.compile(
        r"[。！？!?\.\u2026]\s*|[,，;；:：\u2014\u2013\-]\s*|\)\s*|\]\s*|\n"
    )

    def __init__(
        self,
        *,
        inferencer: MossTTSRealtimeInference,
        processor: MossTTSRealtimeProcessor,
        chunk_frames: int = 40,
        overlap_frames: int = 4,
        decode_kwargs: Optional[dict[str, Any]] = None,
        max_pending_frames: int = 4096,
        prefill_text_len: int = 12,
        text_buffer_size: int = 32,
        min_text_chunk_chars: int = 8,
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        repetition_window: Optional[int] = 50,
    ):
        self.inferencer = inferencer
        self.processor = processor

        self.decoder = AudioStreamDecoder(
            processor=processor,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
            decode_kwargs=decode_kwargs,
            max_pending_frames=max_pending_frames,
        )

        self.prefill_text_len = int(prefill_text_len)
        self.text_buffer_size = int(text_buffer_size)
        self.min_text_chunk_chars = int(min_text_chunk_chars)

        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.do_sample = bool(do_sample)
        self.repetition_penalty = float(repetition_penalty)
        self.repetition_window = (
            None
            if repetition_window is None or int(repetition_window) <= 0
            else int(repetition_window)
        )

        self._turn_input_ids: Optional[mx.array] = None
        self._turn_index = 0
        self._voice_prompt_tokens: Optional[mx.array] = None
        self._text_cache = ""
        self._pending_tokens: list[int] = []
        self._prefilled = False
        self._text_ended = False
        self._turn_drained = True
        self._closed = False

    def _assert_open(self):
        if self._closed:
            raise RuntimeError("RealtimeSession is closed")

    def set_voice_prompt_tokens(self, audio_tokens: Optional[mx.array]):
        self._assert_open()
        if audio_tokens is None:
            self._voice_prompt_tokens = None
            return
        if hasattr(self.processor, "normalize_audio_prompt_tokens"):
            self._voice_prompt_tokens = self.processor.normalize_audio_prompt_tokens(
                audio_tokens
            )
        else:
            self._voice_prompt_tokens = mx.array(audio_tokens, dtype=mx.int32)

    def set_voice_prompt_audio(self, audio: Any):
        self._assert_open()
        if not hasattr(self.processor, "encode_prompt_audio"):
            raise RuntimeError("processor does not support prompt-audio encoding")
        self._voice_prompt_tokens = self.processor.encode_prompt_audio(audio)

    def clear_voice_prompt_tokens(self):
        self._assert_open()
        self._voice_prompt_tokens = None

    def reset_turn(
        self,
        user_text: Optional[str],
        user_audio_tokens: Optional[Any] = None,
        include_system_prompt: Optional[bool] = None,
        reset_cache: bool = False,
        input_ids: Optional[mx.array] = None,
    ):
        self._assert_open()

        if (
            self._prefilled
            and not self._turn_drained
            and not self.inferencer.is_finished
        ):
            raise RuntimeError(
                "drain() must be called before reset_turn() in an active turn"
            )

        if include_system_prompt is None:
            include_system_prompt = self._turn_index == 0
            if not include_system_prompt and bool(reset_cache):
                include_system_prompt = self._voice_prompt_tokens is not None

        if input_ids is None:
            build_turn = self.processor.build_turn_input_ids
            params = inspect.signature(build_turn).parameters
            if "voice_prompt_tokens" in params:
                input_ids = build_turn(
                    user_text="" if user_text is None else str(user_text),
                    user_audio_tokens=user_audio_tokens,
                    include_system_prompt=bool(include_system_prompt),
                    voice_prompt_tokens=self._voice_prompt_tokens,
                )
            else:
                input_ids = build_turn(
                    user_text="" if user_text is None else str(user_text),
                    user_audio_tokens=user_audio_tokens,
                    include_system_prompt=bool(include_system_prompt),
                )

        if input_ids.ndim != 3 or int(input_ids.shape[2]) != self.processor.model_config.channels:
            raise ValueError(
                "reset_turn input_ids must have shape [B, T, channels], "
                f"got {input_ids.shape}"
            )

        self._turn_input_ids = input_ids.astype(mx.int32)
        self._turn_index += 1

        self._text_cache = ""
        self._pending_tokens = []
        self._prefilled = False
        self._text_ended = False
        self._turn_drained = False

        self.decoder.reset()
        self.inferencer.reset_turn(
            user_text=user_text,
            user_audio_tokens=user_audio_tokens,
            include_system_prompt=bool(include_system_prompt),
            reset_cache=bool(reset_cache),
        )

    def push_text_tokens(self, token_ids: Iterable[int]) -> list[mx.array]:
        self._assert_open()
        if self._turn_input_ids is None:
            raise RuntimeError("reset_turn() must be called before push_text_tokens()")
        self._pending_tokens.extend([int(token) for token in token_ids])
        return self._drain_pending_tokens()

    def push_text(self, text_fragment: str) -> list[mx.array]:
        self._assert_open()
        if self._turn_input_ids is None:
            raise RuntimeError("reset_turn() must be called before push_text()")

        self._text_cache += str(text_fragment)
        segments = self._extract_text_segments(force=False)
        for segment in segments:
            self._pending_tokens.extend(self._tokenize(segment))
        return self._drain_pending_tokens()

    def end_text(self) -> list[mx.array]:
        self._assert_open()
        if self._turn_input_ids is None:
            raise RuntimeError("reset_turn() must be called before end_text()")

        self._text_ended = True
        if self._text_cache:
            self._pending_tokens.extend(self._tokenize(self._text_cache))
            self._text_cache = ""
        return self._drain_pending_tokens()

    def drain(self, max_steps: Optional[int] = None) -> list[mx.array]:
        self._assert_open()
        if self._turn_input_ids is None:
            raise RuntimeError("reset_turn() must be called before drain()")

        chunks: list[mx.array] = []
        if self._prefilled:
            frames = self.inferencer.finish(
                max_steps=max_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=self.do_sample,
                repetition_penalty=self.repetition_penalty,
                repetition_window=self.repetition_window,
            )
            chunks.extend(self._decode_audio_frames(frames))

        tail = self.decoder.flush()
        if tail is not None and int(tail.shape[0]) > 0:
            chunks.append(tail)

        self._turn_drained = True
        return chunks

    def reset(self, *, reset_cache: bool = False):
        self._assert_open()
        if (
            self._prefilled
            and not self._turn_drained
            and not self.inferencer.is_finished
        ):
            raise RuntimeError(
                "end_text() + drain() are required before reset() in active turns"
            )

        self.inferencer.reset_generation_state(keep_cache=not bool(reset_cache))
        self._turn_input_ids = None
        self._text_cache = ""
        self._pending_tokens = []
        self._prefilled = False
        self._text_ended = False
        self._turn_drained = True
        self.decoder.reset()

    def close(self):
        if self._closed:
            return

        if (
            self._prefilled
            and not self._turn_drained
            and not self.inferencer.is_finished
        ):
            self.end_text()
            self.drain()

        self.inferencer.reset_generation_state(keep_cache=False)
        self.decoder.reset()
        self._turn_input_ids = None
        self._pending_tokens = []
        self._text_cache = ""
        self._prefilled = False
        self._text_ended = False
        self._turn_drained = True
        self._voice_prompt_tokens = None
        self._closed = True

    def bridge_text_stream(
        self,
        text_deltas: Iterable[str],
        *,
        hold_back: int = 0,
        drain_step: int = 1,
    ) -> Iterator[mx.array]:
        bridge = RealtimeTextDeltaBridge(self, hold_back=hold_back)
        for delta in text_deltas:
            for chunk in bridge.push_text_delta(delta):
                yield chunk
        for chunk in bridge.end_text():
            yield chunk

        while True:
            chunks = bridge.drain(max_steps=drain_step)
            if not chunks:
                break
            for chunk in chunks:
                yield chunk
            if self.inferencer.is_finished:
                break

    def _extract_text_segments(self, force: bool) -> list[str]:
        segments: list[str] = []
        if force:
            if self._text_cache:
                segments.append(self._text_cache)
                self._text_cache = ""
            return segments

        while self._text_cache:
            cut_idx = None
            if len(self._text_cache) >= self.min_text_chunk_chars:
                matches = list(self._split_pattern.finditer(self._text_cache))
                for match in matches:
                    if match.end() >= self.min_text_chunk_chars:
                        cut_idx = match.end()
                        break
            if cut_idx is None and len(self._text_cache) >= self.text_buffer_size:
                whitespace_idx = self._text_cache.rfind(" ")
                if whitespace_idx != -1:
                    cut_idx = whitespace_idx + 1
            if cut_idx is None:
                break
            segments.append(self._text_cache[:cut_idx])
            self._text_cache = self._text_cache[cut_idx:]

        return segments

    def _tokenize(self, text: str) -> list[int]:
        try:
            return list(self.processor.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.processor.tokenizer.encode(text))

    def _prefill_if_needed(self) -> list[mx.array]:
        if self._prefilled:
            return []
        if self._turn_input_ids is None:
            raise RuntimeError("reset_turn() must be called before streaming text")
        if not self._pending_tokens and not self._text_ended:
            return []
        if len(self._pending_tokens) < self.prefill_text_len and not self._text_ended:
            return []

        prefill_len = (
            len(self._pending_tokens)
            if self._text_ended
            else min(len(self._pending_tokens), self.prefill_text_len)
        )
        if prefill_len <= 0:
            return []

        prefix_tokens = [self._pending_tokens.pop(0) for _ in range(prefill_len)]
        first_audio = self.inferencer.prefill(
            input_ids=self._turn_input_ids,
            text_prefix_ids=[prefix_tokens],
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            repetition_window=self.repetition_window,
        )
        self._prefilled = True
        return self._decode_audio_frames([first_audio])

    def _drain_pending_tokens(self) -> list[mx.array]:
        chunks: list[mx.array] = []
        chunks.extend(self._prefill_if_needed())

        if not self._prefilled:
            return chunks

        while self._pending_tokens and not self.inferencer.is_finished:
            token_id = self._pending_tokens.pop(0)
            frame = self.inferencer.step(
                token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=self.do_sample,
                repetition_penalty=self.repetition_penalty,
                repetition_window=self.repetition_window,
            )
            chunks.extend(self._decode_audio_frames([frame]))
        return chunks

    def _decode_audio_frames(self, audio_frames: Sequence[mx.array]) -> list[mx.array]:
        chunks: list[mx.array] = []
        for frame in audio_frames:
            tokens = frame
            if tokens.ndim == 3 and int(tokens.shape[0]) == 1:
                tokens = tokens[0]
            if tokens.ndim == 1:
                tokens = tokens[None, :]

            if tokens.ndim != 2:
                raise ValueError(
                    f"Expected [T, RVQ] frame tokens, got {tokens.shape}"
                )

            sanitized, stop = _sanitize_audio_tokens(
                tokens,
                audio_vocab_size=self.processor.model_config.audio_vocab_size,
                audio_eos_token=self.processor.model_config.audio_eos_token,
            )
            if int(sanitized.shape[0]) > 0:
                self.decoder.push_tokens(sanitized)
                chunks.extend(self.decoder.audio_chunks())
            if stop:
                break

        return chunks


def bridge_text_stream(
    session: RealtimeSession,
    text_deltas: Iterable[str],
    *,
    hold_back: int = 0,
    drain_step: int = 1,
) -> Iterator[mx.array]:
    """Top-level helper that streams WAV chunks from text deltas."""

    yield from session.bridge_text_stream(
        text_deltas,
        hold_back=hold_back,
        drain_step=drain_step,
    )


__all__ = [
    "AudioStreamDecoder",
    "MossTTSRealtimeInference",
    "RealtimeSession",
    "RealtimeTextDeltaBridge",
    "TextDeltaTokenizer",
    "bridge_text_stream",
]
