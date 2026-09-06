from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from mlx_audio.codec.models.nemotron_voicechat import CausalConv1dCache
from mlx_audio.stt.models.nemotron_asr import (
    ConformerStreamingState,
    StreamingLogMelSpectrogram,
)
from mlx_audio.stt.models.nemotron_asr import tokenizer as rnnt_tokenizer
from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram

VoiceChatEventKind = Literal[
    "assistant_text_delta",
    "function_delta",
    "user_transcript_delta",
    "audio",
    "done",
    "cancelled",
]


@dataclass
class VoiceChatEvent:
    kind: VoiceChatEventKind
    frame_index: int | None = None
    token_id: int | None = None
    delta: str | None = None
    text: str | None = None
    samples: mx.array | None = None
    sample_rate: int | None = None
    audio_codes: mx.array | None = None


class VoiceChatContextLimitError(RuntimeError):
    pass


class _TokenAccumulator:
    def __init__(self, tokenizer, special_ids: set[int]):
        self.tokenizer = tokenizer
        self.special_ids = special_ids
        self.tokens: list[int] = []
        self.text = ""

    def append(self, token_id: int) -> tuple[str, str] | None:
        if token_id in self.special_ids:
            return None
        self.tokens.append(token_id)
        updated = self.tokenizer.decode(self.tokens, skip_special_tokens=False)
        delta = (
            updated[len(self.text) :]
            if updated.startswith(self.text)
            else self.tokenizer.decode([token_id], skip_special_tokens=False)
        )
        self.text = updated
        return delta, updated


class _RNNTState:
    def __init__(self, session):
        self.session = session
        self.last_token = session.config.rnnt_blank_id
        self.hidden = None
        self.tokens: list[int] = []
        self.text = ""

    def step(self, encoded: mx.array) -> tuple[str, str] | None:
        config = self.session.config
        if not config.rnnt_vocabulary:
            return None
        decoder = self.session.model.stt_model.rnnt_decoder
        joint = self.session.model.stt_model.rnnt_joint
        previous_count = len(self.tokens)
        new_symbols = 0
        while True:
            token = (
                mx.array([[self.last_token]], dtype=mx.int32)
                if self.last_token != config.rnnt_blank_id
                else None
            )
            prediction, proposed_hidden = decoder(token, self.hidden)
            logits = joint(encoded, prediction.astype(encoded.dtype))
            next_token = int(mx.argmax(logits))
            if next_token == config.rnnt_blank_id:
                break
            self.last_token = next_token
            self.hidden = tuple(
                value.astype(encoded.dtype) for value in proposed_hidden
            )
            if not rnnt_tokenizer.is_special_token(next_token, config.rnnt_vocabulary):
                self.tokens.append(next_token)
            new_symbols += 1
            if new_symbols >= config.rnnt_max_symbols:
                break
        if len(self.tokens) == previous_count:
            return None
        updated = rnnt_tokenizer.decode(self.tokens, config.rnnt_vocabulary).strip()
        delta = updated[len(self.text) :] if updated.startswith(self.text) else updated
        self.text = updated
        return delta, updated


class VoiceChatStreamingSession:
    def __init__(
        self,
        parent,
        *,
        system_prompt: str | None = None,
        seed: int = 0,
        max_streaming_seconds: float | None = None,
        use_language_cache: bool = True,
        use_perception_cache: bool = True,
    ):
        if max_streaming_seconds is not None and max_streaming_seconds <= 0:
            raise ValueError("max_streaming_seconds must be positive")
        self.parent = parent
        self.model = parent.model
        self.tokenizer = parent.tokenizer
        self.config = self.model.config
        self.input_sample_rate = self.config.source_sample_rate
        self.output_sample_rate = self.config.target_sample_rate
        self.frame_samples = round(self.input_sample_rate * self.config.frame_duration)
        self.max_frames = (
            None
            if max_streaming_seconds is None
            else int(max_streaming_seconds / self.config.frame_duration)
        )
        self._pending_audio = mx.zeros((0,), dtype=mx.float32)
        self._audio_window = mx.zeros((0,), dtype=mx.float32)
        left, right = self.config.encoder.att_context_size[0]
        self._perception_window_frames = max(2, left + right + 1)
        self._language_cache = (
            self.model.stt_model.make_cache() if use_language_cache else None
        )
        self._mel_stream = None
        self._conformer_stream = None
        if use_perception_cache:
            self._mel_stream = StreamingLogMelSpectrogram(
                self.config.preprocessor,
                lookahead_samples=self.frame_samples,
            )
            self._conformer_stream = ConformerStreamingState(
                self.model.stt_model.perception.encoder,
                chunk_frames=1,
                att_context_size=[left, right],
            )
        self._input_history: list[mx.array] = []
        self._text_tokens: list[int] = []
        self._function_tokens: list[int] = []
        self._frame_index = 0
        self._timeline_index = 0
        self._closed = False
        special_ids = {
            self.config.pad_token_id,
            self.config.silence_token_id,
            self.config.bos_token_id,
            self.config.eos_token_id,
        }
        self._text = _TokenAccumulator(self.tokenizer, special_ids)
        self._function = _TokenAccumulator(self.tokenizer, special_ids)
        self._rnnt = _RNNTState(self)
        self._codec_cache = CausalConv1dCache()
        mx.random.seed(seed)
        prompt = self.parent._tts_prompt()
        self._previous_code, self._tts_cache = self.model.tts_model.tts_model.warmup(
            *prompt,
            guidance_enabled=True,
        )
        self._prefill_prompt(system_prompt)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def _language_step(self, inputs: mx.array):
        if self._language_cache is not None:
            return self.model.stt_model(inputs, cache=self._language_cache)
        self._input_history.append(inputs)
        return self.model.stt_model(mx.concatenate(self._input_history, axis=1))

    def _prefill_prompt(self, system_prompt: str | None) -> None:
        prompt_text = (
            self.config.default_system_prompt
            if system_prompt is None
            else system_prompt
        )
        if not prompt_text.strip():
            return
        prompt_ids = [
            self.config.bos_token_id,
            *self.tokenizer.encode(prompt_text, add_special_tokens=False),
            self.config.eos_token_id,
        ]
        prompt_embeds = self.model.stt_model.embed_tokens(
            mx.array([prompt_ids], dtype=mx.int32)
        )
        for index in range(len(prompt_ids)):
            self._run_timeline_step(
                prompt_embeds[:, index : index + 1],
                generate_channels=False,
                decode_audio=False,
            )

    def _perception_step(self, frame: mx.array) -> tuple[mx.array, mx.array]:
        if self._mel_stream is not None and self._conformer_stream is not None:
            mel = self._mel_stream.push(frame)
            chunks = self._conformer_stream.push(mel, emit_partial=True)
            if len(chunks) != 1 or chunks[0].shape[1] != 1:
                shapes = [tuple(chunk.shape) for chunk in chunks]
                raise RuntimeError(
                    "cached perception did not emit one encoder frame: "
                    f"mel={tuple(mel.shape)}, encoded={shapes}"
                )
            encoded = chunks[0]
            projected = self.model.stt_model.perception.proj(encoded)
            self._conformer_stream.materialize(projected, encoded)
            return projected, encoded

        self._audio_window = mx.concatenate([self._audio_window, frame])
        max_samples = self._perception_window_frames * self.frame_samples
        if self._audio_window.shape[0] > max_samples:
            self._audio_window = self._audio_window[-max_samples:]
        mel = log_mel_spectrogram(self._audio_window, self.config.preprocessor)
        mel_lengths = mx.array([mel.shape[1]], dtype=mx.int32)
        projected, _, encoded = self.model.stt_model.perception(mel, mel_lengths)
        if projected.shape[1] < 2:
            raise RuntimeError(
                "perception encoder returned fewer than two frames for one input frame"
            )
        projected = projected[:, -2:-1]
        encoded = encoded[:, -2:-1]
        mx.eval(projected, encoded)
        return projected, encoded

    def _run_timeline_step(
        self,
        audio_embedding: mx.array,
        *,
        generate_channels: bool,
        decode_audio: bool,
    ) -> list[VoiceChatEvent]:
        pad_id = self.config.pad_token_id
        previous_text_id = (
            pad_id if self._timeline_index == 0 else self._text_tokens[-1]
        )
        previous_function_id = (
            pad_id if self._timeline_index == 0 else self._function_tokens[-1]
        )
        previous_text = mx.array([[previous_text_id]], dtype=mx.int32)
        previous_function = mx.array([[previous_function_id]], dtype=mx.int32)
        fused = (
            self.config.text_channel_weight
            * self.model.stt_model.embed_tokens(previous_text)
            + self.config.audio_channel_weight * audio_embedding
            + self.config.function_channel_weight
            * self.model.stt_model.embed_tokens(previous_function)
        )
        output = self._language_step(fused)
        text_id = (
            int(mx.argmax(output.text_logits[:, -1])) if generate_channels else pad_id
        )
        function_id = (
            int(mx.argmax(output.function_logits[:, -1]))
            if generate_channels
            else pad_id
        )
        self._text_tokens.append(text_id)
        self._function_tokens.append(function_id)
        if self._timeline_index > 0:
            current = mx.array([[text_id]], dtype=mx.int32)
            self._previous_code, self._tts_cache = (
                self.model.tts_model.tts_model.generate_step(
                    current,
                    self._previous_code,
                    self._tts_cache,
                    text_eos_id=self.config.eos_token_id,
                    silence_codes=self.model.tts_model.codec_silence_tokens[None, None],
                    guidance_enabled=True,
                )
            )
            code = self._previous_code
        else:
            code = self.model.tts_model.codec_silence_tokens[None, None, :]
        self._timeline_index += 1
        mx.eval(code, output.text_logits, output.function_logits)
        if not generate_channels:
            return []

        events: list[VoiceChatEvent] = []
        text_update = self._text.append(text_id)
        if text_update is not None:
            delta, text = text_update
            events.append(
                VoiceChatEvent(
                    kind="assistant_text_delta",
                    frame_index=self._frame_index,
                    token_id=text_id,
                    delta=delta,
                    text=text,
                )
            )
        function_update = self._function.append(function_id)
        if function_update is not None:
            delta, text = function_update
            events.append(
                VoiceChatEvent(
                    kind="function_delta",
                    frame_index=self._frame_index,
                    token_id=function_id,
                    delta=delta,
                    text=text,
                )
            )
        if decode_audio:
            clean_code = self.parent._replace_control_codes(code)
            samples = self.model.tts_model.audio_codec.decode_step(
                clean_code.transpose(0, 2, 1), self._codec_cache
            )[0, 0]
            mx.eval(samples)
            expected = self.model.tts_model.audio_codec.waveform_to_token_ratio
            if samples.shape[0] != expected:
                raise RuntimeError(
                    f"streaming codec produced {samples.shape[0]} samples, expected {expected}"
                )
            events.append(
                VoiceChatEvent(
                    kind="audio",
                    frame_index=self._frame_index,
                    samples=samples,
                    sample_rate=self.output_sample_rate,
                    audio_codes=clean_code[0, 0],
                )
            )
        return events

    def _step_audio_frame(self, frame: mx.array) -> list[VoiceChatEvent]:
        if self.max_frames is not None and self._frame_index >= self.max_frames:
            raise VoiceChatContextLimitError(
                f"stream exceeded the configured {self.max_frames}-frame context"
            )
        projected, encoded = self._perception_step(frame)
        events: list[VoiceChatEvent] = []
        transcript_update = self._rnnt.step(encoded)
        if transcript_update is not None:
            delta, text = transcript_update
            events.append(
                VoiceChatEvent(
                    kind="user_transcript_delta",
                    frame_index=self._frame_index,
                    delta=delta,
                    text=text,
                )
            )
        events.extend(
            self._run_timeline_step(
                projected,
                generate_channels=True,
                decode_audio=True,
            )
        )
        self._frame_index += 1
        return events

    def push_audio(
        self, samples, *, sample_rate: int | None = None
    ) -> list[VoiceChatEvent]:
        if self._closed:
            raise RuntimeError("streaming session is closed")
        sample_rate = self.input_sample_rate if sample_rate is None else sample_rate
        if sample_rate != self.input_sample_rate:
            raise ValueError(
                f"expected {self.input_sample_rate} Hz PCM, received {sample_rate} Hz"
            )
        chunk = mx.array(samples, dtype=mx.float32)
        if chunk.ndim == 2 and 1 in chunk.shape:
            chunk = chunk.reshape(-1)
        if chunk.ndim != 1:
            raise ValueError("audio must be mono PCM")
        if chunk.shape[0] == 0:
            return []
        self._pending_audio = mx.concatenate([self._pending_audio, chunk])
        events: list[VoiceChatEvent] = []
        while self._pending_audio.shape[0] >= self.frame_samples:
            frame = self._pending_audio[: self.frame_samples]
            self._pending_audio = self._pending_audio[self.frame_samples :]
            events.extend(self._step_audio_frame(frame))
        return events

    def flush(self, *, pad_partial: bool = True) -> list[VoiceChatEvent]:
        if self._closed:
            return []
        events: list[VoiceChatEvent] = []
        if self._pending_audio.shape[0] and pad_partial:
            frame = mx.pad(
                self._pending_audio,
                (0, self.frame_samples - self._pending_audio.shape[0]),
            )
            self._pending_audio = mx.zeros((0,), dtype=mx.float32)
            events.extend(self._step_audio_frame(frame))
        else:
            self._pending_audio = mx.zeros((0,), dtype=mx.float32)
        self._closed = True
        self._codec_cache.clear()
        events.append(VoiceChatEvent(kind="done", frame_index=self._frame_index))
        return events

    def cancel(self) -> list[VoiceChatEvent]:
        if self._closed:
            return []
        self._closed = True
        self._pending_audio = mx.zeros((0,), dtype=mx.float32)
        self._codec_cache.clear()
        return [VoiceChatEvent(kind="cancelled", frame_index=self._frame_index)]
