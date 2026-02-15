"""Prompt and codec helpers for MOSS-TTS-Realtime."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import mlx.core as mx
import numpy as np

from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer
from mlx_audio.utils import load_audio

from .config import ModelConfig


def _normalize_waveform_layout_to_time_major(wav: mx.array) -> mx.array:
    if wav.ndim != 2:
        return wav

    rows = int(wav.shape[0])
    cols = int(wav.shape[1])
    max_expected_channels = 8

    if rows <= max_expected_channels and cols > rows:
        return wav.transpose(1, 0)
    if cols <= max_expected_channels and rows >= cols:
        return wav
    if rows < cols:
        return wav.transpose(1, 0)
    return wav


def _normalize_preencoded_audio_tokens(
    audio_tokens: mx.array,
    *,
    rvq: int,
    audio_pad_token: int,
) -> Optional[mx.array]:
    if audio_tokens.ndim != 2:
        return None

    tokens_np = np.array(audio_tokens)
    if not np.issubdtype(tokens_np.dtype, np.integer):
        return None

    rows = int(audio_tokens.shape[0])
    cols = int(audio_tokens.shape[1])
    if rows < rvq and cols < rvq:
        return None

    if rows == rvq and cols != rvq:
        normalized = tokens_np[:rvq, :].T
    elif cols == rvq and rows != rvq:
        normalized = tokens_np[:, :rvq]
    elif rows >= rvq and cols < rvq:
        normalized = tokens_np[:rvq, :].T
    elif cols >= rvq and rows < rvq:
        normalized = tokens_np[:, :rvq]
    elif rvq > 0 and rows % rvq == 0 and rows <= 64:
        normalized = tokens_np[:rvq, :].T
    elif rvq > 0 and cols % rvq == 0 and cols <= 64:
        normalized = tokens_np[:, :rvq]
    elif rows <= 64 and cols > 64:
        normalized = tokens_np[:rvq, :].T
    elif cols <= 64 and rows > 64:
        normalized = tokens_np[:, :rvq]
    else:
        normalized = tokens_np[:, :rvq]

    normalized = normalized.astype(np.int32)
    normalized = np.clip(normalized, 0, audio_pad_token)
    return mx.array(normalized, dtype=mx.int32)


class MossTTSRealtimeProcessor:
    """Processor for realtime turn shaping and codec encode/decode."""

    def __init__(
        self,
        tokenizer: Any,
        audio_tokenizer: Optional[MossAudioTokenizer],
        model_config: ModelConfig,
    ):
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.model_config = model_config

    def tokens_from_text(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        try:
            return list(
                self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            )
        except TypeError:
            return list(self.tokenizer.encode(text))

    def encode_prompt_audio(self, audio: Any) -> mx.array:
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is not loaded")

        if isinstance(audio, str):
            wav = load_audio(audio, sample_rate=self.model_config.sampling_rate)
            normalized_audio = wav.astype(mx.float32)
        elif isinstance(audio, mx.array):
            preencoded = _normalize_preencoded_audio_tokens(
                audio,
                rvq=self.model_config.rvq,
                audio_pad_token=self.model_config.audio_pad_token,
            )
            if preencoded is not None:
                return preencoded
            normalized_audio = audio
        else:
            raise TypeError(
                "reference audio must be a file path, mx.array waveform, or pre-encoded "
                "audio token matrix"
            )

        if normalized_audio.ndim == 2:
            normalized_audio = _normalize_waveform_layout_to_time_major(normalized_audio)
            normalized_audio = mx.mean(normalized_audio, axis=1)
        if normalized_audio.ndim != 1:
            raise ValueError(
                f"Expected 1D waveform or 2D matrix, got shape {normalized_audio.shape}"
            )

        encoded = self.audio_tokenizer.batch_encode(
            [normalized_audio.astype(mx.float32)],
            num_quantizers=self.model_config.rvq,
        )
        if encoded.audio_codes is None or encoded.audio_codes_lengths is None:
            raise RuntimeError("audio tokenizer encode returned empty outputs")

        length = int(encoded.audio_codes_lengths[0])
        codes = encoded.audio_codes[:, 0, :length].transpose(1, 0)
        return codes.astype(mx.int32)

    def decode_audio_codes(
        self,
        audio_codes: mx.array,
        *,
        chunk_duration: Optional[float],
        decode_kwargs: Optional[dict[str, Any]] = None,
    ) -> mx.array:
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is not loaded")
        if audio_codes.ndim != 2:
            raise ValueError(f"Expected audio_codes with shape (T, RVQ), got {audio_codes.shape}")

        kwargs = {} if decode_kwargs is None else dict(decode_kwargs)
        decoded = self.audio_tokenizer.decode(
            audio_codes.transpose(1, 0),
            return_dict=True,
            chunk_duration=chunk_duration,
            num_quantizers=int(audio_codes.shape[1]),
            **kwargs,
        )

        if isinstance(decoded, dict):
            audio = decoded.get("audio")
            audio_lengths = decoded.get("audio_lengths")
            if audio is None or audio_lengths is None:
                raise RuntimeError("audio tokenizer decode returned empty outputs")
            length = int(audio_lengths[0])
            return audio[0, 0, :length]

        if decoded.audio is None or decoded.audio_lengths is None:
            raise RuntimeError("audio tokenizer decode returned empty outputs")
        length = int(decoded.audio_lengths[0])
        return decoded.audio[0, 0, :length]

    def build_turn_input_ids(
        self,
        *,
        user_text: str,
        user_audio_tokens: Optional[mx.array] = None,
        include_system_prompt: bool = True,
    ) -> mx.array:
        channels = self.model_config.channels
        rvq = self.model_config.rvq
        audio_pad_token = self.model_config.audio_pad_token

        segments: list[mx.array] = []

        if include_system_prompt:
            system_ids = self.tokens_from_text(
                self.model_config.tts_system_prompt,
                add_special_tokens=False,
            )
            if not system_ids:
                system_ids = [self.model_config.text_pad]
            system = mx.full((len(system_ids), channels), audio_pad_token, dtype=mx.int32)
            system[:, 0] = mx.array(system_ids, dtype=mx.int32)
            segments.append(system)

        user_ids = self.tokens_from_text(user_text, add_special_tokens=False)
        if not user_ids:
            user_ids = [self.model_config.text_pad]
        user = mx.full((len(user_ids), channels), audio_pad_token, dtype=mx.int32)
        user[:, 0] = mx.array(user_ids, dtype=mx.int32)
        segments.append(user)

        if user_audio_tokens is not None:
            normalized_tokens = _normalize_preencoded_audio_tokens(
                user_audio_tokens,
                rvq=rvq,
                audio_pad_token=audio_pad_token,
            )
            if normalized_tokens is None:
                normalized_tokens = self.encode_prompt_audio(user_audio_tokens)

            prompt = mx.full(
                (int(normalized_tokens.shape[0]), channels),
                audio_pad_token,
                dtype=mx.int32,
            )
            prompt[:, 0] = int(self.model_config.reference_audio_pad)
            prompt[:, 1 : 1 + rvq] = normalized_tokens[:, :rvq]
            segments.append(prompt)

        turn_input = mx.concatenate(segments, axis=0)
        return turn_input[None, :, :].astype(mx.int32)

    def make_text_prefix(self, text_token_ids: Iterable[int]) -> list[int]:
        return [int(token_id) for token_id in text_token_ids]


__all__ = ["MossTTSRealtimeProcessor"]
