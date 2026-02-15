"""Prompt and codec helpers for MOSS-TTS-Realtime."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import mlx.core as mx
import numpy as np

from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer
from mlx_audio.utils import load_audio

from .config import ModelConfig

_AUDIO_PAD_TOKEN_TEXT = "<|audio_pad|>"
_TEXT_PAD_TOKEN_TEXT = "<|text_pad|>"
_USER_PREFIX = "<|im_end|>\n<|im_start|>user\n"
_ASSISTANT_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"


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
        self.delay_tokens_len = int(model_config.delay_tokens_len)
        self.channels = int(model_config.channels)
        self.audio_pad_token_id = self._convert_token_to_id(
            _AUDIO_PAD_TOKEN_TEXT,
            fallback_id=int(model_config.reference_audio_pad),
        )
        self.text_pad_token_id = self._convert_token_to_id(
            _TEXT_PAD_TOKEN_TEXT,
            fallback_id=int(model_config.text_pad),
        )

    def _tokenize(
        self,
        text: str,
        *,
        add_special_tokens: Optional[bool] = None,
    ) -> list[int]:
        if callable(getattr(self.tokenizer, "__call__", None)):
            try:
                if add_special_tokens is None:
                    encoded = self.tokenizer(text)
                else:
                    encoded = self.tokenizer(
                        text,
                        add_special_tokens=bool(add_special_tokens),
                    )
                if isinstance(encoded, dict) and "input_ids" in encoded:
                    input_ids = encoded["input_ids"]
                    if input_ids and isinstance(input_ids[0], list):
                        input_ids = input_ids[0]
                    return [int(token_id) for token_id in input_ids]
            except TypeError:
                pass

        try:
            if add_special_tokens is None:
                token_ids = self.tokenizer.encode(text)
            else:
                token_ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=bool(add_special_tokens),
                )
        except TypeError:
            token_ids = self.tokenizer.encode(text)
        return [int(token_id) for token_id in token_ids]

    def _convert_token_to_id(self, token: str, *, fallback_id: int) -> int:
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
            if token_id is not None and token_id != unk_token_id:
                return int(token_id)

        token_ids = self._tokenize(token, add_special_tokens=False)
        if len(token_ids) == 1:
            return int(token_ids[0])
        return int(fallback_id)

    def tokens_from_text(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self._tokenize(text, add_special_tokens=add_special_tokens)

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

    def _normalize_audio_prompt_tokens(self, audio_tokens: Optional[Any]) -> mx.array:
        if audio_tokens is None:
            return mx.zeros((0, self.model_config.rvq), dtype=mx.int32)

        candidate = audio_tokens
        if isinstance(audio_tokens, np.ndarray):
            candidate = mx.array(audio_tokens)

        if isinstance(candidate, mx.array):
            preencoded = _normalize_preencoded_audio_tokens(
                candidate,
                rvq=self.model_config.rvq,
                audio_pad_token=self.model_config.audio_pad_token,
            )
            if preencoded is not None:
                return preencoded

        return self.encode_prompt_audio(candidate)

    def normalize_audio_prompt_tokens(self, audio_tokens: Optional[Any]) -> mx.array:
        return self._normalize_audio_prompt_tokens(audio_tokens)

    @staticmethod
    def make_voice_clone_prompt(prompt_audio_tokens_len: int) -> str:
        padded_audio_prompt = f"{_AUDIO_PAD_TOKEN_TEXT * int(prompt_audio_tokens_len)}"
        return (
            "<|im_start|>context\n"
            "The assistant section should be synthesized using the following voice timbre:"
            f"{padded_audio_prompt}"
        )

    def make_ensemble(self, prompt_audio_tokens: Optional[mx.array] = None) -> mx.array:
        normalized_prompt_tokens = self._normalize_audio_prompt_tokens(prompt_audio_tokens)
        if int(normalized_prompt_tokens.shape[0]) > 0:
            system_prompt_text = (
                f"{self.model_config.tts_system_prompt}"
                f"{self.make_voice_clone_prompt(int(normalized_prompt_tokens.shape[0]))}"
            )
        else:
            system_prompt_text = f"{self.model_config.tts_system_prompt}"

        system_prompt_tokens = self._tokenize(system_prompt_text, add_special_tokens=None)
        if not system_prompt_tokens:
            system_prompt_tokens = [self.text_pad_token_id]

        system_prompt = mx.full(
            (len(system_prompt_tokens), self.channels),
            self.model_config.audio_pad_token,
            dtype=mx.int32,
        )
        system_prompt[:, 0] = mx.array(system_prompt_tokens, dtype=mx.int32)

        if int(normalized_prompt_tokens.shape[0]) > 0:
            token_array = np.array(system_prompt_tokens)
            indices = np.where(token_array == int(self.audio_pad_token_id))[0]
            if indices.size == 0:
                raise ValueError("No <|audio_pad|> tokens found in the system prompt.")
            prompt_audio_start_pos = int(indices[0])
            prompt_audio_end_pos = int(indices[-1])
            prompt_audio_len = prompt_audio_end_pos - prompt_audio_start_pos + 1
            if prompt_audio_len != int(normalized_prompt_tokens.shape[0]):
                raise ValueError(
                    "Voice prompt placeholder length mismatch: "
                    f"{prompt_audio_len} placeholders vs {int(normalized_prompt_tokens.shape[0])} frames"
                )
            system_prompt[prompt_audio_start_pos : prompt_audio_end_pos + 1, 1:] = (
                normalized_prompt_tokens[:, : self.model_config.rvq]
            )

        return system_prompt.astype(mx.int32)

    def make_user_prompt(self, text: str, audio_tokens: Optional[mx.array]) -> mx.array:
        token = self._normalize_audio_prompt_tokens(audio_tokens)
        token = token[:, : self.model_config.rvq]

        text_tokens = self._tokenize(text, add_special_tokens=None)
        text_start_pos = len(self._tokenize(_USER_PREFIX, add_special_tokens=None))
        text_len = len(text_tokens)
        audio_len = int(token.shape[0])

        if text_len >= self.delay_tokens_len:
            padded_text_len = max(0, audio_len + self.delay_tokens_len - text_len + 1)
            cur_input_id_ch1 = f"{_USER_PREFIX}{text}{_TEXT_PAD_TOKEN_TEXT * padded_text_len}"
            assistant_tokens_ch1 = self._tokenize(cur_input_id_ch1, add_special_tokens=None)
            cur_input_id = mx.full(
                (len(assistant_tokens_ch1), self.channels),
                self.model_config.audio_pad_token,
                dtype=mx.int32,
            )
            cur_input_id[:, 0] = mx.array(assistant_tokens_ch1, dtype=mx.int32)
            cur_input_id[
                text_start_pos + self.delay_tokens_len : text_start_pos + self.delay_tokens_len + audio_len,
                1:,
            ] = token
            cur_input_id[text_start_pos + self.delay_tokens_len - 1, 1] = int(
                self.model_config.audio_bos_token
            )
            cur_input_id[text_start_pos + self.delay_tokens_len + audio_len, 1] = int(
                self.model_config.audio_eos_token
            )
        else:
            padded_text_len = audio_len + 1
            cur_input_id_ch1 = f"{_USER_PREFIX}{text}{_TEXT_PAD_TOKEN_TEXT * padded_text_len}"
            assistant_tokens_ch1 = self._tokenize(cur_input_id_ch1, add_special_tokens=None)
            cur_input_id = mx.full(
                (len(assistant_tokens_ch1), self.channels),
                self.model_config.audio_pad_token,
                dtype=mx.int32,
            )
            cur_input_id[:, 0] = mx.array(assistant_tokens_ch1, dtype=mx.int32)
            cur_input_id[-(audio_len + 1) : -1, 1:] = token
            cur_input_id[-(audio_len + 2), 1] = int(self.model_config.audio_bos_token)
            cur_input_id[-1, 1] = int(self.model_config.audio_eos_token)

        begin_of_response = self._tokenize(_ASSISTANT_PREFIX, add_special_tokens=None)
        begin_of_response_full = mx.full(
            (len(begin_of_response), self.channels),
            self.model_config.audio_pad_token,
            dtype=mx.int32,
        )
        begin_of_response_full[:, 0] = mx.array(begin_of_response, dtype=mx.int32)
        return mx.concatenate([cur_input_id, begin_of_response_full], axis=0).astype(mx.int32)

    def build_turn_input_ids(
        self,
        *,
        user_text: str,
        user_audio_tokens: Optional[Any] = None,
        include_system_prompt: bool = True,
        voice_prompt_tokens: Optional[Any] = None,
    ) -> mx.array:
        user_prompt = self.make_user_prompt(user_text, user_audio_tokens)
        if include_system_prompt:
            system_prompt = self.make_ensemble(voice_prompt_tokens)
            turn_input = mx.concatenate([system_prompt, user_prompt], axis=0)
        else:
            turn_input = user_prompt
        return turn_input[None, :, :].astype(mx.int32)

    def make_text_prefix(self, text_token_ids: Iterable[int]) -> list[int]:
        return [int(token_id) for token_id in text_token_ids]


__all__ = ["MossTTSRealtimeProcessor"]
