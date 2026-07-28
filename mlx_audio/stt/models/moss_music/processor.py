import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np

from .audio import MossMusicFeatureExtractor
from .config import ModelConfig


@dataclass
class ProcessorOutput:
    input_ids: mx.array
    audio_input_mask: mx.array
    audio_data: Optional[mx.array] = None
    audio_data_seqlens: Optional[mx.array] = None
    token_lens: Optional[List[int]] = None
    audio_durations: Optional[List[float]] = None


class MossMusicProcessor:
    AUDIO_SPAN_RE = re.compile(r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>")

    def __init__(self, model_path: Union[str, Path], config: ModelConfig):
        self.config = config
        self.audio_token_id = int(config.audio_token_id)
        self.audio_start_id = int(config.audio_start_id)
        self.audio_end_id = int(config.audio_end_id)
        self.enable_time_marker = bool(config.enable_time_marker)
        self.feature_extractor = MossMusicFeatureExtractor(
            num_mel_bins=config.audio_config.num_mel_bins,
            sample_rate=config.sample_rate,
        )
        self.tokenizer = self._load_tokenizer(model_path)
        self._patch_tokenizer_aliases()
        self._digit_token_ids = {
            str(digit): int(
                self.tokenizer.encode(str(digit), add_special_tokens=False)[0]
            )
            for digit in range(10)
        }
        self.audio_tokens_per_second = 12.5
        self.time_marker_every_seconds = 2
        self.time_marker_every_audio_tokens = int(
            self.audio_tokens_per_second * self.time_marker_every_seconds
        )

    def _load_tokenizer(self, model_path: Union[str, Path]):
        import transformers
        from transformers import AutoTokenizer

        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            return AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                use_fast=False,
            )
        finally:
            transformers.logging.set_verbosity(prev)

    def _patch_tokenizer_aliases(self) -> None:
        alias_map = {
            "<|AUDIO|>": self.audio_token_id,
            "<|audio_bos|>": self.audio_start_id,
            "<|audio_eos|>": self.audio_end_id,
        }
        original = self.tokenizer.convert_tokens_to_ids

        def convert_tokens_to_ids(tokenizer_self, tokens):
            if isinstance(tokens, (list, tuple)):
                converted = [convert_tokens_to_ids(tokenizer_self, t) for t in tokens]
                return converted if isinstance(tokens, list) else tuple(converted)
            if isinstance(tokens, str) and tokens in alias_map:
                return alias_map[tokens]
            return original(tokens)

        self.tokenizer.convert_tokens_to_ids = types.MethodType(
            convert_tokens_to_ids,
            self.tokenizer,
        )

    @staticmethod
    def conv3_downsample_len(raw_mel_len: int) -> int:
        def conv_out_len(length: int) -> int:
            return (int(length) - 1) // 2 + 1

        return conv_out_len(conv_out_len(conv_out_len(raw_mel_len)))

    def _get_time_marker_token_ids(self, second: int) -> List[int]:
        return [self._digit_token_ids[digit] for digit in str(second)]

    def _build_audio_tokens_with_time_markers(self, audio_seq_len: int) -> List[int]:
        total_duration_seconds = audio_seq_len / self.audio_tokens_per_second
        num_full_seconds = int(total_duration_seconds)
        token_ids: List[int] = []
        audio_tokens_consumed = 0

        for second in range(
            self.time_marker_every_seconds,
            num_full_seconds + 1,
            self.time_marker_every_seconds,
        ):
            marker_pos = (
                second // self.time_marker_every_seconds
            ) * self.time_marker_every_audio_tokens
            audio_segment_len = marker_pos - audio_tokens_consumed
            if audio_segment_len > 0:
                token_ids.extend([self.audio_token_id] * audio_segment_len)
                audio_tokens_consumed += audio_segment_len
            token_ids.extend(self._get_time_marker_token_ids(second))

        remaining = int(audio_seq_len) - audio_tokens_consumed
        if remaining > 0:
            token_ids.extend([self.audio_token_id] * remaining)
        return token_ids

    def _build_audio_placeholder_ids(
        self,
        num_audio_tokens: int,
        enable_time_marker: Optional[bool] = None,
    ) -> List[int]:
        use_time_marker = (
            self.enable_time_marker
            if enable_time_marker is None
            else bool(enable_time_marker)
        )
        if use_time_marker:
            return self._build_audio_tokens_with_time_markers(num_audio_tokens)
        return [self.audio_token_id] * int(num_audio_tokens)

    def _default_prompt(self, text: str, has_audio: bool) -> str:
        if has_audio:
            return (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
                f"{text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _build_input_from_prompt(
        self,
        prompt: str,
        token_lens: List[int],
        enable_time_marker: Optional[bool] = None,
    ) -> List[int]:
        spans = list(self.AUDIO_SPAN_RE.finditer(prompt))
        if len(spans) != len(token_lens):
            raise ValueError(
                f"Audio placeholder count mismatch: found {len(spans)} spans "
                f"but got {len(token_lens)} audio inputs."
            )

        input_ids: List[int] = []
        cursor = 0
        for index, match in enumerate(spans):
            prefix = prompt[cursor : match.start()]
            if prefix:
                input_ids.extend(
                    self.tokenizer.encode(prefix, add_special_tokens=False)
                )
            input_ids.append(self.audio_start_id)
            input_ids.extend(
                self._build_audio_placeholder_ids(
                    token_lens[index],
                    enable_time_marker=enable_time_marker,
                )
            )
            input_ids.append(self.audio_end_id)
            cursor = match.end()

        suffix = prompt[cursor:]
        if suffix:
            input_ids.extend(self.tokenizer.encode(suffix, add_special_tokens=False))
        return input_ids

    def _load_single_audio(self, audio: Union[str, mx.array, np.ndarray]) -> mx.array:
        if isinstance(audio, str):
            from mlx_audio.stt.utils import load_audio

            return load_audio(audio, sr=self.config.sample_rate)
        if isinstance(audio, np.ndarray):
            return mx.array(audio, dtype=mx.float32)
        if isinstance(audio, mx.array):
            return audio.astype(mx.float32)
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    def _normalize_audio_list(
        self,
        audio: Optional[Union[str, mx.array, np.ndarray, Sequence]],
    ) -> List[mx.array]:
        if audio is None:
            return []
        if isinstance(audio, (list, tuple)):
            return [self._load_single_audio(a) for a in audio]
        return [self._load_single_audio(audio)]

    def __call__(
        self,
        *,
        text: Optional[str],
        audio: Optional[Union[str, mx.array, np.ndarray, Sequence]] = None,
        enable_time_marker: Optional[bool] = None,
    ) -> ProcessorOutput:
        audio_list = self._normalize_audio_list(audio)
        mels: List[mx.array] = []
        raw_lengths: List[int] = []
        token_lens: List[int] = []
        audio_durations: List[float] = []

        for item in audio_list:
            audio_durations.append(
                float(item.shape[-1]) / float(self.config.sample_rate)
            )
            mel, raw_len = self.feature_extractor(item)
            mels.append(mel)
            raw_lengths.append(raw_len)
            token_lens.append(self.conv3_downsample_len(raw_len))

        prompt_text = text or self.config.default_prompt
        if self.AUDIO_SPAN_RE.search(prompt_text) is None:
            prompt_text = self._default_prompt(prompt_text, has_audio=bool(audio_list))

        input_ids = mx.array(
            self._build_input_from_prompt(
                prompt_text,
                token_lens,
                enable_time_marker=enable_time_marker,
            )
        )
        audio_mask = input_ids == self.audio_token_id

        audio_batch = None
        seqlens = None
        if mels:
            max_len = max(raw_lengths)
            padded = []
            for mel in mels:
                if mel.shape[1] < max_len:
                    mel = mx.pad(mel, ((0, 0), (0, max_len - mel.shape[1])))
                padded.append(mel)
            audio_batch = mx.stack(padded, axis=0)
            seqlens = mx.array(raw_lengths, dtype=mx.int32)

        return ProcessorOutput(
            input_ids=input_ids.astype(mx.int32),
            audio_input_mask=audio_mask,
            audio_data=audio_batch,
            audio_data_seqlens=seqlens,
            token_lens=token_lens,
            audio_durations=audio_durations,
        )

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
