"""Configuration models for MOSS-TTS-Realtime runtime."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

from mlx_audio.tts.models.base import BaseModelArgs

DEFAULT_TTS_SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are a highly expressive text-to-speech (TTS) engine developed by "
    "Mosi Intelligence.\n"
    "<|im_end|>\n"
)


def _filter_dataclass_kwargs(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    valid = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in valid}


@dataclass
class RealtimeLanguageConfig(BaseModelArgs):
    """Subset of Qwen3 config fields used by the realtime backbone."""

    model_type: str = "qwen3"
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rope_theta: float = 1_000_000.0
    head_dim: int = 128
    tie_word_embeddings: bool = False
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"

    def __post_init__(self):
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )


@dataclass
class RealtimeLocalConfig(BaseModelArgs):
    """Config for the realtime local token decoder."""

    model_type: str = "moss_tts_realtime_local_transformer"
    hidden_size: int = 2048
    num_hidden_layers: int = 4
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 33
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    rvq: int = 16
    audio_vocab_size: int = 1027
    audio_pad_token: int = 1024

    def __post_init__(self):
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.rvq <= 0:
            raise ValueError("rvq must be positive")
        if self.audio_vocab_size <= 0:
            raise ValueError("audio_vocab_size must be positive")


@dataclass
class ModelConfig(BaseModelArgs):
    """Unified config for the MLX MOSS-TTS-Realtime runtime."""

    model_type: str = "moss_tts_realtime"
    language_config: Optional[RealtimeLanguageConfig | Dict[str, Any]] = None
    local_config: Optional[RealtimeLocalConfig | Dict[str, Any]] = None

    rvq: int = 16
    audio_vocab_size: int = 1027
    audio_pad_token: int = 1024
    audio_bos_token: int = 1025
    audio_eos_token: int = 1026

    reference_audio_pad: int = 151654
    text_pad: int = 151655
    delay_tokens_len: int = 12
    sampling_rate: int = 24000
    max_context_tokens: int = 32768
    initializer_range: float = 0.02
    tts_system_prompt: str = DEFAULT_TTS_SYSTEM_PROMPT

    def __post_init__(self):
        if isinstance(self.language_config, dict):
            filtered = _filter_dataclass_kwargs(
                RealtimeLanguageConfig,
                self.language_config,
            )
            self.language_config = RealtimeLanguageConfig(**filtered)
        elif self.language_config is None:
            self.language_config = RealtimeLanguageConfig()

        if isinstance(self.local_config, dict):
            filtered = _filter_dataclass_kwargs(RealtimeLocalConfig, self.local_config)
            self.local_config = RealtimeLocalConfig(**filtered)
        elif self.local_config is None:
            assert self.language_config is not None
            self.local_config = RealtimeLocalConfig(
                hidden_size=self.language_config.hidden_size,
                intermediate_size=self.language_config.intermediate_size,
                num_attention_heads=self.language_config.num_attention_heads,
                num_key_value_heads=self.language_config.num_key_value_heads,
                head_dim=self.language_config.head_dim,
            )

        if self.model_type != "moss_tts_realtime":
            self.model_type = "moss_tts_realtime"
        if self.rvq <= 0:
            raise ValueError("rvq must be positive")
        if self.audio_vocab_size <= 0:
            raise ValueError("audio_vocab_size must be positive")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.delay_tokens_len <= 0:
            raise ValueError("delay_tokens_len must be positive")

        if not (0 <= self.audio_pad_token < self.audio_vocab_size):
            raise ValueError("audio_pad_token must be within audio vocab range")
        if not (0 <= self.audio_bos_token < self.audio_vocab_size):
            raise ValueError("audio_bos_token must be within audio vocab range")
        if not (0 <= self.audio_eos_token < self.audio_vocab_size):
            raise ValueError("audio_eos_token must be within audio vocab range")

        assert self.local_config is not None
        self.local_config.rvq = int(self.rvq)
        self.local_config.audio_vocab_size = int(self.audio_vocab_size)
        self.local_config.audio_pad_token = int(self.audio_pad_token)

    @property
    def channels(self) -> int:
        return 1 + self.rvq

    @property
    def hidden_size(self) -> int:
        assert self.language_config is not None
        return self.language_config.hidden_size

    @property
    def vocab_size(self) -> int:
        assert self.language_config is not None
        return self.language_config.vocab_size

    def local_transformer_config(self) -> RealtimeLocalConfig:
        if self.local_config is None:
            raise ValueError("local_config is required")
        return self.local_config


__all__ = [
    "DEFAULT_TTS_SYSTEM_PROMPT",
    "ModelConfig",
    "RealtimeLanguageConfig",
    "RealtimeLocalConfig",
]
