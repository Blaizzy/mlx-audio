"""Configuration models for MOSS-TTS Local/Delay variants."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional

from mlx_audio.tts.models.base import BaseModelArgs


def _filter_dataclass_kwargs(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    valid = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in valid}


@dataclass
class MossQwen3Config(BaseModelArgs):
    """Subset of Qwen3 config fields used by the MOSS backbone."""

    model_type: str = "qwen3"
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 155648
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rope_theta: float = 1_000_000.0
    head_dim: int = 128
    tie_word_embeddings: bool = True
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"

    def __post_init__(self):
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )


@dataclass
class ModelConfig(BaseModelArgs):
    """
    Unified config for MOSS-TTS family models.

    Local and Delay variants share the same `model_type`. Local is identified by
    the presence of `local_num_layers`.
    """

    model_type: str = "moss_tts_delay"
    language_config: Optional[MossQwen3Config | Dict[str, Any]] = None
    initializer_range: float = 0.02
    n_vq: int = 32
    pad_token_id: int = 151643
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    audio_vocab_size: int = 1024
    audio_user_slot_token_id: int = 151654
    audio_assistant_gen_slot_token_id: int = 151656
    audio_assistant_delay_slot_token_id: int = 151662
    audio_start_token_id: int = 151652
    audio_end_token_id: int = 151653
    audio_pad_code: int = 1024
    sampling_rate: int = 24000
    additional_mlp_ffn_hidden_size: int = 2048
    local_ffn_hidden_size: Optional[int] = None
    local_hidden_size: Optional[int] = None
    local_num_layers: Optional[int] = None
    audio_ch0_vocab_size: Optional[int] = None
    gen_token_id: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.language_config, dict):
            filtered = _filter_dataclass_kwargs(MossQwen3Config, self.language_config)
            self.language_config = MossQwen3Config(**filtered)
        elif self.language_config is None:
            self.language_config = MossQwen3Config()

        if self.n_vq <= 0:
            raise ValueError("n_vq must be positive")
        if self.audio_vocab_size <= 0:
            raise ValueError("audio_vocab_size must be positive")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        # Local-only defaults (Delay configs leave these unset).
        if self.local_num_layers is not None and self.local_num_layers <= 0:
            raise ValueError("local_num_layers must be positive when provided")
        if self.local_num_layers is not None:
            if self.local_hidden_size is None:
                self.local_hidden_size = self.language_config.hidden_size
            if self.local_ffn_hidden_size is None:
                self.local_ffn_hidden_size = self.language_config.intermediate_size

    @property
    def is_local_variant(self) -> bool:
        return self.local_num_layers is not None

    @property
    def channels(self) -> int:
        return 1 + self.n_vq

    @property
    def hidden_size(self) -> int:
        return self.language_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.language_config.vocab_size

    def local_transformer_config(self) -> MossQwen3Config:
        """
        Build the no-RoPE local transformer config.

        Local transformer reuses Qwen3 attention/MLP dimensions but runs on a
        short `(1 + n_vq)` sequence per global step.
        """

        if not self.is_local_variant:
            raise ValueError("Delay variant has no local transformer config")
        assert self.local_hidden_size is not None
        assert self.local_ffn_hidden_size is not None
        assert self.local_num_layers is not None

        if self.local_hidden_size % self.language_config.num_attention_heads != 0:
            raise ValueError(
                "local_hidden_size must be divisible by num_attention_heads"
            )

        return replace(
            self.language_config,
            hidden_size=self.local_hidden_size,
            intermediate_size=self.local_ffn_hidden_size,
            num_hidden_layers=self.local_num_layers,
            head_dim=self.local_hidden_size
            // self.language_config.num_attention_heads,
        )


__all__ = ["ModelConfig", "MossQwen3Config"]
