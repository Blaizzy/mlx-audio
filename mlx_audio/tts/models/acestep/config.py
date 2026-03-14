from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseModelArgs


@dataclass
class AceStepDiTConfig:
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    in_channels: int = 192
    audio_acoustic_hidden_dim: int = 64
    patch_size: int = 2
    sliding_window: int = 128
    layer_types: List[str] = field(default_factory=lambda: ["cross", "self"] * 12)
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768

    # Extra config needed by ConditionEncoder
    text_hidden_dim: int = 1536  # Qwen3-1.7B default hidden size
    num_lyric_encoder_hidden_layers: int = 4

    @classmethod
    def from_dict(cls, params: dict):
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})


@dataclass
class AceStepVAEConfig:
    encoder_hidden_size: int = 128
    downsampling_ratios: List[int] = field(default_factory=lambda: [2, 4, 4, 6, 10])
    channel_multiples: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    decoder_channels: int = 128
    decoder_input_channels: int = 64
    audio_channels: int = 2

    @classmethod
    def from_dict(cls, params: dict):
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})


@dataclass
class AceStepConfig(BaseModelArgs):
    model_type: str = "acestep"
    dit_config: AceStepDiTConfig = field(default_factory=AceStepDiTConfig)
    vae_config: AceStepVAEConfig = field(default_factory=AceStepVAEConfig)
    lm_repo: str = "ACE-Step/acestep-5Hz-lm-1.7B"

    @classmethod
    def from_dict(cls, params: dict):
        dit_params = {
            k: v
            for k, v in params.items()
            if k in AceStepDiTConfig.__dataclass_fields__
        }
        dit_config = (
            AceStepDiTConfig.from_dict(dit_params) if dit_params else AceStepDiTConfig()
        )

        vae_config = AceStepVAEConfig()
        if "vae_config" in params:
            vae_config = AceStepVAEConfig.from_dict(params["vae_config"])

        return cls(dit_config=dit_config, vae_config=vae_config)
