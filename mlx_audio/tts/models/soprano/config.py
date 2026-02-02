# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar

from mlx_audio.tts.models.base import BaseModelArgs

T = TypeVar("T")


def filter_dict_for_dataclass(cls: Type[T], data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a dictionary to only include keys that are valid dataclass fields."""
    valid_fields = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid_fields}


@dataclass
class SopranoDecoderConfig(BaseModelArgs):
    """Configuration for Soprano decoder."""

    decoder_num_layers: int = 8
    decoder_dim: int = 512
    decoder_intermediate_dim: int = 1536
    hop_length: int = 512
    n_fft: int = 2048
    upscale: int = 4
    dw_kernel: int = 3
    input_kernel_size: int = 3
    token_size: int = 2048  # Samples per audio token
    receptive_field: int = 4  # Decoder receptive field
    decoder_path: Optional[str] = "decoder.pth"


@dataclass
class ModelConfig(BaseModelArgs):
    """Configuration for Soprano model."""

    model_type: str = "soprano"
    hidden_size: int = 512
    num_hidden_layers: int = 17
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    vocab_size: int = 8192
    rms_norm_eps: float = 1e-6
    intermediate_size: int = 2304
    max_position_embeddings: int = 1024
    head_dim: int = 128
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 3
    eos_token_id: int = 3
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    sliding_window: Optional[int] = None

    layer_types: Optional[List[str]] = None

    sample_rate: int = 32000
    decoder_config: Optional[SopranoDecoderConfig] = None
    model_path: Optional[str] = None

    def __post_init__(self):
        if self.decoder_config is None:
            # Heuristic for backward compatibility if decoder_config is missing
            if self.max_position_embeddings == 1024:
                # Soprano 1.1-80M
                self.decoder_config = SopranoDecoderConfig(
                    decoder_dim=768,
                    decoder_intermediate_dim=2304,
                    input_kernel_size=1,
                )
            else:
                # Soprano 80M
                self.decoder_config = SopranoDecoderConfig(
                    decoder_dim=512,
                    decoder_intermediate_dim=1536,
                    input_kernel_size=3,
                )
        elif isinstance(self.decoder_config, dict):
            filtered = filter_dict_for_dataclass(
                SopranoDecoderConfig, self.decoder_config
            )
            self.decoder_config = SopranoDecoderConfig(**filtered)

    @classmethod
    def from_dict(cls, params: dict) -> "ModelConfig":
        """Create config from a dictionary."""
        # Ensure model_type is soprano
        params["model_type"] = "soprano"
        decoder_cfg = params.pop("decoder_config", None)

        # Create sub-configs
        if isinstance(decoder_cfg, dict):
            decoder_config = SopranoDecoderConfig.from_dict(decoder_cfg)
        else:
            decoder_config = decoder_cfg

        # Filter main config params
        config = cls(
            decoder_config=decoder_config,
            **{
                k: v
                for k, v in params.items()
                if hasattr(cls, k) or k in cls.__dataclass_fields__
            },
        )

        return config
