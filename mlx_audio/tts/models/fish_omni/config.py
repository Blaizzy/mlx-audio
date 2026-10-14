from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class AudioDecoderConfig(BaseModelArgs):
    """Codebook tower of Drama 3 (`audio_decoder_config` in config.json)."""

    model_type: str = "fish_qwen3_audio_decoder"
    vocab_size: int = 4096
    num_codebooks: int = 10
    dim: int = 2048
    n_head: int = 16
    n_local_heads: int = 2
    head_dim: int = 256
    intermediate_size: int = 6144
    norm_eps: float = 1e-6
    rope_base: float = 10000000.0
    rotary_percent: float = 0.25
    shared_enrichment_layers: int = 4
    vq_backbone_codebooks: int = 1

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_percent)


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "fish_omni"
    model_path: Optional[str] = None
    eos_token_id: int = 248046
    pad_token_id: int = 248044
    semantic_start_token_id: int = 248099
    semantic_end_token_id: int = 252194
    sample_rate: int = 44100
    # Backbone config, passed through to the vendored Qwen3.5 text model.
    text_config: dict = field(default_factory=dict)
    audio_decoder_config: AudioDecoderConfig = field(default_factory=AudioDecoderConfig)
    quantization: Optional[dict] = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelConfig":
        decoder = params.get("audio_decoder_config", {})
        if not isinstance(decoder, AudioDecoderConfig):
            decoder = AudioDecoderConfig.from_dict(decoder)
        return cls(
            model_type=params.get("model_type", "fish_omni"),
            model_path=params.get("model_path"),
            eos_token_id=params.get("eos_token_id", 248046),
            pad_token_id=params.get("pad_token_id", 248044),
            semantic_start_token_id=params.get("semantic_start_token_id", 248099),
            semantic_end_token_id=params.get("semantic_end_token_id", 252194),
            sample_rate=params.get("sample_rate", 44100),
            text_config=dict(params.get("text_config", {})),
            audio_decoder_config=decoder,
            quantization=params.get("quantization"),
        )
