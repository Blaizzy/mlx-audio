from dataclasses import dataclass
from typing import Any

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class TowerConfig(BaseModelArgs):
    """The audio decoder that samples codebooks 1..9 of every frame."""

    dim: int
    n_head: int
    n_local_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    num_codebooks: int
    shared_enrichment_layers: int
    vq_backbone_codebooks: int
    norm_eps: float = 1e-6
    rope_base: float = 1e7
    rotary_percent: float = 0.25


@dataclass
class ModelConfig(BaseModelArgs):
    # Mapped onto the vendored Qwen3.5 arguments by backbone.text_model_args.
    text_config: dict[str, Any]
    audio_decoder_config: TowerConfig
    semantic_start_token_id: int
    semantic_end_token_id: int
    eos_token_id: int
    model_type: str = "fish_drama3"
    speech_template_version: str = "reference-bracket-v1"

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelConfig":
        tower = params["audio_decoder_config"]
        if not isinstance(tower, TowerConfig):
            tower = TowerConfig.from_dict(tower)
        return super().from_dict({**params, "audio_decoder_config": tower})
