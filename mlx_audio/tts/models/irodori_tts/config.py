from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class IrodoriDiTConfig(BaseModelArgs):
    # Audio latent dimensions (v2: 32-dim Semantic-DACVAE, v1: 128-dim DACVAE)
    latent_dim: int = 32
    latent_patch_size: int = 1

    # DiT backbone
    model_dim: int = 1280
    num_layers: int = 12
    num_heads: int = 20
    mlp_ratio: float = 2.875
    text_mlp_ratio: Optional[float] = 2.6
    speaker_mlp_ratio: Optional[float] = 2.6

    # Text encoder
    text_vocab_size: int = 99574
    text_tokenizer_repo: str = "llm-jp/llm-jp-3-150m"
    text_add_bos: bool = True
    text_dim: int = 512
    text_layers: int = 10
    text_heads: int = 8

    # Speaker (reference latent) encoder
    speaker_dim: int = 768
    speaker_layers: int = 8
    speaker_heads: int = 12
    speaker_patch_size: int = 1

    # Conditioning
    timestep_embed_dim: int = 512
    adaln_rank: int = 192
    norm_eps: float = 1e-5

    # Caption (Voice Design) conditioning — mutually exclusive with speaker
    use_caption_condition: bool = False
    caption_vocab_size: Optional[int] = None
    caption_tokenizer_repo: Optional[str] = None
    caption_add_bos: Optional[bool] = None
    caption_dim: Optional[int] = None
    caption_layers: Optional[int] = None
    caption_heads: Optional[int] = None
    caption_mlp_ratio: Optional[float] = None

    @property
    def use_speaker_condition(self) -> bool:
        return not self.use_caption_condition

    @property
    def caption_vocab_size_resolved(self) -> int:
        return (
            self.caption_vocab_size
            if self.caption_vocab_size is not None
            else self.text_vocab_size
        )

    @property
    def caption_tokenizer_repo_resolved(self) -> str:
        return (
            self.caption_tokenizer_repo
            if self.caption_tokenizer_repo is not None
            else self.text_tokenizer_repo
        )

    @property
    def caption_add_bos_resolved(self) -> bool:
        return (
            self.caption_add_bos
            if self.caption_add_bos is not None
            else self.text_add_bos
        )

    @property
    def caption_dim_resolved(self) -> int:
        return self.caption_dim if self.caption_dim is not None else self.text_dim

    @property
    def caption_layers_resolved(self) -> int:
        return (
            self.caption_layers if self.caption_layers is not None else self.text_layers
        )

    @property
    def caption_heads_resolved(self) -> int:
        return self.caption_heads if self.caption_heads is not None else self.text_heads

    @property
    def caption_mlp_ratio_resolved(self) -> float:
        if self.caption_mlp_ratio is not None:
            return float(self.caption_mlp_ratio)
        return self.text_mlp_ratio_resolved

    @property
    def patched_latent_dim(self) -> int:
        return self.latent_dim * self.latent_patch_size

    @property
    def speaker_patched_latent_dim(self) -> int:
        return self.patched_latent_dim * self.speaker_patch_size

    @property
    def text_mlp_ratio_resolved(self) -> float:
        return (
            self.mlp_ratio
            if self.text_mlp_ratio is None
            else float(self.text_mlp_ratio)
        )

    @property
    def speaker_mlp_ratio_resolved(self) -> float:
        return (
            self.mlp_ratio
            if self.speaker_mlp_ratio is None
            else float(self.speaker_mlp_ratio)
        )


@dataclass
class SamplerConfig(BaseModelArgs):
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_scale_caption: float = 3.0
    cfg_guidance_mode: str = "independent"
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    truncation_factor: Optional[float] = None
    rescale_k: Optional[float] = None
    rescale_sigma: Optional[float] = None
    context_kv_cache: bool = True
    speaker_kv_scale: Optional[float] = None
    speaker_kv_min_t: Optional[float] = 0.9
    speaker_kv_max_layers: Optional[int] = None
    sequence_length: int = 750


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "irodori_tts"
    sample_rate: int = 48000

    max_text_length: int = 256
    max_caption_length: int = 512
    max_speaker_latent_length: int = 6400
    # DACVAE hop_length = 2*8*10*12 = 1920 (48kHz)
    audio_downsample_factor: int = 1920

    dacvae_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    model_path: Optional[str] = None

    dit: IrodoriDiTConfig = field(default_factory=IrodoriDiTConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        return cls(
            model_type=config.get("model_type", "irodori_tts"),
            sample_rate=config.get("sample_rate", 48000),
            max_text_length=config.get("max_text_length", 256),
            max_caption_length=config.get("max_caption_length", 512),
            max_speaker_latent_length=config.get("max_speaker_latent_length", 6400),
            audio_downsample_factor=config.get("audio_downsample_factor", 1920),
            dacvae_repo=config.get(
                "dacvae_repo", "Aratako/Semantic-DACVAE-Japanese-32dim"
            ),
            model_path=config.get("model_path"),
            dit=IrodoriDiTConfig.from_dict(config.get("dit", {})),
            sampler=SamplerConfig.from_dict(config.get("sampler", {})),
        )
