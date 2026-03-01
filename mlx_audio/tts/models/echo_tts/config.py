from __future__ import annotations

from dataclasses import dataclass, field

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class EchoDiTConfig(BaseModelArgs):
    latent_size: int = 80

    model_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 5888
    norm_eps: float = 1e-5

    text_vocab_size: int = 256
    text_model_size: int = 1280
    text_num_layers: int = 14
    text_num_heads: int = 10
    text_intermediate_size: int = 3328

    speaker_patch_size: int = 4
    speaker_model_size: int = 1280
    speaker_num_layers: int = 14
    speaker_num_heads: int = 10
    speaker_intermediate_size: int = 3328

    timestep_embed_size: int = 512
    adaln_rank: int = 256


@dataclass
class SamplerConfig(BaseModelArgs):
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 8.0
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    speaker_kv_scale: float | None = None
    speaker_kv_max_layers: int | None = None
    speaker_kv_min_t: float | None = None
    sequence_length: int = 640


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "echo_tts"
    sample_rate: int = 44100

    max_text_length: int = 768
    max_speaker_latent_length: int = 6400
    audio_downsample_factor: int = 2048

    normalize_text: bool = True
    delete_blockwise_modules: bool = False
    pca_filename: str = "pca_state.safetensors"
    fish_codec_repo: str = "jordand/fish-s1-dac-min"

    model_path: str | None = None

    dit: EchoDiTConfig = field(default_factory=EchoDiTConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        return cls(
            model_type=config.get("model_type", "echo_tts"),
            sample_rate=config.get("sample_rate", 44100),
            max_text_length=config.get("max_text_length", 768),
            max_speaker_latent_length=config.get("max_speaker_latent_length", 6400),
            audio_downsample_factor=config.get("audio_downsample_factor", 2048),
            normalize_text=config.get("normalize_text", True),
            delete_blockwise_modules=config.get("delete_blockwise_modules", False),
            pca_filename=config.get("pca_filename", "pca_state.safetensors"),
            fish_codec_repo=config.get("fish_codec_repo", "jordand/fish-s1-dac-min"),
            model_path=config.get("model_path"),
            dit=EchoDiTConfig.from_dict(config.get("dit", {})),
            sampler=SamplerConfig.from_dict(config.get("sampler", {})),
        )
