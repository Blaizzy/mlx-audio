from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from mlx_audio.tts.models.base import BaseModelArgs

DEFAULT_TEXT_ENCODER = "mlx-community/gemma-3-12b-it-8bit"


def _filter_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in values.items() if k in allowed}


@dataclass
class TransformerConfig(BaseModelArgs):
    num_layers: int = 48
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_cross_attention_dim: int = 2048
    norm_eps: float = 1e-6
    attention_type: str = "default"
    positional_embedding_theta: float = 10000.0
    audio_positional_embedding_max_pos: list[float] = field(
        default_factory=lambda: [20.0]
    )
    timestep_scale_multiplier: int = 1000
    use_middle_indices_grid: bool = True
    rope_type: str = "split"
    frequencies_precision: str | bool = "float64"
    apply_gated_attention: bool = True
    cross_attention_adaln: bool = True
    caption_proj_before_connector: bool = True
    audio_connector_attention_head_dim: int = 64
    audio_connector_num_attention_heads: int = 32
    connector_num_layers: int = 8
    connector_positional_embedding_max_pos: list[int] = field(
        default_factory=lambda: [4096]
    )
    connector_num_learnable_registers: int = 128


@dataclass
class AudioConfig(BaseModelArgs):
    sample_rate: int = 48000
    latent_sample_rate: int = 16000
    hop_length: int = 160
    latent_downsample_factor: int = 4
    vae_channels: int = 8
    mel_bins: int = 16
    fps: float = 25.0


@dataclass
class InferenceDefaults(BaseModelArgs):
    cfg_scale: float = 2.5
    stg_scale: float = 1.5
    stg_block: int = 29
    rescale_scale: str | float = "auto"
    modality_scale: float = 1.0
    duration_multiplier: float = 1.1
    seed: int = 42
    steps: int = 30
    ref_duration: float = 10.0
    negative_prompt: str = (
        "worst quality, inconsistent motion, blurry, jittery, distorted, "
        "robotic voice, echo, background noise, off-sync audio, repetitive speech"
    )


@dataclass
class ModelFiles(BaseModelArgs):
    transformer: str = "dramabox-dit-v1.safetensors"
    audio_components: str = "dramabox-audio-components.safetensors"
    silence_latent: str = "assets/silence_latent_frame.pt"


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "dramabox-tts"
    architecture: str = "DiT-FlowMatching"
    base_model: str = "ltx-2.3-22b-dev-audio-only"
    parameters: str = "3.3B"
    text_encoder: str = DEFAULT_TEXT_ENCODER
    text_encoder_hidden_size: int = 3840
    model_path: str | None = None

    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    inference_defaults: InferenceDefaults = field(default_factory=InferenceDefaults)
    files: ModelFiles = field(default_factory=ModelFiles)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ModelConfig":
        transformer_config = dict(config.get("transformer", {}))
        if not transformer_config:
            transformer_config = {
                "num_layers": config.get("num_layers", 48),
                "audio_num_attention_heads": config.get(
                    "audio_num_attention_heads", 32
                ),
                "audio_attention_head_dim": config.get("audio_attention_head_dim", 64),
                "audio_cross_attention_dim": config.get(
                    "audio_cross_attention_dim", 2048
                ),
            }

        inference_defaults = dict(config.get("inference_defaults", {}))
        # The MLX port intentionally follows the warm reference server default.
        inference_defaults["rescale_scale"] = inference_defaults.get(
            "rescale_scale", "auto"
        )
        if inference_defaults["rescale_scale"] == 0.0:
            inference_defaults["rescale_scale"] = "auto"

        return cls(
            **_filter_fields(
                cls,
                {
                    **config,
                    "transformer": TransformerConfig.from_dict(transformer_config),
                    "audio": AudioConfig.from_dict(config.get("audio", {})),
                    "inference_defaults": InferenceDefaults.from_dict(
                        inference_defaults
                    ),
                    "files": ModelFiles.from_dict(config.get("files", {})),
                },
            )
        )
