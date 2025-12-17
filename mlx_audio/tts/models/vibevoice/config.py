# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class AcousticTokenizerConfig(BaseModelArgs):
    """Configuration for the acoustic tokenizer (VAE decoder)."""

    model_type: str = "vibevoice_acoustic_tokenizer"
    channels: int = 1
    corpus_normalize: float = 0.0
    causal: bool = True
    vae_dim: int = 64
    fix_std: float = 0.5
    std_dist_type: str = "gaussian"

    # Common parameters
    mixer_layer: str = "depthwise_conv"
    conv_norm: str = "none"
    pad_mode: str = "constant"
    disable_last_norm: bool = True
    layernorm: str = "RMSNorm"
    layernorm_eps: float = 1e-5
    layernorm_elementwise_affine: bool = True
    conv_bias: bool = True
    layer_scale_init_value: float = 1e-6
    weight_init_value: float = 0.01

    # Encoder specific
    encoder_n_filters: int = 32
    encoder_ratios: List[int] = field(default_factory=lambda: [8, 5, 5, 4, 2, 2])
    encoder_depths: str = "3-3-3-3-3-3-8"

    # Decoder specific
    decoder_n_filters: int = 32
    decoder_ratios: Optional[List[int]] = None
    decoder_depths: Optional[str] = None


@dataclass
class DiffusionHeadConfig(BaseModelArgs):
    """Configuration for the diffusion prediction head."""

    model_type: str = "vibevoice_diffusion_head"
    hidden_size: int = 896
    head_layers: int = 4
    head_ffn_ratio: float = 3.0
    rms_norm_eps: float = 1e-5
    latent_size: int = 64
    speech_vae_dim: Optional[int] = 64
    prediction_type: str = "v_prediction"
    diffusion_type: str = "ddpm"
    ddpm_num_steps: int = 1000
    ddpm_num_inference_steps: int = 20
    ddpm_beta_schedule: str = "cosine"
    ddpm_batch_mul: int = 4


@dataclass
class Qwen2DecoderConfig(BaseModelArgs):
    """Configuration for the Qwen2 decoder backbone."""

    model_type: str = "qwen2"
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    hidden_size: int = 896
    initializer_range: float = 0.02
    intermediate_size: int = 4864
    max_position_embeddings: int = 8192
    max_window_layers: int = 24
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[Dict] = None
    rope_theta: float = 1000000.0
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936
    head_dim: Optional[int] = None


@dataclass
class ModelConfig(BaseModelArgs):
    """Main configuration for VibeVoice streaming model."""

    model_type: str = "vibevoice_streaming"
    model_path: Optional[str] = None
    sample_rate: int = 24000

    # Sub-configurations
    acoustic_tokenizer_config: AcousticTokenizerConfig = field(
        default_factory=AcousticTokenizerConfig
    )
    decoder_config: Qwen2DecoderConfig = field(default_factory=Qwen2DecoderConfig)
    diffusion_head_config: DiffusionHeadConfig = field(
        default_factory=DiffusionHeadConfig
    )

    # Model architecture parameters
    acoustic_vae_dim: int = 64
    tts_backbone_num_hidden_layers: int = 20

    @classmethod
    def from_dict(cls, params: dict) -> "ModelConfig":
        """Create config from a dictionary."""
        # Handle nested configs
        acoustic_cfg = params.pop("acoustic_tokenizer_config", {})
        decoder_cfg = params.pop("decoder_config", {})
        diffusion_cfg = params.pop("diffusion_head_config", {})

        # Create sub-configs
        if isinstance(acoustic_cfg, dict):
            acoustic_config = AcousticTokenizerConfig.from_dict(acoustic_cfg)
        else:
            acoustic_config = acoustic_cfg

        if isinstance(decoder_cfg, dict):
            decoder_config = Qwen2DecoderConfig.from_dict(decoder_cfg)
        else:
            decoder_config = decoder_cfg

        if isinstance(diffusion_cfg, dict):
            diffusion_config = DiffusionHeadConfig.from_dict(diffusion_cfg)
        else:
            diffusion_config = diffusion_cfg

        # Filter main config params
        config = cls(
            acoustic_tokenizer_config=acoustic_config,
            decoder_config=decoder_config,
            diffusion_head_config=diffusion_config,
            **{
                k: v
                for k, v in params.items()
                if hasattr(cls, k) or k in cls.__dataclass_fields__
            }
        )

        return config
