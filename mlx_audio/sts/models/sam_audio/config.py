# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass, field
from typing import Optional

from mlx_audio.codec.models.dacvae.codec import DACVAEConfig


@dataclass
class T5EncoderConfig:
    """Configuration for the T5 text encoder."""

    name: str = "t5-base"
    max_length: Optional[int] = 512
    pad_mode: str = "longest"
    dim: int = 768


@dataclass
class TransformerConfig:
    """Configuration for the DiT (Diffusion Transformer).

    Note: SAM-Audio operates in codebook space (128 dim), so:
    - out_channels = 2 * codebook_dim = 256 (target + residual)
    - context_dim = transformer.dim (memory is projected to this dim)
    """

    dim: int = 2816  # Default from sam-audio-large
    n_heads: int = 22
    n_layers: int = 22
    dropout: float = 0.1
    norm_eps: float = 1.0e-05
    qk_norm: bool = True
    fc_bias: bool = False
    ffn_exp: int = 4
    ffn_dim_multiplier: int = 1
    multiple_of: int = 64
    non_linearity: str = "swiglu"
    use_rope: bool = True
    max_positions: int = 10000
    frequency_embedding_dim: int = 256
    timestep_non_linearity: str = "swiglu"
    t_block_non_linearity: str = "silu"
    t_block_bias: bool = True
    # context_dim should match transformer dim (memory is projected before passing to DiT)
    context_dim: int = 2816
    context_non_linearity: str = "swiglu"
    context_embedder_dropout: float = 0.0
    context_norm: bool = False
    # out_channels = 2 * codebook_dim (for target + residual features in codebook space)
    out_channels: int = 256
    in_channels: Optional[int] = None


@dataclass
class SAMAudioConfig:
    """Main configuration for SAMAudio model.

    Note: SAM-Audio operates in the VAE codebook space (codebook_dim=128), not the
    raw latent space (latent_dim=1024). This means:
    - in_channels = 6 * codebook_dim = 768 (for [noisy, zeros, features] x 2)
    - out_channels = 2 * codebook_dim = 256 (for target + residual)
    """

    # in_channels = 6 * codebook_dim (for concat of [noisy, zeros, features] where features is doubled)
    # Default: 6 * 128 = 768
    in_channels: int = 768
    audio_codec: DACVAEConfig = field(default_factory=DACVAEConfig)
    text_encoder: T5EncoderConfig = field(default_factory=T5EncoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    num_anchors: int = 3
    anchor_embedding_dim: int = 128

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SAMAudioConfig":
        """Create config from dictionary."""
        audio_codec = DACVAEConfig(**config_dict.get("audio_codec", {}))
        text_encoder = T5EncoderConfig(**config_dict.get("text_encoder", {}))

        # Get transformer config - use values from config as-is
        transformer_dict = config_dict.get("transformer", {}).copy()
        transformer = TransformerConfig(**transformer_dict)

        # Use in_channels from config
        # SAM-Audio operates in codebook space: in_channels = 6 * codebook_dim = 768
        in_channels = config_dict.get("in_channels", 6 * audio_codec.codebook_dim)

        return cls(
            in_channels=in_channels,
            audio_codec=audio_codec,
            text_encoder=text_encoder,
            transformer=transformer,
            num_anchors=config_dict.get("num_anchors", 3),
            anchor_embedding_dim=config_dict.get("anchor_embedding_dim", 128),
        )


# Predefined configurations for different model sizes
# Note: SAM-Audio operates in codebook space (128 dim)
# - in_channels = 6 * codebook_dim = 768
# - out_channels = 2 * codebook_dim = 256
# - context_dim should match transformer dim (memory is projected before DiT)
SAM_AUDIO_SMALL_CONFIG = SAMAudioConfig(
    in_channels=768,  # 6 * codebook_dim
    audio_codec=DACVAEConfig(),
    text_encoder=T5EncoderConfig(name="t5-base", dim=768),
    transformer=TransformerConfig(
        dim=1024,
        n_heads=8,
        n_layers=12,
        context_dim=1024,  # Match transformer dim
        out_channels=256,  # 2 * codebook_dim
    ),
)

SAM_AUDIO_BASE_CONFIG = SAMAudioConfig(
    in_channels=768,  # 6 * codebook_dim
    audio_codec=DACVAEConfig(),
    text_encoder=T5EncoderConfig(name="t5-base", dim=768),
    transformer=TransformerConfig(
        dim=1536,
        n_heads=12,
        n_layers=16,
        context_dim=1536,  # Match transformer dim
        out_channels=256,  # 2 * codebook_dim
    ),
)

SAM_AUDIO_LARGE_CONFIG = SAMAudioConfig(
    in_channels=768,  # 6 * codebook_dim
    audio_codec=DACVAEConfig(),
    text_encoder=T5EncoderConfig(name="t5-base", dim=768),
    transformer=TransformerConfig(
        dim=2816,
        n_heads=22,
        n_layers=22,
        context_dim=2816,  # Match transformer dim
        out_channels=256,  # 2 * codebook_dim
    ),
)
