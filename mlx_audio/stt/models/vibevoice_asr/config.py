# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _filter_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only include fields defined in the dataclass."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return kwargs
    field_names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in kwargs.items() if k in field_names}


@dataclass
class AcousticTokenizerConfig:
    """Configuration for VibeVoice acoustic tokenizer encoder."""

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

    # Decoder specific (for acoustic tokenizer only)
    decoder_n_filters: int = 32
    decoder_ratios: Optional[List[int]] = None
    decoder_depths: Optional[str] = None

    # Dtype (ignored in MLX, kept for compatibility)
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.decoder_ratios is None:
            self.decoder_ratios = self.encoder_ratios

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AcousticTokenizerConfig":
        return cls(**_filter_kwargs(cls, d))

    @property
    def parsed_encoder_depths(self) -> List[int]:
        """Parse encoder depths string to list of ints."""
        if isinstance(self.encoder_depths, str):
            return [int(d) for d in self.encoder_depths.split("-")]
        return self.encoder_depths


@dataclass
class SemanticTokenizerConfig:
    """Configuration for VibeVoice semantic tokenizer encoder."""

    model_type: str = "vibevoice_semantic_tokenizer"
    channels: int = 1
    corpus_normalize: float = 0.0
    causal: bool = True
    vae_dim: int = 128
    fix_std: float = 0
    std_dist_type: str = "none"

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

    # Dtype (ignored in MLX, kept for compatibility)
    dtype: str = "bfloat16"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SemanticTokenizerConfig":
        return cls(**_filter_kwargs(cls, d))

    @property
    def parsed_encoder_depths(self) -> List[int]:
        """Parse encoder depths string to list of ints."""
        if isinstance(self.encoder_depths, str):
            return [int(d) for d in self.encoder_depths.split("-")]
        return self.encoder_depths


@dataclass
class Qwen2Config:
    """Configuration for Qwen2-based language model decoder."""

    model_type: str = "qwen2"
    vocab_size: int = 152064
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    intermediate_size: int = 18944
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False

    # Layer types (all full attention for ASR model)
    layer_types: List[str] = field(default_factory=lambda: ["full_attention"] * 28)

    # Sliding window (not used in ASR)
    sliding_window: Optional[int] = None
    use_sliding_window: bool = False
    max_window_layers: int = 28

    # RoPE scaling
    rope_scaling: Optional[dict] = None
    use_mrope: bool = False

    # Additional fields for mlx_lm compatibility
    rope_traditional: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Qwen2Config":
        return cls(**_filter_kwargs(cls, d))

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class ModelConfig:
    """Main configuration for VibeVoice-ASR model."""

    model_type: str = "vibevoice"

    # Sub-configs
    acoustic_tokenizer_config: AcousticTokenizerConfig = field(
        default_factory=AcousticTokenizerConfig
    )
    semantic_tokenizer_config: SemanticTokenizerConfig = field(
        default_factory=SemanticTokenizerConfig
    )
    decoder_config: Qwen2Config = field(default_factory=Qwen2Config)

    # VAE dimensions (derived from tokenizer configs)
    acoustic_vae_dim: int = 64
    semantic_vae_dim: int = 128

    # Global dtype
    dtype: str = "float32"

    # Sample rate
    sample_rate: int = 24000

    # Speech token compression ratio
    speech_tok_compress_ratio: int = 3200

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create config from dictionary (loaded from config.json)."""
        # Parse sub-configs
        acoustic_config = config_dict.get("acoustic_tokenizer_config", {})
        semantic_config = config_dict.get("semantic_tokenizer_config", {})
        decoder_config = config_dict.get("decoder_config", {})

        return cls(
            model_type=config_dict.get("model_type", "vibevoice"),
            acoustic_tokenizer_config=AcousticTokenizerConfig.from_dict(
                acoustic_config
            ),
            semantic_tokenizer_config=SemanticTokenizerConfig.from_dict(
                semantic_config
            ),
            decoder_config=Qwen2Config.from_dict(decoder_config),
            acoustic_vae_dim=config_dict.get("acoustic_vae_dim", 64),
            semantic_vae_dim=config_dict.get("semantic_vae_dim", 128),
            dtype=config_dict.get("dtype", "float32"),
        )
