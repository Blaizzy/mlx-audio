# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


from dataclasses import dataclass, field
from typing import List, Optional

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class ModelConfig(BaseModelArgs):

    # Model architecture
    model_type: str = "acestep"
    model_version: str = "turbo"

    # Vocabulary and quantization
    vocab_size: int = 64003
    fsq_dim: int = 2048
    fsq_input_levels: List[int] = field(default_factory=lambda: [8, 8, 8, 5, 5, 5])
    fsq_input_num_quantizers: int = 1

    # Core dimensions
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128

    # Activation and normalization
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # Position embeddings
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None

    # Attention settings
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_sliding_window: bool = True
    sliding_window: int = 128
    layer_types: Optional[List[str]] = None  # Will be set in __post_init__

    # Encoder configurations
    text_hidden_dim: int = 1024
    num_lyric_encoder_hidden_layers: int = 8
    num_timbre_encoder_hidden_layers: int = 4
    num_attention_pooler_hidden_layers: int = 2

    # Audio configuration
    audio_acoustic_hidden_dim: int = 64
    timbre_hidden_dim: int = 64
    timbre_fix_frame: int = 750
    pool_window_size: int = 5
    in_channels: int = 192
    patch_size: int = 2

    # Flow matching parameters
    data_proportion: float = 0.5
    timestep_mu: float = -0.4
    timestep_sigma: float = 1.0

    # Generation settings
    use_cache: bool = True
    is_turbo: bool = True

    # Audio output
    sample_rate: int = 48000

    def __post_init__(self):
        """Set default layer types if not provided."""
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


@dataclass
class VAEConfig(BaseModelArgs):

    audio_channels: int = 2
    channel_multiples: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    decoder_channels: int = 128
    decoder_input_channels: int = 64
    downsampling_ratios: List[int] = field(default_factory=lambda: [2, 4, 4, 6, 10])
    encoder_hidden_size: int = 128
    sampling_rate: int = 48000


@dataclass
class TextEncoderConfig(BaseModelArgs):

    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    vocab_size: int = 151669
    tie_word_embeddings: bool = True
