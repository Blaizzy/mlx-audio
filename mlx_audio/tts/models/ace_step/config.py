# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mlx_audio.tts.models.base import BaseModelArgs

# ==============================================================================
# Task Type Constants
# ==============================================================================

TASK_TYPES = ["text2music", "repaint", "cover", "extract", "lego", "complete"]


# ==============================================================================
# Instruction Constants
# ==============================================================================

DEFAULT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

# Instruction templates for each task type
# Note: Some instructions use placeholders like {TRACK_NAME} or {TRACK_CLASSES}
TASK_INSTRUCTIONS: Dict[str, str] = {
    "text2music": "Fill the audio semantic mask based on the given conditions:",
    "repaint": "Repaint the mask area based on the given conditions:",
    "cover": "Generate audio semantic tokens based on the given conditions:",
    "extract": "Extract the {TRACK_NAME} track from the audio:",
    "extract_default": "Extract the track from the audio:",
    "lego": "Generate the {TRACK_NAME} track based on the audio context:",
    "lego_default": "Generate the track based on the audio context:",
    "complete": "Complete the input track with {TRACK_CLASSES}:",
    "complete_default": "Complete the input track:",
}


# ==============================================================================
# Track/Instrument Constants
# ==============================================================================

TRACK_NAMES = [
    "woodwinds",
    "brass",
    "fx",
    "synth",
    "strings",
    "percussion",
    "keyboard",
    "guitar",
    "bass",
    "drums",
    "backing_vocals",
    "vocals",
]


# ==============================================================================
# SFT Prompt Template
# ==============================================================================

SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""


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


