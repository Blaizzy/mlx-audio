"""Native MLX implementation of MiniMax Music 3.

Adapted and modified from mikolaj92/minimax-music3-mlx under Apache-2.0.
See LICENSE and NOTICE.
"""

from .conversion import copy_supporting_files, load_source_weights, prepare_config
from .minimax_music3 import Model, ModelConfig

DOWNLOAD_ALLOW_PATTERNS = [
    "language_model/*.json",
    "language_model/*.safetensors",
    "rvq_depth_decoder/*.json",
    "rvq_depth_decoder/*.safetensors",
    "condition_encoder/*.json",
    "condition_encoder/*.safetensors",
    "transformer/*.json",
    "transformer/*.safetensors",
    "vocoder/*.json",
    "vocoder/*.safetensors",
    "tokenizer/*",
    "scheduler/*",
    "modular_model_index.json",
    "README.md",
    "LICENSE*",
]

DETECTION_HINTS = {
    "model_type_aliases": {
        "minimax-music3",
        "minimaxmusic3modularpipeline",
        "minimaxmusic3forconditionalgeneration",
    },
    "architectures": {
        "MiniMaxMusic3ModularPipeline",
        "MiniMaxMusic3ForConditionalGeneration",
    },
    "path_patterns": {"minimax-music3", "minimax_music3"},
}

__all__ = [
    "Model",
    "ModelConfig",
    "DOWNLOAD_ALLOW_PATTERNS",
    "copy_supporting_files",
    "load_source_weights",
    "prepare_config",
]
