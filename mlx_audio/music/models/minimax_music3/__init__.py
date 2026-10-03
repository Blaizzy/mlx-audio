"""Native MLX implementation of MiniMax Music 3."""

from .conversion import copy_supporting_files, load_source_weights, prepare_config
from .minimax_music3 import Model, ModelConfig

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
    "copy_supporting_files",
    "load_source_weights",
    "prepare_config",
]
