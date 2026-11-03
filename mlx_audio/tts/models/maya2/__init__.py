from .config import ModelConfig
from .convert import copy_supporting_files, prepare_config, write_model_card
from .maya2 import Model

DETECTION_HINTS = {"model_type_aliases": ["maya3"]}

__all__ = [
    "Model",
    "ModelConfig",
    "copy_supporting_files",
    "prepare_config",
    "write_model_card",
]
