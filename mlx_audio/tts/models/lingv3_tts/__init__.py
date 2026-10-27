from .config import ModelConfig
from .conversion import copy_supporting_files, prepare_config, write_model_card
from .lingv3_tts import Model

DETECTION_HINTS = {"path_patterns": {"ming-3.0-tts-tiny", "lingv3_tts"}}
