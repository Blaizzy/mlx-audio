from .config import AudioConfig, ModelConfig
from .conversion import copy_supporting_files, prepare_config, write_model_card
from .ming_v3_asr import Model

DETECTION_HINTS = {"path_patterns": {"ming-3.0-asr-tiny", "ming_v3_asr"}}
