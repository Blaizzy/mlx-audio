from .config import ModelConfig
from .conversion import copy_supporting_files, prepare_config, write_model_card
from .ming_v3_audio import GenerationResult, Model

DETECTION_HINTS = {
    "model_type_aliases": {"bailing_moe_v3_audio", "ming_v3_audio"},
    "path_patterns": {"ming-3.0-audio-tiny", "ming_v3_audio"},
}
