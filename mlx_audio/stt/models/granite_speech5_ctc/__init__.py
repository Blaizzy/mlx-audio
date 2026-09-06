from .config import EncoderConfig, ModelConfig
from .granite_speech5 import Model

DETECTION_HINTS = {
    "model_type_aliases": {"granite_speech5_ctc"},
    "architectures": {"GraniteSpeech5ForCTC"},
    "path_patterns": {"granite-speech-5.0", "turboctc"},
}

__all__ = ["EncoderConfig", "Model", "ModelConfig", "DETECTION_HINTS"]
