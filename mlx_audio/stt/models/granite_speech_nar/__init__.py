from .config import EncoderConfig, ModelConfig, ProjectorConfig, TextConfig
from .granite_speech_nar import Model

DETECTION_HINTS = {
    "config_keys": {"encoder_config", "projector_config", "encoder_layer_indices"},
    "architectures": {"GraniteSpeechNarForASR"},
}

__all__ = [
    "EncoderConfig",
    "ProjectorConfig",
    "TextConfig",
    "ModelConfig",
    "Model",
    "DETECTION_HINTS",
]
