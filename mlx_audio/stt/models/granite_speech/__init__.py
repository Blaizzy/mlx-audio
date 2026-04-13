from .config import EncoderConfig, ModelConfig, ProjectorConfig, TextConfig
from .granite_speech import Model

DETECTION_HINTS = {
    "config_keys": {"encoder_config", "projector_config", "audio_token_index"},
    "architectures": {"GraniteSpeechForConditionalGeneration"},
}

__all__ = [
    "EncoderConfig",
    "ProjectorConfig",
    "TextConfig",
    "ModelConfig",
    "Model",
    "DETECTION_HINTS",
]
