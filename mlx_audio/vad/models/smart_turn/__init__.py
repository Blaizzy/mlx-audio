from .config import EncoderConfig, ModelConfig, ProcessorConfig
from .smart_turn import EndpointOutput, Model

DETECTION_HINTS = {
    "architectures": ["smart_turn"],
    "config_keys": ["max_audio_seconds", "encoder_config", "processor_config"],
}

__all__ = [
    "EncoderConfig",
    "ProcessorConfig",
    "ModelConfig",
    "EndpointOutput",
    "Model",
    "DETECTION_HINTS",
]
