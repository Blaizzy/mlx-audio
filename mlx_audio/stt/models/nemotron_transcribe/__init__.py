from .config import ModelConfig
from .nemotron_transcribe import Model

DETECTION_HINTS = {
    "model_type_aliases": ["nemo_speechlm"],
    "architectures": ["NeMoSpeechLMForConditionalGeneration"],
}

__all__ = ["Model", "ModelConfig"]
