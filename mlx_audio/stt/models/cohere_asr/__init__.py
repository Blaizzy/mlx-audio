from .cohere_asr import Model
from .config import ModelConfig

DETECTION_HINTS = {
    "architectures": {
        "CohereAsrForConditionalGeneration",
        "CohereAsrModel",
    },
    "config_keys": {
        "max_audio_clip_s",
        "overlap_chunk_second",
        "supported_languages",
    },
}

__all__ = ["Model", "ModelConfig", "DETECTION_HINTS"]
