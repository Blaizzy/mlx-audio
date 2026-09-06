# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from .config import (
    AcousticTokenizerConfig,
    ModelConfig,
    Qwen2Config,
    SemanticTokenizerConfig,
)
from .vibevoice_asr import Model

DETECTION_HINTS = {
    "model_type_aliases": ["vibevoice"],
    "architectures": [
        "VibeVoiceForASRTraining",
        "VibeVoiceForASRStreamingTraining",
    ],
    "path_patterns": ["vibevoice-asr", "vibevoice_asr"],
}

__all__ = [
    "Model",
    "ModelConfig",
    "AcousticTokenizerConfig",
    "SemanticTokenizerConfig",
    "Qwen2Config",
    "DETECTION_HINTS",
]
