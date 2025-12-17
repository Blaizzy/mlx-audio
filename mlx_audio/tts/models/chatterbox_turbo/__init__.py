# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from .chatterbox_turbo import ChatterboxTurboTTS, Conditionals, punc_norm
from .models.s3gen import S3GEN_SIL, S3GEN_SR, S3Gen
from .models.t3 import T3, T3Cond, T3Config
from .models.voice_encoder import VoiceEncoder

# Alias for load_model compatibility
Model = ChatterboxTurboTTS

__all__ = [
    "ChatterboxTurboTTS",
    "Model",
    "Conditionals",
    "punc_norm",
    "T3",
    "T3Config",
    "T3Cond",
    "S3Gen",
    "S3GEN_SR",
    "S3GEN_SIL",
    "VoiceEncoder",
]
