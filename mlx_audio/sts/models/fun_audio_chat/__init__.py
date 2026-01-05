# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""FunAudioChat model for speech-to-speech and speech-to-text."""

from .audio_encoder import (
    FunAudioChatAudioAttention,
    FunAudioChatAudioEncoder,
    FunAudioChatAudioEncoderLayer,
)
from .config import (
    CRQTransformerConfig,
    FunAudioChatAudioEncoderConfig,
    FunAudioChatConfig,
    Qwen3Config,
)
from .discrete_encoder import FunAudioChatDiscreteEncoder
from .language_model import LanguageModel, Qwen3Model
from .model import FunAudioChatForConditionalGeneration, FunAudioChatOutput, Model
from .processor import FunAudioChatProcessor
from .speech_decoder import CRQTransformer, FunAudioChatDecoder

__all__ = [
    # Config classes
    "FunAudioChatConfig",
    "FunAudioChatAudioEncoderConfig",
    "Qwen3Config",
    "CRQTransformerConfig",
    # Model classes
    "FunAudioChatForConditionalGeneration",
    "FunAudioChatAudioEncoder",
    "FunAudioChatAudioEncoderLayer",
    "FunAudioChatAudioAttention",
    "FunAudioChatDiscreteEncoder",
    "FunAudioChatDecoder",
    "CRQTransformer",
    "LanguageModel",
    "Qwen3Model",
    # Processor
    "FunAudioChatProcessor",
    # Output
    "FunAudioChatOutput",
    "Model",
]
