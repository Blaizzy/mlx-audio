# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""FunAudioChat model for speech-to-speech and speech-to-text."""

from .audio_encoder import (
    FunAudioChatAudioAttention,
    FunAudioChatAudioEncoder,
    FunAudioChatAudioEncoderLayer,
)
from .audio_utils import (
    TOKEN_FPS,
    estimate_audio_duration,
    filter_audio_tokens,
    load_audio_tokens,
    save_audio_tokens,
)
from .config import (
    CRQTransformerConfig,
    FunAudioChatAudioEncoderConfig,
    FunAudioChatConfig,
    Qwen3Config,
)
from .cosyvoice_decoder import (
    CosyVoiceDecoder,
    CosyVoiceDecoderConfig,
    decode_audio_tokens,
)
from .discrete_encoder import FunAudioChatDiscreteEncoder
from .language_model import LanguageModel, Qwen3Model
from .model import (
    AUDIO_TEMPLATE,
    DEFAULT_S2M_PROMPT,
    DEFAULT_S2T_PROMPT,
    SPOKEN_S2M_PROMPT,
    FunAudioChatForConditionalGeneration,
    FunAudioChatOutput,
    Model,
)
from .processor import FunAudioChatProcessor
from .speech_decoder import CRQTransformer, FunAudioChatDecoder

__all__ = [
    # Config classes
    "FunAudioChatConfig",
    "FunAudioChatAudioEncoderConfig",
    "Qwen3Config",
    "CRQTransformerConfig",
    "CosyVoiceDecoderConfig",
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
    # Audio decoder
    "CosyVoiceDecoder",
    "decode_audio_tokens",
    # Processor
    "FunAudioChatProcessor",
    # Output
    "FunAudioChatOutput",
    "Model",
    # S2S Constants
    "DEFAULT_S2T_PROMPT",
    "DEFAULT_S2M_PROMPT",
    "SPOKEN_S2M_PROMPT",
    "AUDIO_TEMPLATE",
    "TOKEN_FPS",
    # Audio utilities
    "filter_audio_tokens",
    "estimate_audio_duration",
    "save_audio_tokens",
    "load_audio_tokens",
]
