"""MOSS-TTS-Realtime model package."""

from .inference import (
    AudioStreamDecoder,
    MossTTSRealtimeInference,
    RealtimeSession,
    RealtimeTextDeltaBridge,
    TextDeltaTokenizer,
    bridge_text_stream,
)
from .model import Model, ModelConfig
from .request import RealtimeNormalizedRequest

DETECTION_HINTS = {
    "architectures": ["MossTTSRealtime"],
    "config_keys": [
        "audio_vocab_size",
        "audio_pad_token",
        "reference_audio_pad",
        "text_pad",
        "rvq",
        "local_config",
    ],
    "path_patterns": {
        "moss_tts_realtime",
        "moss-tts-realtime",
        "mossrealtime",
    },
}

__all__ = [
    "AudioStreamDecoder",
    "MossTTSRealtimeInference",
    "Model",
    "ModelConfig",
    "RealtimeNormalizedRequest",
    "RealtimeSession",
    "RealtimeTextDeltaBridge",
    "TextDeltaTokenizer",
    "bridge_text_stream",
    "DETECTION_HINTS",
]
