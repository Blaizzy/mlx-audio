"""MOSS-TTS family package."""

from .model import Model, ModelConfig
from .pronunciation import (
    PronunciationHelperUnavailableError,
    convert_text_to_ipa,
    convert_text_to_tone_numbered_pinyin,
    validate_pronunciation_input_contract,
)
from .request import MossNormalizedRequest

DETECTION_HINTS = {
    "architectures": ["MossTTSDelayModel"],
    "config_keys": [
        "audio_vocab_size",
        "audio_start_token_id",
        "audio_end_token_id",
        "n_vq",
    ],
    "path_patterns": {
        "moss_tts",
        "moss-tts",
        "moss_tts_local",
        "moss-tts-local",
        "moss_ttsd",
        "moss-ttsd",
        "moss_voice_generator",
        "moss-voice-generator",
        "moss_sound_effect",
        "moss-soundeffect",
        "moss_soundeffect",
    },
}

__all__ = [
    "Model",
    "ModelConfig",
    "MossNormalizedRequest",
    "PronunciationHelperUnavailableError",
    "convert_text_to_ipa",
    "convert_text_to_tone_numbered_pinyin",
    "validate_pronunciation_input_contract",
    "DETECTION_HINTS",
]
