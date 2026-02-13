"""MOSS-TTS family package scaffolding.

Phase 2 will add the model implementation modules under this package.
"""

from .request import MossNormalizedRequest

# Used by mlx_audio.convert dynamic detection.
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
    },
}

__all__ = ["DETECTION_HINTS", "MossNormalizedRequest"]

