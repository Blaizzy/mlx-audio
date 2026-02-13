"""MOSS-TTS family package."""

from .request import MossNormalizedRequest

__all__ = ["MossNormalizedRequest"]

# Guard convert auto-detection during bootstrap: only advertise hints once the
# runtime model entry points exist.
try:
    from .model import Model, ModelConfig  # type: ignore[attr-defined]
except ImportError as exc:
    if exc.name != f"{__name__}.model":
        raise
else:
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
    __all__.extend(["Model", "ModelConfig", "DETECTION_HINTS"])
