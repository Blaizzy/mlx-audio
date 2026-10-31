"""Deprecated import path; use ``mlx_audio.tts.models.indextts2``."""

from mlx_audio.tts.models.indextts2.emotion import (
    CN_TO_EN,
    EMO_BIAS,
    EMOTION_KEYS,
    QwenEmotion,
    QwenEmotionConfig,
    normalize_emo_vector,
    parse_emotion_response,
)

__all__ = [
    "EMOTION_KEYS",
    "CN_TO_EN",
    "EMO_BIAS",
    "parse_emotion_response",
    "normalize_emo_vector",
    "QwenEmotionConfig",
    "QwenEmotion",
]
