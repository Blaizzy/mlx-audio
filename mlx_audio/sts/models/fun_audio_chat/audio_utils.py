# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Audio utilities for FunAudioChat S2S mode."""

from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

# Audio token constants
TOKEN_FPS = 25  # Audio tokens per second (25 Hz)
COSYVOICE_SAMPLE_RATE = 22050


def filter_audio_tokens(
    audio_tokens: Union[mx.array, List[int]],
    codebook_size: int = 6561,
) -> List[int]:
    """Filter audio tokens to valid range [0, codebook_size-1]."""
    if isinstance(audio_tokens, mx.array):
        tokens = audio_tokens.tolist()
    else:
        tokens = list(audio_tokens)
    return [t for t in tokens if 0 <= t < codebook_size]


def estimate_audio_duration(num_tokens: int, token_fps: int = TOKEN_FPS) -> float:
    """Estimate audio duration in seconds from number of tokens."""
    return num_tokens / token_fps


def save_audio_tokens(
    audio_tokens: Union[mx.array, List[int]],
    path: str,
) -> None:
    """Save audio tokens to a .npz file for later decoding."""
    if isinstance(audio_tokens, mx.array):
        tokens = np.array(audio_tokens.tolist(), dtype=np.int32)
    else:
        tokens = np.array(audio_tokens, dtype=np.int32)
    np.savez(path, tokens=tokens)


def load_audio_tokens(path: str) -> np.ndarray:
    """Load audio tokens from a .npz file."""
    data = np.load(path, allow_pickle=True)
    return data["tokens"]
