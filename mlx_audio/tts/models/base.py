import inspect
import re
from dataclasses import dataclass
from typing import List

import mlx.core as mx
import numpy as np


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 3:
        return False

    out_channels, kH, KW = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


def split_text_into_chunks(text: str, max_chars: int = 300) -> List[str]:
    """
    Split text into smaller chunks that can be processed individually.
    Tries to split on sentence boundaries when possible.

    Args:
        text (str): The text to split
        max_chars (int): Maximum characters per chunk

    Returns:
        List[str]: List of text chunks
    """
    # Try to split on sentence boundaries first
    chunks = []
    sentences = re.split(r"([.!?]+)", text)
    current_chunk = ""

    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        # Add the punctuation back if it exists
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]

        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # If no chunks were created (no sentence boundaries), fall back to character-based chunking
    if not chunks:
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    return [chunk for chunk in chunks if chunk.strip()]


def adjust_speed(audio_array, speed_factor):
    """
    Adjust the speed of the audio by resampling
    speed_factor > 1: faster
    speed_factor < 1: slower
    """
    # Ensure we're working with MLX arrays
    if not isinstance(audio_array, mx.array):
        audio_array = mx.array(audio_array)

    # Calculate new length
    old_length = audio_array.shape[0]
    new_length = int(old_length / speed_factor)

    # Create new time points
    old_indices = mx.arange(old_length)
    new_indices = mx.linspace(0, old_length - 1, new_length)

    # Resample using linear interpolation
    # Since mx doesn't have interp, we'll implement it directly
    indices_floor = mx.floor(new_indices).astype(mx.int32)
    indices_ceil = mx.minimum(indices_floor + 1, old_length - 1)
    weights_ceil = new_indices - indices_floor
    weights_floor = 1.0 - weights_ceil

    # Perform the interpolation
    result = (
        weights_floor.reshape(-1, 1) * audio_array[indices_floor]
        + weights_ceil.reshape(-1, 1) * audio_array[indices_ceil]
    )

    return result


@dataclass
class GenerationResult:
    audio: mx.array
    samples: int
    segment_idx: int
    token_count: int
    audio_samples: int
    audio_duration: str
    real_time_factor: float
    prompt: dict
    audio_samples: dict
    processing_time_seconds: float
    peak_memory_usage: float
