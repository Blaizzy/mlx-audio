import inspect
from dataclasses import dataclass

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
    """
    Check if a conv weight array is already in MLX format.

    For 1D convolutions:
        MLX format: (out_channels, kernel_size, in_channels)
        PyTorch format: (out_channels, in_channels, kernel_size)

    Returns True if the array appears to be in MLX format (no transpose needed).
    Returns False if the array appears to be in PyTorch format (needs transpose).

    Heuristic: kernel dimensions are typically small (1, 3, 5, 7, 9, 11),
    while channel dimensions are typically larger (64, 128, 256, 512, etc.).
    """
    shape = arr.shape

    # Common kernel sizes for convolutions
    KERNEL_SIZE_THRESHOLD = 15

    if len(shape) != 3:
        return False

    out_channels, dim1, dim2 = shape

    # If middle dim is small (kernel-like) and last dim is large (channel-like): MLX format
    if dim1 <= KERNEL_SIZE_THRESHOLD and dim2 > KERNEL_SIZE_THRESHOLD:
        return True  # MLX format
    # If last dim is small (kernel-like) and middle dim is large (channel-like): PyTorch format
    elif dim2 <= KERNEL_SIZE_THRESHOLD and dim1 > KERNEL_SIZE_THRESHOLD:
        return False  # PyTorch format

    # Ambiguous case: both dims are small (both could be kernel-like)
    # Special handling when one dim is 1:
    # - in_channels=1 is common for certain operations
    # - kernel_size=1 (pointwise conv) is less common than kernel_size=3,5,7
    # So if dim1=1 and dim2>1, assume dim1 is in_channels (PyTorch format)
    if dim1 == 1 and dim2 > 1:
        return False  # Assume PyTorch format: (out, in=1, kernel)
    if dim2 == 1 and dim1 > 1:
        return True  # Assume MLX format: (out, kernel, in=1)

    # Both dims are similar and neither is 1
    # Kernel is typically smaller than or equal to in_channels
    if dim1 <= dim2:
        return True  # Assume MLX format (kernel in middle is smaller or equal)
    return False  # Assume PyTorch format


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
    sample_rate: int
    segment_idx: int
    token_count: int
    audio_samples: int
    audio_duration: str
    real_time_factor: float
    prompt: dict
    audio_samples: dict
    processing_time_seconds: float
    peak_memory_usage: float
