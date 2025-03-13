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

def adjust_speed(audio_array, speed_factor):
    """
    Adjust the speed of the audio by resampling
    speed_factor > 1: faster
    speed_factor < 1: slower
    """
    # Convert to numpy if it's an MLX array
    if isinstance(audio_array, mx.array):
        audio_array = audio_array.tolist()
    audio_array = np.array(audio_array)

    # Calculate new length
    old_length = len(audio_array)
    new_length = int(old_length / speed_factor)

    # Create new time points
    old_indices = np.arange(old_length)
    new_indices = np.linspace(0, old_length-1, new_length)

    # Resample using linear interpolation
    return np.interp(new_indices, old_indices, audio_array)


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
