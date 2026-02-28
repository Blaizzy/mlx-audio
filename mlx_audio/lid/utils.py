from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.utils import base_load_model, load_audio

SAMPLE_RATE = 16000

MODEL_REMAPPING = {}


def load_model(
    model_path: Union[str, Path], lazy: bool = False, strict: bool = False, **kwargs
) -> nn.Module:
    """
    Load and initialize a LID model from a given path.

    Args:
        model_path: The path or HuggingFace repo to load the model from.
        lazy: If False, evaluate model parameters immediately.
        strict: If True, raise an error if any weights are missing.
        **kwargs: Additional keyword arguments (revision, force_download).

    Returns:
        nn.Module: The loaded and initialized model.
    """
    return base_load_model(
        model_path=model_path,
        category="lid",
        model_remapping=MODEL_REMAPPING,
        lazy=lazy,
        strict=strict,
        **kwargs,
    )


def load(
    model_path: Union[str, Path], lazy: bool = False, strict: bool = False, **kwargs
) -> nn.Module:
    """
    Load a language identification model from a local path or HuggingFace repository.

    This is the main entry point for loading LID models. It automatically
    detects the model type and initializes the appropriate model class.

    Args:
        model_path: The local path or HuggingFace repo ID to load from.
        lazy: If False, evaluate model parameters immediately.
        strict: If True, raise an error if any weights are missing.
        **kwargs: Additional keyword arguments:
            - revision (str): HuggingFace revision/branch to use
            - force_download (bool): Force re-download of model files

    Returns:
        nn.Module: The loaded and initialized model.

    Example:
        >>> from mlx_audio.lid import load
        >>> model = load("facebook/mms-lid-256")
        >>> results = model.predict(audio)
    """
    return load_model(model_path, lazy=lazy, strict=strict, **kwargs)


def normalize_audio(audio: mx.array) -> mx.array:
    """Zero-mean unit-variance normalize audio for wav2vec2 input.

    Args:
        audio: Raw waveform, shape (T,) or (B, T).

    Returns:
        Normalized audio of shape (1, T) for 1D input, or (B, T) for 2D input.
    """
    if audio.ndim == 1:
        audio = audio[None, :]

    mean = mx.mean(audio, axis=-1, keepdims=True)
    var = mx.var(audio, axis=-1, keepdims=True)
    return (audio - mean) / mx.sqrt(var + 1e-7)
