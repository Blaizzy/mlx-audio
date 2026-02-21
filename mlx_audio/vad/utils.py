from pathlib import Path
from typing import Union

import mlx.nn as nn

from mlx_audio.utils import base_load_model

MODEL_REMAPPING = {}


def load_model(
    model_path: Union[str, Path], lazy: bool = False, strict: bool = False, **kwargs
) -> nn.Module:
    """
    Load and initialize a VAD/diarization model from a given path.

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
        category="vad",
        model_remapping=MODEL_REMAPPING,
        lazy=lazy,
        strict=strict,
        **kwargs,
    )


def load(
    model_path: Union[str, Path], lazy: bool = False, strict: bool = False, **kwargs
) -> nn.Module:
    """
    Load a VAD/diarization model from a local path or HuggingFace repository.

    This is the main entry point for loading VAD models. It automatically
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
        >>> from mlx_audio.vad import load
        >>> model = load("mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16")
        >>> result = model.generate("audio.wav", verbose=True)
    """
    return load_model(model_path, lazy=lazy, strict=strict, **kwargs)
