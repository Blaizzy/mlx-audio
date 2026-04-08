from pathlib import Path
from typing import Union

import mlx.nn as nn

from mlx_audio.utils import base_load_model

MODEL_REMAPPING = {
    "lfm_audio": "lfm_audio",
    "deepfilternet": "deepfilternet",
    "deepfilter": "deepfilternet",
    "mossformer2_se": "mossformer2_se",
    "mossformer2": "mossformer2_se",
}


def load_model(
    model_path: Union[str, Path], lazy: bool = False, strict: bool = False, **kwargs
) -> nn.Module:
    """
    Load and initialize an STS (speech-to-speech) model from a given path.

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
        category="sts",
        model_remapping=MODEL_REMAPPING,
        lazy=lazy,
        strict=strict,
        **kwargs,
    )


def load(
    model_path: Union[str, Path], lazy: bool = False, strict: bool = False, **kwargs
) -> nn.Module:
    """
    Load an STS model from a local path or HuggingFace repository.

    This is the main entry point for loading STS models. It automatically
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
    """
    return load_model(model_path, lazy=lazy, strict=strict, **kwargs)
