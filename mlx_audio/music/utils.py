"""Loading utilities for music generation models."""

from pathlib import Path
from typing import Any, Union

import mlx.nn as nn

from mlx_audio.utils import base_load_model

MODEL_REMAPPING = {"minimax_music3": "minimax_music3"}


def load_model(
    model_path: Union[str, Path],
    lazy: bool = False,
    strict: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Load a music model from a local path or Hugging Face repository."""
    return base_load_model(
        model_path=model_path,
        category="music",
        model_remapping=MODEL_REMAPPING,
        lazy=lazy,
        strict=strict,
        **kwargs,
    )


def load(
    model_path: Union[str, Path],
    lazy: bool = False,
    strict: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Alias for :func:`load_model`."""
    return load_model(model_path, lazy=lazy, strict=strict, **kwargs)


__all__ = ["MODEL_REMAPPING", "load", "load_model"]
