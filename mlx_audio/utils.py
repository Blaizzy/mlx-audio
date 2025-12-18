"""Utility functions for mlx_audio.

This module provides a unified interface for loading TTS and STT models,
with lazy imports to avoid loading unnecessary dependencies.
"""

import importlib.util
from pathlib import Path
from typing import List, Optional, Union

from mlx_audio.dsp import (
    STR_TO_WINDOW_FN,
    bartlett,
    blackman,
    hamming,
    hanning,
    istft,
    mel_filters,
    stft,
)

# Lazy-loaded modules
_stt_utils = None
_tts_utils = None


def _get_stt_utils():
    """Lazy load STT utils."""
    global _stt_utils
    if _stt_utils is None:
        from mlx_audio.stt import utils as stt_utils

        _stt_utils = stt_utils
    return _stt_utils


def _get_tts_utils():
    """Lazy load TTS utils."""
    global _tts_utils
    if _tts_utils is None:
        from mlx_audio.tts import utils as tts_utils

        _tts_utils = tts_utils
    return _tts_utils


__all__ = [
    # DSP functions (re-exported from dsp.py)
    "hanning",
    "hamming",
    "blackman",
    "bartlett",
    "STR_TO_WINDOW_FN",
    "stft",
    "istft",
    "mel_filters",
    # Model utilities
    "is_valid_module_name",
    "get_model_category",
    "get_model_name_parts",
    "load_model",
]


def is_valid_module_name(name: str) -> bool:
    """Check if a string is a valid Python module name."""
    if not name or not isinstance(name, str):
        return False

    return name[0].isalpha() or name[0] == "_"


def get_model_category(model_type: str, model_name: List[str]) -> Optional[str]:
    """Determine whether a model belongs to the TTS or STT category."""
    stt_utils = _get_stt_utils()
    tts_utils = _get_tts_utils()

    candidates = [model_type] + (model_name or [])

    for category, remap in (
        ("tts", tts_utils.MODEL_REMAPPING),
        ("stt", stt_utils.MODEL_REMAPPING),
    ):
        for hint in candidates:
            arch = remap.get(hint, hint)
            # Double-check that the architecture name is valid before trying to import
            if not is_valid_module_name(arch):
                continue
            module_path = f"mlx_audio.{category}.models.{arch}"
            if importlib.util.find_spec(module_path) is not None:
                return category

    return None


def get_model_name_parts(model_path: Union[str, Path]) -> str:
    model_name = None
    if isinstance(model_path, str):
        model_name = model_path.lower().split("/")[-1].split("-")
    elif isinstance(model_path, Path):
        index = model_path.parts.index("hub")
        model_name = model_path.parts[index + 1].lower().split("--")[-1].split("-")
    else:
        raise ValueError(f"Invalid model path type: {type(model_path)}")
    return model_name


def load_model(model_name: str):
    """Load a TTS or STT model based on its configuration and name.

    Args:
        model_name (str): Name or path of the model to load

    Returns:
        The loaded model instance

    Raises:
        ValueError: If the model type cannot be determined or is not supported
    """
    tts_utils = _get_tts_utils()
    stt_utils = _get_stt_utils()

    config = tts_utils.load_config(model_name)
    model_name_parts = get_model_name_parts(model_name)

    # Try to determine model type from config first, then from name
    model_type = config.get("model_type", None)
    model_category = get_model_category(model_type, model_name_parts)

    if not model_category:
        raise ValueError(f"Could not determine model type for {model_name}")

    model_loaders = {"tts": tts_utils.load_model, "stt": stt_utils.load_model}

    if model_category not in model_loaders:
        raise ValueError(f"Model type '{model_category}' not supported")

    return model_loaders[model_category](model_name)
