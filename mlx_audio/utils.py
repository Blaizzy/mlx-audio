"""MLX Audio utilities with lazy imports.

This module provides model loading utilities and re-exports audio processing functions.
TTS and STT imports are lazy to avoid circular dependencies and allow importing
only what's needed.
"""

import importlib.util
from pathlib import Path
from typing import List, Optional, Union

# Re-export audio utilities for backward compatibility
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

__all__ = [
    "hanning",
    "hamming",
    "blackman",
    "bartlett",
    "STR_TO_WINDOW_FN",
    "stft",
    "istft",
    "mel_filters",
    "get_model_category",
    "get_model_name_parts",
    "load_model",
]


def _get_tts_remapping():
    """Lazy import of TTS model remapping."""
    from mlx_audio.tts.utils import MODEL_REMAPPING
    return MODEL_REMAPPING


def _get_stt_remapping():
    """Lazy import of STT model remapping."""
    from mlx_audio.stt.utils import MODEL_REMAPPING
    return MODEL_REMAPPING


def _get_tts_loader():
    """Lazy import of TTS model loader."""
    from mlx_audio.tts.utils import load_model
    return load_model


def _get_stt_loader():
    """Lazy import of STT model loader."""
    from mlx_audio.stt.utils import load_model
    return load_model


def _get_config_loader():
    """Lazy import of config loader."""
    from mlx_audio.tts.utils import load_config
    return load_config


def get_model_category(model_type: str, model_name: List[str]) -> Optional[str]:
    """Determine whether a model belongs to the TTS or STT category."""
    candidates = [model_type] + (model_name or [])

    # Check TTS first
    try:
        tts_remap = _get_tts_remapping()
        for hint in candidates:
            arch = tts_remap.get(hint, hint)
            module_path = f"mlx_audio.tts.models.{arch}"
            if importlib.util.find_spec(module_path) is not None:
                return "tts"
    except ImportError:
        pass

    # Then check STT
    try:
        stt_remap = _get_stt_remapping()
        for hint in candidates:
            arch = stt_remap.get(hint, hint)
            module_path = f"mlx_audio.stt.models.{arch}"
            if importlib.util.find_spec(module_path) is not None:
                return "stt"
    except ImportError:
        pass

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
    load_config = _get_config_loader()
    config = load_config(model_name)
    model_name_parts = get_model_name_parts(model_name)

    # Try to determine model type from config first, then from name
    model_type = config.get("model_type", None)
    model_category = get_model_category(model_type, model_name_parts)

    if not model_category:
        raise ValueError(f"Could not determine model type for {model_name}")

    if model_category == "tts":
        loader = _get_tts_loader()
    elif model_category == "stt":
        loader = _get_stt_loader()
    else:
        raise ValueError(f"Model type '{model_category}' not supported")

    return loader(model_name)
