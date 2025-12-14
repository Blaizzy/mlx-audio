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
from mlx_audio.stt.utils import MODEL_REMAPPING as MODEL_STT_REMAPPING
from mlx_audio.stt.utils import load_model as load_stt_model
from mlx_audio.tts.utils import MODEL_REMAPPING as MODEL_TTS_REMAPPING
from mlx_audio.tts.utils import load_config
from mlx_audio.tts.utils import load_model as load_tts_model

__all__ = [
    "hanning",
    "hamming",
    "blackman",
    "bartlett",
    "STR_TO_WINDOW_FN",
    "stft",
    "istft",
    "mel_filters",
    "load_model",
]


def is_valid_module_name(name: str) -> bool:
    """Check if a string is a valid Python module name."""
    if not name or not isinstance(name, str):
        return False

    return name[0].isalpha() or name[0] == "_"


def get_model_category(model_type: str, model_name: List[str]) -> Optional[str]:
    """Determine whether a model belongs to the TTS or STT category."""

    candidates = [model_type] + (model_name or [])

    for category, remap in (
        ("tts", MODEL_TTS_REMAPPING),
        ("stt", MODEL_STT_REMAPPING),
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
    config = load_config(model_name)
    model_name_parts = get_model_name_parts(model_name)

    # Try to determine model type from config first, then from name
    model_type = config.get("model_type", None)
    model_category = get_model_category(model_type, model_name_parts)

    if not model_category:
        raise ValueError(f"Could not determine model type for {model_name}")

    model_loaders = {"tts": load_tts_model, "stt": load_stt_model}

    if model_category not in model_loaders:
        raise ValueError(f"Model type '{model_category}' not supported")

    return model_loaders[model_category](model_name)
