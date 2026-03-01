"""Utility functions for mlx_audio.

This module provides a unified interface for loading TTS and STT models,
with lazy imports to avoid loading unnecessary dependencies.
"""

import dataclasses
import glob
import importlib
import importlib.util
import json
import logging
from pathlib import Path
from typing import (
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_origin,
    get_type_hints,
)

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

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

T = TypeVar("T")


def from_dict(data_class: Type[T], data: dict) -> T:

    if not dataclasses.is_dataclass(data_class):
        raise TypeError(f"{data_class} is not a dataclass")

    field_types = get_type_hints(data_class)
    kwargs = {}

    for field in dataclasses.fields(data_class):
        field_name = field.name
        if field_name not in data:
            continue

        value = data[field_name]
        field_type = field_types[field_name]

        # Handle Optional types
        origin = get_origin(field_type)
        if origin is Union:
            # For Optional[X], get the non-None type
            args = [a for a in field_type.__args__ if a is not type(None)]
            if args:
                field_type = args[0]

        # Recursively convert nested dataclasses
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            value = from_dict(field_type, value)

        kwargs[field_name] = value

    return data_class(**kwargs)


# =============================================================================
# Shared Model Loading Utilities
# =============================================================================

# Default file patterns to download from HuggingFace
DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.py",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jsonl",
    "*.yaml",
    "*.wav",
    "*.pth",
    "*.npz",
]


def _is_local_path(path: str) -> bool:
    """Check if the path looks like a local filesystem path."""
    return (
        path.startswith(".")
        or path.startswith("/")
        or path.startswith("~")
        or (len(path) > 1 and path[1] == ":")  # Windows drive letter
    )


def get_model_path(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
    force_download: bool = False,
    allow_patterns: Optional[List[str]] = None,
) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo: The local path or Hugging Face repository ID of the model.
        revision: A revision id which can be a branch name, a tag, or a commit hash.
        force_download: Force re-download even if cached.
        allow_patterns: File patterns to download. Defaults to common model files.

    Returns:
        Path: The path to the model.

    Raises:
        FileNotFoundError: If a local path is provided but doesn't exist.
    """
    model_path = Path(path_or_hf_repo)

    if model_path.exists():
        return model_path

    # If it looks like a local path but doesn't exist, raise an error
    if _is_local_path(path_or_hf_repo):
        raise FileNotFoundError(f"Local path not found: {path_or_hf_repo}")

    # Use default patterns if none provided
    if allow_patterns is None:
        allow_patterns = DEFAULT_ALLOW_PATTERNS

    model_path = Path(
        snapshot_download(
            path_or_hf_repo,
            revision=revision,
            allow_patterns=allow_patterns,
            force_download=force_download,
        )
    )

    return model_path


def load_config(model_path: Union[str, Path], **kwargs) -> dict:
    """Load model configuration from a path or Hugging Face repo.

    Args:
        model_path: Local path or Hugging Face repo ID to load config from
        **kwargs: Additional keyword arguments (revision, force_download)

    Returns:
        dict: Model configuration

    Raises:
        FileNotFoundError: If config.json is not found at the path
    """
    if isinstance(model_path, str):
        model_path = get_model_path(model_path, **kwargs)

    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Config not found at {model_path}")


def load_weights(model_path: Path) -> dict:
    """Load model weights from safetensors or npz files.

    Args:
        model_path: Path to the model directory

    Returns:
        dict: Dictionary of weight name -> array

    Raises:
        FileNotFoundError: If no weight files found
    """
    # Try safetensors first, then npz
    weight_files = glob.glob(str(model_path / "*.safetensors"))

    if not weight_files:
        weight_files = glob.glob(str(model_path / "*.npz"))

    if not weight_files:
        raise FileNotFoundError(
            f"No weight files (safetensors or npz) found in {model_path}"
        )

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    return weights


def apply_quantization(
    model: nn.Module,
    config: dict,
    weights: dict,
    model_quant_predicate: Optional[callable] = None,
) -> None:
    """Apply quantization to a model if specified in config.

    Args:
        model: The model to quantize
        config: Model config dict (should contain 'quantization' key if quantized)
        weights: Loaded weights dict (used to check which layers have scales)
        model_quant_predicate: Optional model-specific predicate for which layers to quantize
    """
    quantization = config.get("quantization", None)
    if quantization is None:
        return

    def get_class_predicate(p, m):
        # Skip layers without quantization capability
        if not hasattr(m, "to_quantized"):
            return False
        # Skip layers not divisible by 64
        if hasattr(m, "weight") and m.weight.size % 64 != 0:
            return False
        # Use model-specific predicate if available
        if model_quant_predicate is not None:
            pred_result = model_quant_predicate(p, m)
            if isinstance(pred_result, dict):
                return pred_result
            if not pred_result:
                return False
        # Handle custom per layer quantizations
        if p in config["quantization"]:
            return config["quantization"][p]
        # Handle legacy models which may not have everything quantized
        return f"{p}.scales" in weights

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization.get("mode", "affine"),
        class_predicate=get_class_predicate,
    )


def get_model_class(
    model_type: str,
    model_name: List[str],
    category: str,
    model_remapping: dict,
) -> Tuple:
    """
    Retrieve the model architecture module based on the model type and name.

    Args:
        model_type: The type of model to load (e.g., "whisper", "voxtral").
        model_name: List of model name components for remapping hints.
        category: Either "tts" or "stt".
        model_remapping: Dictionary mapping model names to architecture names.

    Returns:
        Tuple[module, str]: The imported architecture module and resolved model_type.

    Raises:
        ValueError: If the model type is not supported.
    """
    # Stage 1: Check if the model type is in the remapping
    model_type_mapped = model_remapping.get(model_type, None)

    # Stage 2: Check for partial matches in segments of the model name
    # Only do this if the initial mapping didn't find a match
    models_dir = Path(__file__).parent / category / "models"
    available_models = []
    if models_dir.exists() and models_dir.is_dir():
        for item in models_dir.iterdir():
            if item.is_dir() and not item.name.startswith("__"):
                available_models.append(item.name)

    if model_name is not None and model_type_mapped != model_type:
        for part in model_name:
            if part in available_models:
                model_type = part
            if part in model_remapping:
                model_type = model_remapping[part]
                break
    elif model_type_mapped is not None:
        model_type = model_type_mapped

    try:
        module_path = f"mlx_audio.{category}.models.{model_type}"
        arch = importlib.import_module(module_path)
    except ImportError as e:
        if e.name != module_path:
            print("\n", flush=True)

            raise ImportError(
                f"\nMissing dependency while loading {model_type}: {e}\n"
                f"Please install it using: pip install {e.name}"
            ) from e

        msg = f"Model type {model_type} not supported for {category}."
        logging.error(msg)
        raise ValueError(msg)

    return arch, model_type


def base_load_model(
    model_path: Union[str, Path],
    category: str,
    model_remapping: dict,
    lazy: bool = False,
    strict: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Base implementation for loading models (shared between TTS and STT).

    Args:
        model_path: The path or HuggingFace repo to load the model from.
        category: Either "tts" or "stt".
        model_remapping: Dictionary mapping model names to architecture names.
        lazy: If False, evaluate model parameters immediately.
        strict: If True, raise an error if any weights are missing.
        **kwargs: Additional keyword arguments (revision, force_download).

    Returns:
        nn.Module: The loaded and initialized model.
    """
    model_name = None
    model_type = None

    if isinstance(model_path, str):
        model_name = model_path.lower().split("/")[-1].split("-")
        revision = kwargs.get("revision", None)
        force_download = kwargs.get("force_download", False)
        model_path = get_model_path(
            model_path, revision=revision, force_download=force_download
        )
    elif isinstance(model_path, Path):
        try:
            index = model_path.parts.index("hub")
            model_name = model_path.parts[index + 1].lower().split("--")[-1].split("-")
        except ValueError:
            model_name = model_path.name.lower().split("-")
    else:
        raise ValueError(f"Invalid model path type: {type(model_path)}")

    config = load_config(model_path)
    config["model_path"] = str(model_path)

    # Determine model_type from config or model_name
    model_type = config.get("model_type", None)
    if model_type is None:
        model_type = config.get("architecture", None)
    if model_type is None:
        model_type = model_name[0].lower() if model_name is not None else None

    model_class, model_type = get_model_class(
        model_type=model_type,
        model_name=model_name,
        category=category,
        model_remapping=model_remapping,
    )

    # Get model config from model class if it exists, otherwise use the config
    model_config = (
        model_class.ModelConfig.from_dict(config)
        if hasattr(model_class, "ModelConfig")
        else config
    )
    model = model_class.Model(model_config)

    # Load weights
    weights = load_weights(model_path)

    # Sanitize weights if the model has a sanitize method
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Apply quantization if specified
    model_quant_predicate = getattr(model, "model_quant_predicate", None)
    apply_quantization(model, config, weights, model_quant_predicate)

    model.load_weights(list(weights.items()), strict=strict)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()

    # Call post-load hook if the model defines one
    if hasattr(model_class.Model, "post_load_hook"):
        model = model_class.Model.post_load_hook(model, model_path)

    return model


# Lazy-loaded modules
_stt_utils = None
_tts_utils = None
_vad_utils = None
_lid_utils = None


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


def _get_vad_utils():
    """Lazy load VAD utils."""
    global _vad_utils
    if _vad_utils is None:
        from mlx_audio.vad import utils as vad_utils

        _vad_utils = vad_utils
    return _vad_utils


def _get_lid_utils():
    """Lazy load LID utils."""
    global _lid_utils
    if _lid_utils is None:
        from mlx_audio.lid import utils as lid_utils

        _lid_utils = lid_utils
    return _lid_utils


def audio_volume_normalize(audio, coeff: float = 0.2):
    """Normalize the volume of an audio signal.

    Args:
        audio: Input audio signal array (numpy array).
        coeff: Target coefficient for normalization, default is 0.2.

    Returns:
        numpy array: The volume-normalized audio signal.
    """
    import numpy as np

    # Sort the absolute values of the audio signal
    temp = np.sort(np.abs(audio))

    # If the maximum value is less than 0.1, scale the array to have a maximum of 0.1
    if temp[-1] < 0.1:
        scaling_factor = max(temp[-1], 1e-3)
        audio = audio / scaling_factor * 0.1

    # Filter out values less than 0.01 from temp
    temp = temp[temp > 0.01]
    L = temp.shape[0]

    # If there are fewer than or equal to 10 significant values, return as-is
    if L <= 10:
        return audio

    # Compute the average of the top 10% to 1% of values in temp
    volume = np.mean(temp[int(0.9 * L) : int(0.99 * L)])

    # Normalize the audio to the target coefficient level
    audio = audio * np.clip(coeff / volume, a_min=0.1, a_max=10)

    # Ensure the maximum absolute value does not exceed 1
    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio / max_value

    return audio


def random_select_audio_segment(audio, length: int):
    """Get a random audio segment of given length.

    Args:
        audio: Input audio array (numpy array).
        length: Desired segment length (sample_rate * duration).

    Returns:
        numpy array: Audio segment of specified length.
    """
    import random

    import numpy as np

    if audio.shape[0] < length:
        audio = np.pad(audio, (0, int(length - audio.shape[0])))
    start_index = random.randint(0, audio.shape[0] - length)
    end_index = int(start_index + length)

    return audio[start_index:end_index]


def load_audio(
    audio: Union[str, mx.array],
    sample_rate: int = 24000,
    length: Optional[int] = None,
    volume_normalize: bool = False,
    segment_duration: Optional[int] = None,
) -> mx.array:
    """Load audio from file path or return mx.array as-is.

    Args:
        audio: Audio input - can be:
            - str: Path to audio file (will be loaded and resampled to sample_rate)
            - mx.array: MLX array (returned as-is)
        sample_rate: Target sample rate (default 24000)
        length: Target length in samples (pad or truncate if specified)
        volume_normalize: Whether to normalize audio volume
        segment_duration: If specified, randomly select a segment of this duration (seconds)

    Returns:
        mx.array: Audio waveform at target sample rate

    Raises:
        FileNotFoundError: If audio file path does not exist
        TypeError: If audio is not str or mx.array
    """
    if isinstance(audio, mx.array):
        return audio

    if not isinstance(audio, str):
        raise TypeError(f"audio must be str or mx.array, got {type(audio)}")

    import os

    import numpy as np
    from scipy.signal import resample

    from mlx_audio.audio_io import read as audio_read

    if not os.path.exists(audio):
        raise FileNotFoundError(f"Audio file not found: {audio}")

    samples, orig_sample_rate = audio_read(audio)
    shape = samples.shape

    # Collapse multi channel as mono
    if len(shape) > 1:
        samples = samples.sum(axis=1)
        samples = samples / shape[1]

    # Resample if needed
    if sample_rate != orig_sample_rate:
        duration = samples.shape[0] / orig_sample_rate
        num_samples = int(duration * sample_rate)
        samples = resample(samples, num_samples)

    # Random segment selection
    if segment_duration is not None:
        seg_length = int(sample_rate * segment_duration)
        samples = random_select_audio_segment(samples, seg_length)

    # Volume normalization
    if volume_normalize:
        samples = audio_volume_normalize(samples)

    # Length adjustment
    if length is not None:
        if samples.shape[0] > length:
            samples = samples[:length]
        else:
            samples = np.pad(samples, (0, int(length - samples.shape[0])))

    return mx.array(samples, dtype=mx.float32)


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
    # Audio utilities
    "load_audio",
    "audio_volume_normalize",
    "random_select_audio_segment",
    # Model utilities
    "from_dict",
    "is_valid_module_name",
    "get_model_category",
    "get_model_name_parts",
    "load_model",
    # Shared loading utilities
    "get_model_path",
    "load_config",
    "load_weights",
    "apply_quantization",
    "get_model_class",
    "base_load_model",
]


def is_valid_module_name(name: str) -> bool:
    """Check if a string is a valid Python module name."""
    if not name or not isinstance(name, str):
        return False

    return name[0].isalpha() or name[0] == "_"


def get_model_category(model_type: str, model_name: List[str]) -> Optional[str]:
    """Determine whether a model belongs to the TTS, STT, LID, or VAD category."""
    stt_utils = _get_stt_utils()
    tts_utils = _get_tts_utils()
    vad_utils = _get_vad_utils()
    lid_utils = _get_lid_utils()

    candidates = [model_type] + (model_name or [])

    categories = [
        ("tts", tts_utils.MODEL_REMAPPING),
        ("stt", stt_utils.MODEL_REMAPPING),
        ("lid", lid_utils.MODEL_REMAPPING),
        ("vad", vad_utils.MODEL_REMAPPING),
    ]

    # First pass: check for explicit remapping matches (higher priority)
    for category, remap in categories:
        for hint in candidates:
            if hint in remap:
                arch = remap[hint]
                if not is_valid_module_name(arch):
                    continue
                module_path = f"mlx_audio.{category}.models.{arch}"
                if importlib.util.find_spec(module_path) is not None:
                    return category

    # Second pass: check for direct module matches (fallback)
    for category, remap in categories:
        for hint in candidates:
            if hint not in remap and is_valid_module_name(hint):
                module_path = f"mlx_audio.{category}.models.{hint}"
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
    """Load a TTS, STT, LID, or VAD model based on its configuration and name.

    Args:
        model_name (str): Name or path of the model to load

    Returns:
        The loaded model instance

    Raises:
        ValueError: If the model type cannot be determined or is not supported
    """
    tts_utils = _get_tts_utils()
    stt_utils = _get_stt_utils()
    vad_utils = _get_vad_utils()
    lid_utils = _get_lid_utils()

    config = tts_utils.load_config(model_name)
    model_name_parts = get_model_name_parts(model_name)

    # Try to determine model type from config first, then from name
    model_type = config.get("model_type", None)
    model_category = get_model_category(model_type, model_name_parts)

    if not model_category:
        raise ValueError(f"Could not determine model type for {model_name}")

    model_loaders = {
        "tts": tts_utils.load_model,
        "stt": stt_utils.load_model,
        "lid": lid_utils.load_model,
        "vad": vad_utils.load_model,
    }

    if model_category not in model_loaders:
        raise ValueError(f"Model type '{model_category}' not supported")

    return model_loaders[model_category](model_name)
