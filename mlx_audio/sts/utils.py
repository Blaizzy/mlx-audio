from pathlib import Path
from typing import Optional, Union

import mlx.nn as nn

from mlx_audio.utils import (
    base_load_model,
    get_model_name_parts,
    get_model_path,
    load_config,
)

MODEL_REMAPPING = {
    "deepfilter": "deepfilternet",
    "deepfilternet": "deepfilternet",
    "deepfilternet3": "deepfilternet",
    "lfm_audio": "lfm_audio",
    "lfm2_audio": "lfm_audio",
    "lfm2.5": "lfm_audio",
    "moshi": "moshi",
    "moshiko": "moshi",
    "mossformer2": "mossformer2_se",
    "mossformer2_se": "mossformer2_se",
    "sam_audio": "sam_audio",
    "samaudio": "sam_audio",
}


def infer_model_type_from_config(config: dict) -> Optional[str]:
    if not config:
        return None

    model_type = config.get("model_type", None)
    if model_type is None:
        model_type = config.get("architecture", None)

    if model_type is not None:
        return MODEL_REMAPPING.get(model_type, model_type)

    if str(config.get("model_version", "")).startswith("DeepFilterNet"):
        return "deepfilternet"

    if {
        "audio_codec",
        "text_encoder",
        "transformer",
    }.issubset(config):
        return "sam_audio"

    if {
        "win_len",
        "win_inc",
        "num_mels",
        "out_channels_final",
    }.issubset(config):
        return "mossformer2_se"

    return None


def _resolve_model_type(model_path: Union[str, Path], **kwargs) -> Optional[str]:
    try:
        config = load_config(model_path, **kwargs)
    except FileNotFoundError:
        config = {}

    model_type = infer_model_type_from_config(config)
    if model_type is not None:
        return MODEL_REMAPPING.get(model_type, model_type)

    for hint in get_model_name_parts(model_path):
        if hint in MODEL_REMAPPING:
            return MODEL_REMAPPING[hint]
        if hint in MODEL_REMAPPING.values():
            return hint

    return None


def _resolve_deepfilternet_model_path(
    model_path: Union[str, Path],
    **kwargs,
) -> Path:
    model_dir = (
        model_path.expanduser()
        if isinstance(model_path, Path)
        else get_model_path(str(model_path), **kwargs)
    )

    if (model_dir / "config.json").exists():
        return model_dir

    default_subfolder = model_dir / "v3"
    if (default_subfolder / "config.json").exists():
        return default_subfolder

    raise FileNotFoundError(
        f"Could not find DeepFilterNet weights in {model_dir} or {default_subfolder}"
    )


def _infer_moshi_quantization(model_path: Union[str, Path]) -> Optional[int]:
    for hint in get_model_name_parts(model_path):
        if hint in {"q4", "4bit"}:
            return 4
        if hint in {"q8", "8bit"}:
            return 8

    return None


def load_model(
    model_path: Union[str, Path],
    lazy: bool = False,
    strict: bool = False,
    **kwargs,
) -> nn.Module:
    """Load an STS model from a local path or HuggingFace repository.

    This loader supports STS models that can be expressed through the shared
    MLX Audio loading contract with sensible defaults. Model-specific
    ``from_pretrained(...)`` paths remain available for custom behavior.
    """
    model_type = _resolve_model_type(model_path, **kwargs)
    if model_type == "moshi":
        from mlx_audio.sts.models.moshi import MoshiSTSModel

        return MoshiSTSModel.from_pretrained(
            str(model_path),
            quantized=_infer_moshi_quantization(model_path),
            revision=kwargs.get("revision"),
            force_download=kwargs.get("force_download", False),
        )

    if model_type in {
        "lfm_audio",
        "mossformer2_se",
        "deepfilternet",
        "sam_audio",
    }:
        resolved_model_path = model_path
        if model_type == "deepfilternet":
            resolved_model_path = _resolve_deepfilternet_model_path(
                model_path, **kwargs
            )

        load_kwargs = {
            "model_path": resolved_model_path,
            "category": "sts",
            "model_remapping": MODEL_REMAPPING,
            "model_type": model_type,
            "model_name_parts": get_model_name_parts(model_path),
            "lazy": lazy,
            "strict": strict,
            **kwargs,
        }
        return base_load_model(**load_kwargs)

    raise ValueError(
        f"STS model type '{model_type}' is not supported by the generic STS "
        "loader yet. Use the model-specific from_pretrained(...) initializer "
        "for custom STS model families."
    )


def load(
    model_path: Union[str, Path],
    lazy: bool = False,
    strict: bool = False,
    **kwargs,
) -> nn.Module:
    """Alias for :func:`load_model`."""
    return load_model(model_path, lazy=lazy, strict=strict, **kwargs)
