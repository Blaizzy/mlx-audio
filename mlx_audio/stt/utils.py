from pathlib import Path
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.utils import base_load_model, get_model_path, load_config

SAMPLE_RATE = 16000

MODEL_REMAPPING = {
    "cohere_asr": "cohere_asr",
    "fireredasr2": "fireredasr2",
    "glm": "glmasr",
    "mimo_v2_asr": "mimo_v2_asr",
    "sensevoice": "sensevoice",
    "voxtral": "voxtral",
    "voxtral_realtime": "voxtral_realtime",
    "vibevoice": "vibevoice_asr",
    "qwen3_asr": "qwen3_asr",
    "canary": "canary",
    "moonshine": "moonshine",
    "mms": "mms",
    "granite_speech": "granite_speech",
    "qwen2_audio": "qwen2_audio",
}


def _looks_like_mimo(model_name: list[str] | None, config: dict | None) -> bool:
    if config:
        architectures = config.get("architectures") or []
        if isinstance(architectures, list) and any(
            "MiMoV2ASR" in str(arch) for arch in architectures
        ):
            return True
    for part in model_name or []:
        lowered = part.lower()
        if "mimo" in lowered:
            return True
    return False


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from mlx_audio.utils import resample_audio as _resample_audio

    return _resample_audio(audio, orig_sr, target_sr, axis=0)


def load_audio(
    file: str = Optional[str],
    sr: int = SAMPLE_RATE,
    from_stdin=False,
    dtype: mx.Dtype = mx.float32,
):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    from mlx_audio.audio_io import read as audio_read

    audio, sample_rate = audio_read(file, always_2d=True)
    if sample_rate != sr:
        audio = resample_audio(audio, sample_rate, sr)
    return mx.array(audio, dtype=dtype).mean(axis=1)


def load_model(
    model_path: Union[str, Path],
    lazy: bool = False,
    strict: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    Load and initialize an STT model from a given path.

    Args:
        model_path: The path or HuggingFace repo to load the model from.
        lazy: If False, evaluate model parameters immediately.
        strict: If True, raise an error if any weights are missing.
        **kwargs: Additional keyword arguments (revision, force_download).

    Returns:
        nn.Module: The loaded and initialized model.
    """
    if isinstance(model_path, str):
        model_name = [part.lower() for part in Path(model_path).parts if part]
        resolved_path = get_model_path(
            model_path,
            revision=kwargs.get("revision"),
            force_download=kwargs.get("force_download", False),
        )
    elif isinstance(model_path, Path):
        model_name = [part.lower() for part in model_path.parts if part]
        resolved_path = model_path
    else:
        raise ValueError(f"Invalid model path type: {type(model_path)}")

    config = load_config(resolved_path)
    if _looks_like_mimo(model_name, config):
        from mlx_audio.stt.models.mimo_v2_asr import MiMoASR

        audio_tokenizer_dir = kwargs.pop("audio_tokenizer_dir", None)
        return MiMoASR.from_pretrained(
            resolved_path,
            audio_tokenizer_dir=audio_tokenizer_dir,
            revision=kwargs.get("revision"),
            force_download=kwargs.get("force_download", False),
        )

    return base_load_model(
        model_path=resolved_path,
        category="stt",
        model_remapping=MODEL_REMAPPING,
        model_name_parts=model_name,
        lazy=lazy,
        strict=strict,
        **kwargs,
    )


def load(
    model_path: Union[str, Path],
    lazy: bool = False,
    strict: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    Load a speech-to-text model from a local path or HuggingFace repository.

    This is the main entry point for loading STT models. It automatically
    detects the model type and initializes the appropriate model class.

    Args:
        model_path: The local path or HuggingFace repo ID to load from.
        lazy: If False, evaluate model parameters immediately.
        strict: If True, raise an error if any weights are missing.
        **kwargs: Additional keyword arguments such as `revision` and
            `force_download`.

    Returns:
        nn.Module: The loaded and initialized model.

    """
    return load_model(model_path, lazy=lazy, strict=strict, **kwargs)
