import contextlib
from pathlib import Path
from typing import Any, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_reduce

from mlx_audio.utils import (
    base_load_model,
    get_model_name_parts,
    get_model_path,
    load_config,
)

SAMPLE_RATE = 16000


def merge_hotwords(base: Optional[str], hotwords: Optional[List[str]]) -> Optional[str]:
    """Merge a structured vocabulary/hotword list into a model's native prompt.

    ASR backends bias their transcription toward rare words (names, acronyms,
    product terms) through a text field that differs per model — ``system_prompt``
    for Qwen3-ASR, ``initial_prompt`` for Whisper, ``context`` for VibeVoice,
    ``prompt`` for MOSS, etc. This helper lets each model accept a uniform
    ``hotwords`` list and fold it into whichever field it already uses, so callers
    (e.g. the mlx-vlm server) can pass one structured list regardless of backend.

    Returns ``base`` unchanged when ``hotwords`` is empty/None (silent no-op), the
    joined terms when ``base`` is empty, or ``base`` followed by the terms.
    """
    if not hotwords:
        return base
    terms = [str(t).strip() for t in hotwords if t is not None and str(t).strip()]
    if not terms:
        return base
    joined = ", ".join(terms)
    return f"{base}\n{joined}" if base else joined


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        try:
            yield
        finally:
            pass
    else:
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
        )
        max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
        if model_bytes > 0.9 * max_rec_size:
            model_mb = model_bytes // 2**20
            max_rec_mb = max_rec_size // 2**20
            print(
                f"[WARNING] Generating with a model that requires {model_mb} MB "
                f"which is close to the maximum recommended size of {max_rec_mb} "
                "MB. This can be slow. See the documentation for possible work-arounds: "
                "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
            )
        old_limit = mx.set_wired_limit(max_rec_size)
        try:
            yield
        finally:
            if streams is not None:
                for s in streams:
                    mx.synchronize(s)
            else:
                mx.synchronize()
            mx.set_wired_limit(old_limit)


MODEL_REMAPPING = {
    "cohere_asr": "cohere_asr",
    "fireredasr2": "fireredasr2",
    "glm": "glmasr",
    "sensevoice": "sensevoice",
    "voxtral": "voxtral",
    "voxtral_realtime": "voxtral_realtime",
    "vibevoice": "vibevoice_asr",
    "qwen3_asr": "qwen3_asr",
    "phonon": "phonon",
    "moss_transcribe_diarize": "moss_transcribe_diarize",
    "fun_asr_nano": "fun_asr_nano",
    "canary": "canary",
    "moonshine": "moonshine",
    "mms": "mms",
    "granite_speech": "granite_speech",
    "granite_speech5_ctc": "granite_speech5_ctc",
    "granite_speech_nar": "granite_speech_nar",
    "qwen2_audio": "qwen2_audio",
    "mega_asr": "mega_asr",
    "higgs_audio_3": "higgs_audio_3",
    "moss_music": "moss_music",
}


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

    audio, _ = audio_read(file, dtype="float32", sample_rate=sr, nchannels=1)
    return mx.array(audio, dtype=dtype)


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
    source = model_path
    if isinstance(model_path, str):
        kwargs.setdefault("model_name_parts", get_model_name_parts(model_path))
        model_path = get_model_path(
            model_path,
            revision=kwargs.get("revision"),
            force_download=kwargs.get("force_download", False),
            allow_patterns=kwargs.get("allow_patterns"),
        )

    config = load_config(model_path)
    from mlx_audio.stt.models.phonon.transport import (
        is_phonon_model,
        prepare_model_path,
    )

    if is_phonon_model(Path(model_path), config):
        remote_source = (
            source
            if isinstance(source, str) and not Path(source).expanduser().exists()
            else None
        )
        model_path = prepare_model_path(
            Path(model_path),
            source=remote_source,
            revision=kwargs.get("revision"),
            force_download=kwargs.get("force_download", False),
        )
        kwargs["model_type"] = "phonon"

    return base_load_model(
        model_path=model_path,
        category="stt",
        model_remapping=MODEL_REMAPPING,
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
