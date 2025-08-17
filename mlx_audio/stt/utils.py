import importlib
import logging
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from scipy import signal

SAMPLE_RATE = 16000

MODEL_REMAPPING = {}
MAX_FILE_SIZE_GB = 5
MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down, padtype="edge")
    return resampled


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
    audio, sample_rate = sf.read(file, always_2d=True)
    if sample_rate != sr:
        audio = resample_audio(audio, sample_rate, sr)
    return mx.array(audio, dtype=dtype).mean(axis=1)


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "*.model",
                    "*.tiktoken",
                    "*.txt",
                    "*.jsonl",
                    "*.yaml",
                ],
            )
        )

    return model_path


# Get a list of all available model types from the models directory
def get_available_models():
    """
    Get a list of all available TTS model types by scanning the models directory.

    Returns:
        List[str]: A list of available model type names
    """
    models_dir = Path(__file__).parent / "models"
    available_models = []

    if models_dir.exists() and models_dir.is_dir():
        for item in models_dir.iterdir():
            if item.is_dir() and not item.name.startswith("__"):
                available_models.append(item.name)

    return available_models


def get_model_and_args(model_type: str, model_name: List[str]):
    """
    Retrieve the model architecture module based on the model type and name.

    This function attempts to find the appropriate model architecture by:
    1. Checking if the model_type is directly in the MODEL_REMAPPING dictionary
    2. Looking for partial matches in segments of the model_name

    Args:
        model_type (str): The type of model to load (e.g., "outetts").
        model_name (List[str]): List of model name components that might contain
                               remapping information.

    Returns:
        Tuple[module, str]: A tuple containing:
            - The imported architecture module
            - The resolved model_type string after remapping

    Raises:
        ValueError: If the model type is not supported (module import fails).
    """
    # Stage 1: Check if the model type is in the remapping
    model_type = MODEL_REMAPPING.get(model_type, model_type)

    # Stage 2: Check for partial matches in segments of the model name
    models = get_available_models()
    if model_name is not None:
        for part in model_name:
            # First check if the part matches an available model directory name
            if part in models:
                model_type = part

            # Then check if the part is in our custom remapping dictionary
            if part in MODEL_REMAPPING:
                model_type = MODEL_REMAPPING[part]
                break

    try:
        arch = importlib.import_module(f"mlx_audio.stt.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return arch, model_type


def load_model(model_path: str, lazy: bool = False, strict: bool = True, **kwargs):
    """
    Load and initialize the model from a given path.

    Args:
        model_path (str): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    model_name = None
    model_type = None
    if isinstance(model_path, str):
        model_name = model_path.lower().split("/")[-1].split("-")
    elif isinstance(model_path, Path):
        index = model_path.parts.index("hub")
        model_name = model_path.parts[index + 1].lower().split("--")[-1].split("-")
    else:
        raise ValueError(f"Invalid model path type: {type(model_path)}")

    model_class, model_type = get_model_and_args(
        model_type=model_type, model_name=model_name
    )
    model = model_class.Model.from_pretrained(model_path)

    if not lazy:
        model.eval()

    return model
