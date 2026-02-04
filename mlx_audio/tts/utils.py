import glob
import logging
import shutil
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.utils import (
    base_load_model,
    get_model_class,
    get_model_path,
    load_config,
)

MODEL_REMAPPING = {
    "qwen3_tts": "qwen3_tts",
    "outetts": "outetts",
    "spark": "spark",
    "marvis": "sesame",
    "csm": "sesame",
    "voxcpm": "voxcpm",
    "voxcpm1.5": "voxcpm",
    "vibevoice_streaming": "vibevoice",
    "chatterbox_turbo": "chatterbox_turbo",
    "soprano": "soprano",
    "acestep": "ace_step",
    "ace_step": "ace_step",
}
MAX_FILE_SIZE_GB = 5
MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


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
    return get_model_class(
        model_type=model_type,
        model_name=model_name,
        category="tts",
        model_remapping=MODEL_REMAPPING,
    )


def load_model(
    model_path: Path, lazy: bool = False, strict: bool = True, **kwargs
) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    return base_load_model(
        model_path=model_path,
        category="tts",
        model_remapping=MODEL_REMAPPING,
        lazy=lazy,
        strict=strict,
        **kwargs,
    )


def load(
    model_path: Union[str, Path], lazy: bool = False, strict: bool = True, **kwargs
) -> nn.Module:
    """
    Load a text-to-speech model from a local path or HuggingFace repository.

    This is the main entry point for loading TTS models. It automatically
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

    Example:
        >>> from mlx_audio.tts import load
        >>> model = load("mlx-community/outetts-0.3-500M-bf16")
        >>> audio = model.generate("Hello world!")
    """
    return load_model(model_path, lazy=lazy, strict=strict, **kwargs)


def fetch_from_hub(
    model_path: Path, lazy: bool = False, **kwargs
) -> Tuple[nn.Module, dict]:
    model = load_model(model_path, lazy, **kwargs)
    config = load_config(model_path, **kwargs)
    return model, config


def upload_to_hub(path: str, upload_repo: str, hf_path: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    from ..version import __version__

    card = ModelCard.load(hf_path)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.text = dedent(
        f"""
        # {upload_repo}
        This model was converted to MLX format from [`{hf_path}`](https://huggingface.co/{hf_path}) using mlx-audio version **{__version__}**.
        Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
        ## Use with mlx

        ```bash
        pip install -U mlx-audio
        ```

        ```bash
        python -m mlx_audio.tts.generate --model {upload_repo} --text "Describe this image."
        ```
        """
    )
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: str = None,
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    trust_remote_code: bool = True,
    quant_predicate: Optional[str] = None,
):
    from mlx_lm.convert import mixed_quant_predicate_builder
    from mlx_lm.utils import dequantize_model, quantize_model, save_config, save_model

    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config = fetch_from_hub(
        model_path, lazy=True, trust_remote_code=trust_remote_code
    )

    if isinstance(quant_predicate, str):
        quant_predicate = mixed_quant_predicate_builder(quant_predicate, model)

    # Get model-specific quantization predicate if available
    model_quant_predicate = getattr(model, "model_quant_predicate", lambda p, m: True)

    # Define base quantization requirements
    def base_quant_requirements(p, m):
        return (
            hasattr(m, "weight")
            and m.weight.shape[-1] % 64 == 0  # Skip layers not divisible by 64
            and hasattr(m, "to_quantized")
            and model_quant_predicate(p, m)
        )

    # Combine with user-provided predicate if available
    if quant_predicate is None:
        quant_predicate = base_quant_requirements
    else:
        original_predicate = quant_predicate
        quant_predicate = lambda p, m: (
            base_quant_requirements(p, m) and original_predicate(p, m)
        )

    weights = dict(tree_flatten(model.parameters()))

    if dtype is None:
        dtype = config.get("torch_dtype", None)
    if dtype in MODEL_CONVERSION_DTYPES:
        print("[INFO] Using dtype:", dtype)
        dtype = getattr(mx, dtype)
        weights = {k: v.astype(dtype) for k, v in weights.items()}

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(
            model, config, q_group_size, q_bits, quant_predicate=quant_predicate
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    # Ensure the destination directory for MLX model exists before copying files
    mlx_path.mkdir(parents=True, exist_ok=True)

    # Copy Python and JSON files from the model path to the MLX path
    for pattern in [
        "*.py",
        "*.json",
        "*.wav",
        "*.pt",
        "*.safetensors",
        "*.yaml",
        "*.txt",
    ]:
        files = glob.glob(str(model_path / pattern))
        for file in files:
            shutil.copy(file, mlx_path)

        # Check files in subdirectories up to two levels deep
        subdir_files = glob.glob(str(model_path / "**" / pattern), recursive=True)
        for file in subdir_files:
            rel_path = Path(file).relative_to(model_path)
            # Create subdirectories if they don't exist
            dest_dir = mlx_path / rel_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(file, dest_dir)

    save_model(mlx_path, model, donate_model=True)

    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, hf_path)
