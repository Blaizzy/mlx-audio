import glob
import importlib
import logging
import shutil
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.convert import mixed_quant_predicate_builder
from mlx_lm.utils import (
    dequantize_model,
    get_model_path,
    quantize_model,
    save_config,
    save_weights,
)
from mlx_vlm.utils import load_config

MODEL_REMAPPING = {"outetts": "outetts"}
MAX_FILE_SIZE_GB = 5


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
    if model_name is not None:
        for part in model_name:
            if part in MODEL_REMAPPING:
                model_type = MODEL_REMAPPING[part]
                break

    try:
        arch = importlib.import_module(f"mlx_audio.tts.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return arch, model_type


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
    model_name = None
    if isinstance(model_path, str):
        model_name = model_path.lower().split("/")[-1].split("-")
        model_path = get_model_path(model_path)
    elif isinstance(model_path, Path):
        index = model_path.parts.index('hub')
        model_name = model_path.parts[index+1].split("--")[-1].split("-")
    else:
        raise ValueError(f"Invalid model path type: {type(model_path)}")

    config = load_config(model_path, **kwargs)

    # Determine model_type from config or model_name
    model_type = config.get("model_type", None)
    if model_type is None:
        model_type = model_name[0].lower() if model_name is not None else None

    quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        message = f"""
No safetensors found in {model_path}
Create safetensors using the following code:
```
from transformers import AutoModelForCausalLM, AutoProcessor

model_id= "<huggingface_model_id>"
model = AutoModelForCausalLM.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

model.save_pretrained("<local_dir>")
processor.save_pretrained("<local_dir>")
```
Then use the <local_dir> as the --hf-path in the convert script.
```
python -m mlx_vlm.convert --hf-path <local_dir> --mlx-path <mlx_dir>
```
        """
        raise FileNotFoundError(message)

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_type = get_model_and_args(
        model_type=model_type, model_name=model_name
    )

    # Get model config from model class if it exists, otherwise use the config
    model_config = (
        model_class.ModelConfig.from_dict(config)
        if hasattr(model_class, "ModelConfig")
        else config
    )

    model = model_class.Model(model_config)
    quantization = config.get("quantization", None)
    if quantization is None:
        weights = model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:

        def get_class_predicate(p, m):
            # Handle custom per layer quantizations
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            # Handle legacy models which may not have everything quantized
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=get_class_predicate,
        )

    model.load_weights(list(weights.items()), strict=strict)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model


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
    dtype: str = "float16",
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    trust_remote_code: bool = True,
    quant_predicate: Optional[str] = None,
):
    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config = fetch_from_hub(
        model_path, lazy=True, trust_remote_code=trust_remote_code
    )

    if isinstance(quant_predicate, str):
        quant_predicate = mixed_quant_predicate_builder(quant_predicate, model)

    weights = dict(tree_flatten(model.parameters()))
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

    del model
    save_weights(mlx_path, weights, donate_weights=True)

    # Copy Python and JSON files from the model path to the MLX path
    for pattern in ["*.py", "*.json", "*.wav", "*.pt"]:
        files = glob.glob(str(model_path / pattern))
        for file in files:
            shutil.copy(file, mlx_path)

    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, hf_path)
