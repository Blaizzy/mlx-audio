import copy
import glob
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import get_model_path, load_config, make_shards

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


def get_class_predicate(weights=None):
    if weights:
        return lambda p, m: (
            hasattr(m, "to_quantized")
            and m.weight.shape[-1] % 64 == 0
            and f"{p}.scales" in weights
        )
    else:
        return lambda _, m: hasattr(m, "to_quantized") and m.weight.shape[-1] % 64 == 0


def quantize_model(
    model: nn.Module,
    config: dict,
    q_group_size: int,
    q_bits: int,
) -> Tuple[dict, dict]:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.
        skip_vision (bool): Whether to skip quantizing vision model weights.

    Returns:
        Tuple[dict, dict]: Tuple containing quantized weights and updated config.
    """
    quantized_config = copy.deepcopy(config)

    # Quantize only layers with to_quantized method and divisible by 64
    nn.quantize(
        model,
        q_group_size,
        q_bits,
        class_predicate=get_class_predicate(),
    )

    # Update config and get weights
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def load_model(model_path: Path, lazy: bool = False, **kwargs) -> nn.Module:
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

    config = load_config(model_path, **kwargs)

    model_type = config.get("model_type", model_name[0])

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

    if quantization is not None:
        # Handle legacy models which may not have everything quantized`
        class_predicate = get_class_predicate(weights)

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model
