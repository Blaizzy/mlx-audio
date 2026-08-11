# Copyright © 2023-2024 Apple Inc.
# Vendored from mlx-lm.

import copy
import json
from pathlib import Path
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

MAX_FILE_SIZE_GB = 5


def mixed_quant_predicate_builder(recipe: str, model: nn.Module, group_size: int = 64):
    recipes = {
        "mixed_2_6": (2, 6),
        "mixed_3_4": (3, 4),
        "mixed_3_6": (3, 6),
        "mixed_4_6": (4, 6),
    }
    if recipe not in recipes:
        raise ValueError(f"Invalid quant recipe {recipe}")
    low_bits, high_bits = recipes[recipe]
    down_keys = [name for name, _ in model.named_modules() if "down_proj" in name]
    if not down_keys:
        raise ValueError("Model does not have expected keys for mixed quant.")
    layer_location = next(
        index for index, key in enumerate(down_keys[0].split(".")) if key.isdigit()
    )
    num_layers = len(model.layers)

    def predicate(path: str, module: nn.Module) -> Union[bool, dict]:
        del module
        index = (
            int(path.split(".")[layer_location])
            if len(path.split(".")) > layer_location
            else 0
        )
        high_precision = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )
        wide = (
            "v_proj" in path
            or "v_a_proj" in path
            or "v_b_proj" in path
            or "down_proj" in path
        )
        # lm_head takes high bits regardless of depth, as upstream does.
        bits = high_bits if (wide and high_precision) or "lm_head" in path else low_bits
        return {"group_size": group_size, "bits": bits, "mode": "affine"}

    return predicate


def quantize_model(
    model: nn.Module,
    config: dict,
    group_size: Optional[int],
    bits: Optional[int],
    mode: str = "affine",
    quant_predicate: Optional[Callable] = None,
):
    defaults = {"affine": (64, 4), "mxfp4": (32, 4), "nvfp4": (16, 4), "mxfp8": (32, 8)}
    group_size, bits = group_size or defaults[mode][0], bits or defaults[mode][1]
    config = copy.deepcopy(config)
    quant_predicate = quant_predicate or getattr(model, "quant_predicate", None)
    params = {"group_size": group_size, "bits": bits, "mode": mode}

    # An existing "quantization" key means the model is already partially
    # quantized, so record parameters per layer rather than globally.
    fine_grained = "quantization" in config
    if not fine_grained:
        config["quantization"] = params

    def predicate(path, module):
        if not hasattr(module, "to_quantized") or module.weight.shape[-1] % group_size:
            return False
        result = quant_predicate(path, module) if quant_predicate else True
        if isinstance(result, dict):
            config["quantization"][path] = result
        elif fine_grained and result:
            config["quantization"][path] = params
        return result

    nn.quantize(model, group_size, bits, mode=mode, class_predicate=predicate)
    config["quantization_config"] = config["quantization"]
    return model, config


def dequantize_model(model: nn.Module) -> nn.Module:
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            layer = nn.Linear(*module.weight.shape[::-1], bias="bias" in module)
        elif isinstance(module, nn.QuantizedEmbedding):
            layer = nn.Embedding(*module.weight.shape)
        else:
            continue
        layer.weight = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            module.group_size,
            module.bits,
            module.mode,
        )
        if "bias" in module:
            layer.bias = module.bias
        replacements.append((name, layer))
    if replacements:
        model.update_modules(tree_unflatten(replacements))
    return model


def save_config(config: dict, config_path: Union[str, Path]) -> None:
    config = copy.deepcopy(config)
    config.pop("_name_or_path", None)
    config.pop("vision_config", None)
    if "quantization" in config:
        config["quantization_config"] = config["quantization"]
    with open(config_path, "w") as handle:
        json.dump(dict(sorted(config.items())), handle, indent=4)


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for name, weight in weights.items():
        if shard_size + weight.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[name] = weight
        shard_size += weight.nbytes
    shards.append(shard)
    return shards


def save_model(
    save_path: Union[str, Path], model: nn.Module, *, donate_model: bool = False
) -> None:
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    weights = dict(tree_flatten(model.parameters()))
    total_size = sum(value.nbytes for value in weights.values())
    shards = make_shards(weights)
    name_format = (
        "model-{:05d}-of-{:05d}.safetensors" if len(shards) > 1 else "model.safetensors"
    )
    weight_map = {
        name: name_format.format(index + 1, len(shards))
        for index, shard in enumerate(shards)
        for name in shard
    }

    # Release the model's references before serializing so each shard can be
    # freed as it is written, rather than holding every weight twice.
    if donate_model:
        model.update(tree_map(lambda _: mx.array([]), model.parameters()))
    weights.clear()

    for index, shard in enumerate(shards):
        shards[index] = None
        mx.save_safetensors(
            str(save_path / name_format.format(index + 1, len(shards))),
            shard,
            metadata={"format": "mlx"},
        )
        del shard

    with open(save_path / "model.safetensors.index.json", "w") as handle:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": {k: weight_map[k] for k in sorted(weight_map)},
            },
            handle,
            indent=4,
        )
