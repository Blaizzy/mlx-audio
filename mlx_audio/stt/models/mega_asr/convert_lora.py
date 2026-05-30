from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import mlx.core as mx

THINKER_PREFIX = "base_model.model.thinker."
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"
LORA_A_FACTOR_SUFFIX = ".lora_A"
LORA_B_FACTOR_SUFFIX = ".lora_B"


class LoraModule(TypedDict):
    A: mx.array
    B: mx.array
    scaling: float


def _module_name(tensor_key: str) -> str:
    name = tensor_key
    for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.startswith(THINKER_PREFIX):
        name = name[len(THINKER_PREFIX) :]
    return name


def _pattern_lookup(module: str, pattern: dict[str, int], default: int) -> int:
    for candidate in (module, "thinker." + module):
        if candidate in pattern:
            return pattern[candidate]
    for key, value in pattern.items():
        if module == key or module.endswith("." + key):
            return value
    return default


def load_lora_adapter(directory: str | Path) -> dict[str, LoraModule]:
    directory = Path(directory)
    config = cast(
        "dict[str, object]",
        json.loads((directory / "adapter_config.json").read_text()),
    )
    global_r = int(cast(int, config.get("r", 1)))
    global_alpha = int(cast(int, config.get("lora_alpha", global_r)))
    rank_pattern = cast("dict[str, int]", config.get("rank_pattern") or {})
    alpha_pattern = cast("dict[str, int]", config.get("alpha_pattern") or {})

    raw = mx.load(str(directory / "adapter_model.safetensors"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected a tensor mapping in {directory!r}")

    a_tensors: dict[str, mx.array] = {}
    b_tensors: dict[str, mx.array] = {}
    for key, tensor in raw.items():
        if key.endswith(LORA_A_SUFFIX):
            a_tensors[_module_name(key)] = tensor.astype(mx.float32)
        elif key.endswith(LORA_B_SUFFIX):
            b_tensors[_module_name(key)] = tensor.astype(mx.float32)

    adapter: dict[str, LoraModule] = {}
    for module, a in a_tensors.items():
        rank = _pattern_lookup(module, rank_pattern, global_r)
        alpha = _pattern_lookup(module, alpha_pattern, global_alpha)
        adapter[module] = {
            "A": a,
            "B": b_tensors[module],
            "scaling": float(alpha) / float(rank),
        }
    return adapter


def load_lora_factors(path: str | Path) -> dict[str, LoraModule]:
    raw = mx.load(str(path))
    if not isinstance(raw, dict):
        raise ValueError(f"expected a tensor mapping in {path!r}")

    a_tensors: dict[str, mx.array] = {}
    b_tensors: dict[str, mx.array] = {}
    for key, tensor in raw.items():
        if key.endswith(LORA_A_FACTOR_SUFFIX):
            a_tensors[key[: -len(LORA_A_FACTOR_SUFFIX)]] = tensor.astype(mx.float32)
        elif key.endswith(LORA_B_FACTOR_SUFFIX):
            b_tensors[key[: -len(LORA_B_FACTOR_SUFFIX)]] = tensor.astype(mx.float32)

    adapter: dict[str, LoraModule] = {}
    for module, a in a_tensors.items():
        adapter[module] = {"A": a, "B": b_tensors[module], "scaling": 1.0}
    return adapter
