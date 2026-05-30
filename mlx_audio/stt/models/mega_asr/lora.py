from __future__ import annotations

from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn

from .convert_lora import LoraModule

_LORA_ACTIVE_ATTR = "_lora_active"


def materialize_delta(module: LoraModule) -> mx.array:
    a = module["A"].astype(mx.float32)
    b = module["B"].astype(mx.float32)
    scaling = float(module["scaling"])
    delta = scaling * (b @ a)
    assert delta.shape == (b.shape[0], a.shape[1])
    return delta


def resolve_linear(model: nn.Module, path: str) -> nn.Linear:
    node: object = model
    for segment in path.split("."):
        if segment.isdigit():
            if not isinstance(node, (list, tuple)):
                raise TypeError(
                    f"path {path!r}: segment {segment!r} indexes a non-sequence "
                    + type(node).__name__
                )
            node = node[int(segment)]
        else:
            node = getattr(node, segment)
    if not isinstance(node, nn.Linear):
        raise TypeError(
            f"path {path!r} resolves to {type(node).__name__}, expected nn.Linear"
        )
    return node


def _accumulate(leaf: nn.Linear, module: LoraModule, sign: float) -> None:
    weight = leaf.weight
    delta = materialize_delta(module)
    if weight.shape != delta.shape:
        raise ValueError(
            f"delta shape {tuple(delta.shape)} does not match "
            + f"weight shape {tuple(weight.shape)}"
        )
    delta = delta.astype(weight.dtype)
    leaf.weight = weight + delta if sign > 0 else weight - delta
    mx.eval(leaf.weight)


def apply_deltas(model: nn.Module, adapter: Mapping[str, LoraModule]) -> None:
    if getattr(model, _LORA_ACTIVE_ATTR, False):
        raise RuntimeError("LoRA deltas already applied; remove before re-applying")
    try:
        for path, module in adapter.items():
            _accumulate(resolve_linear(model, path), module, 1.0)
        setattr(model, _LORA_ACTIVE_ATTR, True)
    finally:
        mx.clear_cache()


def remove_deltas(model: nn.Module, adapter: Mapping[str, LoraModule]) -> None:
    if not getattr(model, _LORA_ACTIVE_ATTR, False):
        raise RuntimeError("LoRA deltas are not applied; nothing to remove")
    try:
        for path, module in adapter.items():
            _accumulate(resolve_linear(model, path), module, -1.0)
        setattr(model, _LORA_ACTIVE_ATTR, False)
    finally:
        mx.clear_cache()
