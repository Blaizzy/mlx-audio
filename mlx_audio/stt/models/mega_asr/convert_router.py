from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _is_conv_kernel(key: str, ndim: int) -> bool:
    return "conv" in key and key.endswith(".weight") and ndim == 3


def convert_router_weights(path: str | Path) -> dict[str, mx.array]:
    raw = mx.load(str(path))
    if not isinstance(raw, dict):
        raise ValueError(f"expected a tensor mapping in {path!r}")
    weights: dict[str, mx.array] = {}
    for key, tensor in raw.items():
        if key.endswith("num_batches_tracked"):
            continue
        if _is_conv_kernel(key, tensor.ndim):
            tensor = mx.transpose(tensor, (0, 2, 1))
        weights[key] = tensor
    return weights
