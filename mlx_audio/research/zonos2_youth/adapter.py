from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class LoRASpec:
    name: str
    base_weight_shape: tuple[int, ...]
    rank: int = 8
    alpha: float = 16.0
    target: str = "linear"
    value_slice: tuple[int, int] | None = None
    dtype: str = "float32"

    @property
    def scaling(self) -> float:
        return float(self.alpha) / max(1, int(self.rank))


@dataclass
class LoRAWeights:
    spec: LoRASpec
    a: mx.array
    b: mx.array

    @classmethod
    def exact_zero(cls, spec: LoRASpec) -> "LoRAWeights":
        if len(spec.base_weight_shape) == 2:
            out_features, in_features = spec.base_weight_shape
        elif len(spec.base_weight_shape) == 3 and spec.value_slice is not None:
            _, out_per_chunk, in_features = spec.base_weight_shape
            start, end = spec.value_slice
            out_features = int(end) - int(start)
            if out_features <= 0 or out_features > out_per_chunk:
                raise ValueError("invalid value_slice for chunked LoRA")
        else:
            raise ValueError("LoRA supports 2-D linear weights or value-sliced chunked weights")
        dtype = getattr(mx, spec.dtype)
        a = mx.zeros((spec.rank, in_features), dtype=dtype)
        b = mx.zeros((out_features, spec.rank), dtype=dtype)
        return cls(spec=spec, a=a, b=b)

    def delta(self) -> mx.array:
        return (self.b @ self.a) * self.spec.scaling

    def apply_to_weight(self, weight: mx.array, *, strength: float = 1.0) -> mx.array:
        delta = self.delta().astype(weight.dtype) * float(strength)
        if len(weight.shape) == 2:
            return weight + delta
        if len(weight.shape) == 3 and self.spec.value_slice is not None:
            start, end = self.spec.value_slice
            updated = mx.array(weight)
            value = updated[1]
            merged_value = mx.concatenate(
                [value[:start], value[start:end] + delta, value[end:]],
                axis=0,
            )
            return mx.stack([updated[0], merged_value], axis=0)
        raise ValueError("unsupported weight shape")

    def contribution(self, x: mx.array, *, strength: float = 1.0) -> mx.array:
        return ((x @ self.a.T) @ self.b.T) * self.spec.scaling * float(strength)


@dataclass
class AdapterManifest:
    adapter_name: str = "YouthNaturalLoRA"
    format_version: int = 1
    base_checkpoint_hash: str = "not_run"
    created_at: str = "2026-06-19"
    rank: int = 8
    alpha: float = 16.0
    scaling: str = "alpha / rank"
    dtype: str = "float32"
    target_modules: list[dict[str, Any]] = field(default_factory=list)
    strength_behavior: str = "strength 0.0 produces exact zero adapter contribution"
    lineage: dict[str, Any] = field(default_factory=lambda: {"dataset_snapshots": []})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def value_slice_for_wkv(weight_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(weight_shape) != 3 or weight_shape[0] != 2:
        raise ValueError("expected ChunkedLinear wkv weight shape (2, kv_dim, in_features)")
    kv_dim = int(weight_shape[1])
    return (0, kv_dim)


def assert_value_slice_only(
    original: mx.array,
    merged: mx.array,
    value_slice: tuple[int, int],
) -> None:
    if original.shape != merged.shape:
        raise AssertionError("shape changed")
    if len(original.shape) != 3:
        raise AssertionError("expected chunked wkv shape")
    if not bool(mx.allclose(original[0], merged[0])):
        raise AssertionError("key slice changed")
    start, end = value_slice
    before = merged[1, :start]
    after = merged[1, end:]
    if not bool(mx.allclose(original[1, :start], before)):
        raise AssertionError("value prefix changed")
    if not bool(mx.allclose(original[1, end:], after)):
        raise AssertionError("value suffix changed")


def write_adapter_manifest(path: str | Path, manifest: AdapterManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

