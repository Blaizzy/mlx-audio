from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_audio.stt.models.mega_asr.convert_lora import LoraModule


class _SelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 32)


class _AudioLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _SelfAttn()


class _AudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_AudioLayer()]
        self.conv_out = nn.Linear(20, 8, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(24, 12, bias=False)


class _TextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()


class _TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_TextLayer()]


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_tower = _AudioTower()
        self.model = _TextModel()


def _build_stub_and_adapter() -> tuple[_StubModel, dict[str, LoraModule]]:
    model = _StubModel()
    r = 4
    adapter: dict[str, LoraModule] = {
        "audio_tower.layers.0.self_attn.q_proj": {
            "A": mx.random.normal((r, 16)),
            "B": mx.random.normal((32, r)),
            "scaling": 1.0,
        },
        "audio_tower.conv_out": {
            "A": mx.random.normal((r, 20)),
            "B": mx.random.normal((8, r)),
            "scaling": 0.5,
        },
        "model.layers.0.mlp.down_proj": {
            "A": mx.random.normal((r, 24)),
            "B": mx.random.normal((12, r)),
            "scaling": 2.0,
        },
    }
    return model, adapter


def test_resolve_linear_returns_correct_leaf():
    from mlx_audio.stt.models.mega_asr.lora import resolve_linear

    model, _ = _build_stub_and_adapter()

    q = resolve_linear(model, "audio_tower.layers.0.self_attn.q_proj")
    assert q is model.audio_tower.layers[0].self_attn.q_proj
    assert isinstance(q, nn.Linear)

    d = resolve_linear(model, "model.layers.0.mlp.down_proj")
    assert d is model.model.layers[0].mlp.down_proj
    assert isinstance(d, nn.Linear)


def test_apply_remove_roundtrip():
    from mlx_audio.stt.models.mega_asr.lora import (
        apply_deltas,
        remove_deltas,
        resolve_linear,
    )

    model, adapter = _build_stub_and_adapter()
    resolve_linear(model, "audio_tower.conv_out").weight = resolve_linear(
        model, "audio_tower.conv_out"
    ).weight.astype(mx.bfloat16)
    paths = list(adapter)

    base = {
        p: np.array(resolve_linear(model, p).weight.astype(mx.float32)) for p in paths
    }

    apply_deltas(model, adapter)
    for p in paths:
        after = np.array(resolve_linear(model, p).weight.astype(mx.float32))
        assert not np.allclose(after, base[p])

    remove_deltas(model, adapter)
    for p in paths:
        restored = np.array(resolve_linear(model, p).weight.astype(mx.float32))
        atol = 5e-2 if resolve_linear(model, p).weight.dtype == mx.bfloat16 else 1e-5
        assert np.allclose(restored, base[p], atol=atol)


def test_double_apply_is_guarded():
    from mlx_audio.stt.models.mega_asr.lora import apply_deltas

    model, adapter = _build_stub_and_adapter()

    apply_deltas(model, adapter)
    with pytest.raises(RuntimeError):
        apply_deltas(model, adapter)


def test_remove_when_inactive_is_guarded():
    from mlx_audio.stt.models.mega_asr.lora import remove_deltas

    model, adapter = _build_stub_and_adapter()

    with pytest.raises(RuntimeError):
        remove_deltas(model, adapter)


def test_fp16_weight_keeps_dtype_and_accumulates_in_fp32():
    from mlx_audio.stt.models.mega_asr.lora import (
        apply_deltas,
        materialize_delta,
        resolve_linear,
    )

    model, adapter = _build_stub_and_adapter()
    path = "audio_tower.layers.0.self_attn.q_proj"
    leaf = resolve_linear(model, path)
    leaf.weight = leaf.weight.astype(mx.float16)
    base = leaf.weight

    apply_deltas(model, adapter)

    out = resolve_linear(model, path).weight
    assert out.dtype == mx.float16
    expected = np.array(
        (base + materialize_delta(adapter[path]).astype(mx.float16)).astype(mx.float32)
    )
    assert np.allclose(np.array(out.astype(mx.float32)), expected, atol=1e-2)
