import dataclasses
import itertools
from dataclasses import dataclass

import mlx.core as mx
import pytest
from conftest import assert_exactly_equal

upstream = pytest.importorskip("mlx_lm.models.base")

from mlx_audio.lm.models import base as vendored

SHAPES = list(
    itertools.product(
        [1, 2, 5, 8, 17],  # N
        [0, 1, 7, 64],  # offset
        [None, 1, 4, 8],  # window_size
    )
)


@pytest.mark.parametrize("n,offset,window", SHAPES)
def test_create_causal_mask_matches_upstream(n, offset, window):
    kwargs = dict(offset=offset, window_size=window)
    assert_exactly_equal(
        vendored.create_causal_mask(n, **kwargs),
        upstream.create_causal_mask(n, **kwargs),
    )


@pytest.mark.parametrize("left", [None, mx.array([0]), mx.array([2, 5])])
@pytest.mark.parametrize("right", [None, mx.array([0]), mx.array([3, 0])])
def test_create_causal_mask_padding_matches_upstream(left, right):
    kwargs = dict(offset=3, left_padding=left, right_padding=right)
    assert_exactly_equal(
        vendored.create_causal_mask(6, **kwargs),
        upstream.create_causal_mask(6, **kwargs),
    )


@pytest.mark.parametrize("n", [1, 2, 8, 17])
@pytest.mark.parametrize("window", [None, 4, 32])
@pytest.mark.parametrize("return_array", [False, True])
def test_create_attention_mask_returns_identical_type(n, window, return_array):
    h = mx.zeros((1, n, 8))
    got = vendored.create_attention_mask(
        h, window_size=window, return_array=return_array
    )
    want = upstream.create_attention_mask(
        h, window_size=window, return_array=return_array
    )
    assert type(got) is type(want)
    if isinstance(want, mx.array):
        assert_exactly_equal(got, want)
    else:
        assert got == want


def test_create_ssm_mask_matches_upstream():
    h = mx.zeros((1, 5, 8))
    assert vendored.create_ssm_mask(h) == upstream.create_ssm_mask(h)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("mask_kind", [None, "causal", "bool", "additive"])
@pytest.mark.parametrize("n_repeats", [1, 4])
def test_scaled_dot_product_attention_matches_upstream(dtype, mask_kind, n_repeats):
    b, n_kv, seq, d = 2, 2, 6, 16
    q = mx.random.normal((b, n_kv * n_repeats, seq, d)).astype(dtype)
    k = mx.random.normal((b, n_kv, seq, d)).astype(dtype)
    v = mx.random.normal((b, n_kv, seq, d)).astype(dtype)
    if mask_kind == "bool":
        mask = vendored.create_causal_mask(seq)
    elif mask_kind == "additive":
        mask = mx.where(vendored.create_causal_mask(seq), 0.0, -1e9).astype(dtype)
    else:
        mask = mask_kind
    scale = d**-0.5
    assert_exactly_equal(
        vendored.scaled_dot_product_attention(q, k, v, None, scale, mask),
        upstream.scaled_dot_product_attention(q, k, v, None, scale, mask),
    )


def test_base_model_args_from_dict_matches_upstream():
    @dataclass
    class V(vendored.BaseModelArgs):
        a: int = 1
        b: str = "x"

    @dataclass
    class U(upstream.BaseModelArgs):
        a: int = 1
        b: str = "x"

    params = {"a": 7, "b": "y", "unknown": 3}
    assert dataclasses.astuple(V.from_dict(params)) == dataclasses.astuple(
        U.from_dict(params)
    )
