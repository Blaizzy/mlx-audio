from dataclasses import asdict

import mlx.core as mx
import pytest
from conftest import assert_exactly_equal
from mlx.utils import tree_flatten

upstream = pytest.importorskip("mlx_lm.models.cohere2")

from mlx_audio.lm.models import cohere2 as vendored


def make_args(module):
    return module.ModelArgs(
        model_type="cohere2",
        hidden_size=16,
        head_dim=8,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        sliding_window=4,
        sliding_window_pattern=4,
        logit_scale=1.0,
    )


def parameter_shapes(model):
    return {
        name: parameter.shape for name, parameter in tree_flatten(model.parameters())
    }


def test_cohere2_config_and_parameter_keys_match_upstream():
    vendored_args = make_args(vendored)
    upstream_args = make_args(upstream)

    assert asdict(vendored_args) == asdict(upstream_args)
    assert parameter_shapes(vendored.Model(vendored_args)) == parameter_shapes(
        upstream.Model(upstream_args)
    )


def test_cohere2_forward_matches_upstream_with_cache():
    v = vendored.Model(make_args(vendored))
    u = upstream.Model(make_args(upstream))
    u.update(v.parameters())

    mx.random.seed(0)
    prompt = mx.random.randint(0, 64, (1, 9))
    v_cache, u_cache = v.make_cache(), u.make_cache()
    assert [type(c).__name__ for c in v_cache] == [type(c).__name__ for c in u_cache]

    assert_exactly_equal(v(prompt, cache=v_cache), u(prompt, cache=u_cache))
    # A second step exercises the rotating sliding-window caches past their size.
    step = mx.random.randint(0, 64, (1, 1))
    assert_exactly_equal(v(step, cache=v_cache), u(step, cache=u_cache))
