from dataclasses import asdict

import pytest
from mlx.utils import tree_flatten

upstream = pytest.importorskip("mlx_lm.models.llama")

from mlx_audio.lm.models import llama as vendored


def make_args(module):
    return module.ModelArgs(
        model_type="llama",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
    )


def parameter_shapes(model):
    return {
        name: parameter.shape
        for name, parameter in tree_flatten(model.parameters())
    }


def test_llama_config_and_parameter_keys_match_upstream():
    vendored_args = make_args(vendored)
    upstream_args = make_args(upstream)

    assert asdict(vendored_args) == asdict(upstream_args)
    assert parameter_shapes(vendored.Model(vendored_args)) == parameter_shapes(
        upstream.Model(upstream_args)
    )
