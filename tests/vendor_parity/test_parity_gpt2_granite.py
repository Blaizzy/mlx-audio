from dataclasses import asdict

import pytest
from mlx.utils import tree_flatten

upstream_gpt2 = pytest.importorskip("mlx_lm.models.gpt2")
upstream_granite = pytest.importorskip("mlx_lm.models.granite")

from mlx_audio.lm.models import gpt2 as vendored_gpt2
from mlx_audio.lm.models import granite as vendored_granite


def parameter_shapes(model):
    return {
        name: parameter.shape
        for name, parameter in tree_flatten(model.parameters())
    }


def gpt2_args(module):
    return module.ModelArgs(
        model_type="gpt2",
        n_ctx=128,
        n_embd=16,
        n_head=2,
        n_layer=1,
        n_positions=128,
        layer_norm_epsilon=1e-5,
        vocab_size=64,
    )


def granite_args(module):
    return module.ModelArgs(
        model_type="granite",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        logits_scaling=1.0,
        attention_multiplier=16**-0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        max_position_embeddings=128,
        attention_bias=False,
        mlp_bias=False,
        rope_theta=10000.0,
    )


@pytest.mark.parametrize(
    ("vendored", "upstream", "make_args"),
    [
        (vendored_gpt2, upstream_gpt2, gpt2_args),
        (vendored_granite, upstream_granite, granite_args),
    ],
)
def test_config_and_parameter_keys_match_upstream(vendored, upstream, make_args):
    vendored_args = make_args(vendored)
    upstream_args = make_args(upstream)

    assert asdict(vendored_args) == asdict(upstream_args)
    assert parameter_shapes(vendored.Model(vendored_args)) == parameter_shapes(
        upstream.Model(upstream_args)
    )
