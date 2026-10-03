from dataclasses import asdict

import pytest
from mlx.utils import tree_flatten

upstream_bailing = pytest.importorskip("mlx_lm.models.bailing_moe")
upstream_lfm2 = pytest.importorskip("mlx_lm.models.lfm2")

from mlx_audio.lm.models import bailing_moe as vendored_bailing
from mlx_audio.lm.models import lfm2 as vendored_lfm2


def parameter_shapes(model):
    return {
        name: parameter.shape for name, parameter in tree_flatten(model.parameters())
    }


def lfm2_args(module):
    return module.ModelArgs(
        model_type="lfm2",
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        norm_eps=1e-5,
        conv_bias=False,
        conv_L_cache=4,
        block_dim=16,
        block_ff_dim=32,
        block_multiple_of=8,
        block_ffn_dim_multiplier=1.0,
        block_auto_adjust_ff_dim=False,
        full_attn_idxs=[0],
        layer_types=["full_attention"],
    )


def bailing_args(module):
    return module.ModelArgs(
        model_type="bailing_moe",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=128,
        moe_intermediate_size=16,
        num_experts=2,
        num_shared_experts=0,
        norm_topk_prob=True,
        num_attention_heads=2,
        num_experts_per_tok=1,
        num_hidden_layers=1,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        vocab_size=64,
        first_k_dense_replace=1,
    )


@pytest.mark.parametrize(
    ("vendored", "upstream", "make_args"),
    [
        (vendored_lfm2, upstream_lfm2, lfm2_args),
        (vendored_bailing, upstream_bailing, bailing_args),
    ],
)
def test_config_and_parameter_keys_match_upstream(vendored, upstream, make_args):
    vendored_args = make_args(vendored)
    upstream_args = make_args(upstream)

    assert asdict(vendored_args) == asdict(upstream_args)
    assert parameter_shapes(vendored.Model(vendored_args)) == parameter_shapes(
        upstream.Model(upstream_args)
    )
