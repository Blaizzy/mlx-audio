from dataclasses import asdict

import mlx.core as mx
import pytest
from conftest import assert_exactly_equal
from mlx.utils import tree_flatten

upstream = pytest.importorskip("mlx_lm.models.qwen3_5")

from mlx_audio.lm.models import qwen3_5 as vendored


def text_args(module):
    # Three gated-delta layers and one full-attention layer, all MoE with a
    # shared expert, like the Qwen3.5 MoE checkpoints.
    return module.TextModelArgs(
        model_type="qwen3_5_moe_text",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=96,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=128,
        moe_intermediate_size=128,
        max_position_embeddings=1024,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    )


def models():
    u = upstream.TextModel(text_args(upstream))
    v = vendored.TextModel(text_args(vendored))
    v.load_weights(tree_flatten(u.parameters()), strict=True)
    # Eval mode selects the Metal gated-delta kernel.
    u.eval()
    v.eval()
    return v, u


def test_config_and_parameter_keys_match_upstream():
    assert asdict(text_args(vendored)) == asdict(text_args(upstream))
    v, u = models()
    assert {k: p.shape for k, p in tree_flatten(v.parameters())} == {
        k: p.shape for k, p in tree_flatten(u.parameters())
    }


def test_cached_decode_from_embeddings_matches_upstream():
    v, u = models()
    v_cache, u_cache = v.make_cache(), u.make_cache()
    for length in (7, 1, 1, 3):
        x = mx.random.normal((1, length, 64))
        assert_exactly_equal(
            v(None, cache=v_cache, input_embeddings=x),
            u(None, cache=u_cache, input_embeddings=x),
        )
    for v_layer, u_layer in zip(v_cache, u_cache):
        upstream_state = (
            u_layer.cache
            if isinstance(v_layer, vendored.ArraysCache)
            else u_layer.keys_and_values()
        )
        for got, want in zip(v_layer.state, upstream_state):
            assert_exactly_equal(got, want)
