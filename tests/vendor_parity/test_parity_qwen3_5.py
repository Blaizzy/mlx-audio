import mlx.core as mx
import pytest
from conftest import assert_exactly_equal
from mlx.utils import tree_flatten

upstream = pytest.importorskip("mlx_lm.models.qwen3_5")

from mlx_audio.lm.models import qwen3_5 as vendored

TEXT_CONFIG = dict(
    model_type="qwen3_5_moe_text",
    hidden_size=32,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    rms_norm_eps=1e-6,
    vocab_size=64,
    max_position_embeddings=128,
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    shared_expert_intermediate_size=16,
    moe_intermediate_size=16,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
)


def make_models():
    vendored_model = vendored.Qwen3_5TextModel(
        vendored.TextModelArgs.from_dict(dict(TEXT_CONFIG))
    )
    upstream_model = upstream.Qwen3_5TextModel(
        upstream.TextModelArgs.from_dict(dict(TEXT_CONFIG))
    )
    upstream_model.update(vendored_model.parameters())
    return vendored_model, upstream_model


def test_parameter_keys_match_upstream():
    vendored_model, upstream_model = make_models()

    def shapes(model):
        return {k: v.shape for k, v in tree_flatten(model.parameters())}

    assert shapes(vendored_model) == shapes(upstream_model)


def test_forward_matches_upstream():
    vendored_model, upstream_model = make_models()
    tokens = mx.array([[1, 7, 3, 9, 2, 5]])
    assert_exactly_equal(vendored_model(tokens), upstream_model(tokens))


def test_cached_forward_matches_upstream():
    vendored_model, upstream_model = make_models()
    vendored_cache = vendored_model.make_cache()
    upstream_cache = [
        upstream.ArraysCache(size=2) if l.is_linear else upstream.KVCache()
        for l in upstream_model.layers
    ]
    prompt = mx.array([[1, 7, 3]])
    assert_exactly_equal(
        vendored_model(prompt, cache=vendored_cache),
        upstream_model(prompt, cache=upstream_cache),
    )
    for token in [9, 2, 5]:
        inputs = mx.array([[token]])
        assert_exactly_equal(
            vendored_model(inputs, cache=vendored_cache),
            upstream_model(inputs, cache=upstream_cache),
        )


def test_accepts_precomputed_embeddings():
    vendored_model, _ = make_models()
    tokens = mx.array([[1, 7, 3, 9]])
    embeddings = vendored_model.embed_tokens(tokens)
    assert_exactly_equal(
        vendored_model(tokens), vendored_model(None, input_embeddings=embeddings)
    )
