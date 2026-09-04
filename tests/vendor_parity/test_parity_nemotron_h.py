from dataclasses import asdict

import mlx.core as mx
import pytest
from conftest import assert_exactly_equal
from mlx.utils import tree_flatten

upstream = pytest.importorskip("mlx_lm.models.nemotron_h")

from mlx_audio.lm.models import nemotron_h as vendored


def make_args(module):
    return module.ModelArgs(
        model_type="nemotron_h",
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_bias=False,
        mamba_num_heads=2,
        mamba_head_dim=4,
        mamba_proj_bias=False,
        ssm_state_size=8,
        conv_kernel=4,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=True,
        hybrid_override_pattern=["M", "*", "-"],
    )


def parameter_shapes(model):
    return {
        name: parameter.shape for name, parameter in tree_flatten(model.parameters())
    }


def test_config_and_parameter_keys_match_upstream():
    vendored_args = make_args(vendored)
    upstream_args = make_args(upstream)

    assert asdict(vendored_args) == asdict(upstream_args)
    assert parameter_shapes(vendored.Model(vendored_args)) == parameter_shapes(
        upstream.Model(upstream_args)
    )


def test_hugging_face_block_names_are_normalized():
    config = asdict(make_args(vendored))
    config.pop("num_hidden_layers")
    config.pop("hybrid_override_pattern")
    config["layers_block_type"] = [
        "linear_attention",
        "full_attention",
        "moe",
        "mlp",
    ]

    args = vendored.ModelArgs.from_dict(config)

    assert args.num_hidden_layers == 4
    assert args.hybrid_override_pattern == ["M", "*", "E", "-"]


def test_forward_matches_upstream():
    vendored_model = vendored.Model(make_args(vendored))
    upstream_model = upstream.Model(make_args(upstream))
    upstream_model.update(vendored_model.parameters())

    tokens = mx.array([[1, 7, 3, 9]])
    assert_exactly_equal(vendored_model(tokens), upstream_model(tokens))


def test_backbone_accepts_precomputed_embeddings():
    model = vendored.Model(make_args(vendored))
    tokens = mx.array([[1, 7, 3, 9]])
    embeddings = model.backbone.embeddings(tokens)

    assert_exactly_equal(
        model.backbone(tokens),
        model.backbone(input_embeddings=embeddings),
    )


def test_cached_forward_matches_upstream():
    vendored_model = vendored.Model(make_args(vendored))
    upstream_model = upstream.Model(make_args(upstream))
    upstream_model.update(vendored_model.parameters())
    vendored_cache = vendored_model.make_cache()
    upstream_cache = upstream_model.make_cache()

    for token in [1, 7, 3, 9]:
        inputs = mx.array([[token]])
        assert_exactly_equal(
            vendored_model(inputs, cache=vendored_cache),
            upstream_model(inputs, cache=upstream_cache),
        )
