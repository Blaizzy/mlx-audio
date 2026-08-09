from dataclasses import asdict

import pytest
from mlx.utils import tree_flatten

upstream_qwen2 = pytest.importorskip("mlx_lm.models.qwen2")
upstream_qwen3 = pytest.importorskip("mlx_lm.models.qwen3")

from mlx_audio.lm.models import qwen2 as vendored_qwen2
from mlx_audio.lm.models import qwen3 as vendored_qwen3


def parameter_shapes(model):
    return {
        name: parameter.shape
        for name, parameter in tree_flatten(model.parameters())
    }


def qwen2_args(module):
    return module.ModelArgs(
        model_type="qwen2",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
    )


def qwen3_args(module):
    return module.ModelArgs(
        model_type="qwen3",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        max_position_embeddings=128,
        rope_theta=1000000.0,
        head_dim=8,
        tie_word_embeddings=True,
    )


@pytest.mark.parametrize(
    ("vendored", "upstream", "make_args"),
    [
        (vendored_qwen2, upstream_qwen2, qwen2_args),
        (vendored_qwen3, upstream_qwen3, qwen3_args),
    ],
)
def test_qwen_config_and_parameter_keys_match_upstream(vendored, upstream, make_args):
    vendored_args = make_args(vendored)
    upstream_args = make_args(upstream)

    assert asdict(vendored_args) == asdict(upstream_args)
    assert parameter_shapes(vendored.Model(vendored_args)) == parameter_shapes(
        upstream.Model(upstream_args)
    )
