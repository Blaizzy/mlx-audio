from dataclasses import asdict

import pytest
from mlx.utils import tree_flatten

upstream_gemma3 = pytest.importorskip("mlx_lm.models.gemma3")
upstream_gemma3_text = pytest.importorskip("mlx_lm.models.gemma3_text")

from mlx_audio.lm.models import gemma3 as vendored_gemma3
from mlx_audio.lm.models import gemma3_text as vendored_gemma3_text


def parameter_shapes(model):
    return {
        name: parameter.shape
        for name, parameter in tree_flatten(model.parameters())
    }


def text_args(module):
    return module.ModelArgs(
        model_type="gemma3_text",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        head_dim=8,
        vocab_size=64,
        num_key_value_heads=2,
        sliding_window=8,
        sliding_window_pattern=1,
    )


def gemma3_args(module):
    return module.ModelArgs(
        model_type="gemma3",
        vocab_size=64,
        text_config=asdict(text_args(vendored_gemma3_text)),
    )


@pytest.mark.parametrize(
    ("vendored", "upstream", "make_args"),
    [
        (vendored_gemma3_text, upstream_gemma3_text, text_args),
        (vendored_gemma3, upstream_gemma3, gemma3_args),
    ],
)
def test_config_and_parameter_keys_match_upstream(vendored, upstream, make_args):
    vendored_args = make_args(vendored)
    upstream_args = make_args(upstream)

    assert asdict(vendored_args) == asdict(upstream_args)
    assert parameter_shapes(vendored.Model(vendored_args)) == parameter_shapes(
        upstream.Model(upstream_args)
    )
