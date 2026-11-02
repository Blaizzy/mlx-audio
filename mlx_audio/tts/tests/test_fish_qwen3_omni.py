# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Keep Fish checkpoint loading independent of its conversion policy."""

import json
import unittest
from dataclasses import asdict
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_audio.convert import build_quant_predicate, convert
from mlx_audio.tts.models.fish_qwen3_omni import Model, ModelConfig
from mlx_audio.utils import base_load_model


class TestFishSpeechModelQuantPredicate(unittest.TestCase):
    def setUp(self):
        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

        self.predicate = Model.model_quant_predicate

    def test_embeddings_are_excluded(self):
        emb = nn.Embedding(128, 64)
        self.assertFalse(self.predicate("model.embeddings", emb))
        self.assertFalse(self.predicate("model.codebook_embeddings", emb))

    def test_fast_path_is_excluded(self):
        linear = nn.Linear(64, 64)
        self.assertFalse(self.predicate("model.fast_layers.0", linear))
        self.assertFalse(self.predicate("model.fast_output", linear))
        self.assertFalse(self.predicate("model.fast_project_in", linear))

    def test_regular_linear_layers_are_included(self):
        linear = nn.Linear(64, 64)
        self.assertTrue(self.predicate("model.layers.0.self_attn.q_proj", linear))
        self.assertTrue(self.predicate("model.text_encoder.block.0.mlp", linear))


@pytest.fixture
def fish_config():
    layer_config = {
        "vocab_size": 128,
        "dim": 64,
        "intermediate_size": 128,
        "n_layer": 1,
        "n_head": 2,
        "n_local_heads": 1,
        "head_dim": 32,
        "max_seq_len": 16,
    }
    return ModelConfig.from_dict(
        {
            "model_type": "fish_qwen3_omni",
            "semantic_start_token_id": 64,
            "semantic_end_token_id": 127,
            "text_config": layer_config,
            "audio_decoder_config": {
                **layer_config,
                "text_dim": 64,
                "num_codebooks": 2,
            },
        }
    )


def write_checkpoint(directory, model, config):
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(
        str(directory / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )


def load_checkpoint(directory):
    # These checkpoints exercise the transformer, without tokenizer/codec assets.
    with patch.object(Model, "post_load_hook", side_effect=lambda model, _: model):
        return base_load_model(directory, category="tts", model_remapping={})


def forward_outputs(model):
    inputs = mx.array([[[64, 65], [1, 2], [3, 4]]], dtype=mx.int32)
    slow = model.model(inputs)
    fast = model.model.fast_forward(slow.hidden_states, mx.array([[1]], dtype=mx.int32))
    cached = model.model.fast_forward_cached(
        slow.hidden_states, model.model.make_fast_cache()
    )
    return slow.logits, fast, cached


@pytest.mark.parametrize(
    "layout,bits",
    [("full", 4), ("full", 8), ("partial", 8), ("mixed", 8), ("bf16", None)],
)
def test_checkpoint_layout_round_trip(tmp_path, fish_config, layout, bits):
    source = Model(fish_config)
    source.set_dtype(mx.bfloat16)
    config = asdict(fish_config)
    if bits is not None:
        quantization = {"group_size": 64, "bits": bits, "mode": "affine"}
        predicate = None
        if layout == "partial":
            predicate = build_quant_predicate(source)
        elif layout == "mixed":
            override = {"group_size": 32, "bits": 4, "mode": "affine"}
            quantization["model.fast_output"] = override
            predicate = lambda p, m: (
                override if p == "model.fast_output" else hasattr(m, "to_quantized")
            )
        nn.quantize(source, group_size=64, bits=bits, class_predicate=predicate)
        config["quantization"] = quantization
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint(checkpoint, source, config)

    restored = load_checkpoint(checkpoint)

    expected = dict(tree_flatten(source.parameters()))
    actual = dict(tree_flatten(restored.parameters()))
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype
        assert actual[name].shape == value.shape
        assert bool(mx.array_equal(actual[name], value))
    for original, loaded in zip(forward_outputs(source), forward_outputs(restored)):
        assert bool(mx.all(mx.isfinite(loaded)))
        assert bool(mx.array_equal(original, loaded))
    if layout == "mixed":
        assert restored.model.fast_output.bits == 4
        assert restored.model.fast_output.group_size == 32


@pytest.mark.parametrize("quantized_source", [False, True])
def test_conversion_preserves_quantization_policy(
    tmp_path, fish_config, quantized_source
):
    source = Model(fish_config)
    source.set_dtype(mx.bfloat16)
    config = asdict(fish_config)
    if quantized_source:
        nn.quantize(source, group_size=64, bits=8)
        config["quantization"] = {"group_size": 64, "bits": 8}
    checkpoint = tmp_path / "source"
    output = tmp_path / "converted"
    write_checkpoint(checkpoint, source, config)

    convert(
        hf_path=str(checkpoint),
        mlx_path=str(output),
        quantize=not quantized_source,
        q_group_size=64,
        q_bits=8,
        model_domain="tts",
    )

    restored = load_checkpoint(output)
    quantized_modules = {
        name.removesuffix(".scales")
        for name, _ in tree_flatten(restored.parameters())
        if name.endswith(".scales")
    }
    if quantized_source:
        expected = {
            name.removesuffix(".scales")
            for name, _ in tree_flatten(source.parameters())
            if name.endswith(".scales")
        }
    else:
        predicate = build_quant_predicate(source)
        expected = {
            name for name, module in source.named_modules() if predicate(name, module)
        }
        assert expected
        assert not any("embedding" in name or "fast_" in name for name in expected)
    assert quantized_modules == expected
    for output in forward_outputs(restored):
        assert bool(mx.all(mx.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
