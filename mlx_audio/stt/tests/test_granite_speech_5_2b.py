"""Offline coverage for the autoregressive Granite Speech 5.0 checkpoint."""

import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.convert import Domain, detect_model_domain, get_model_type
from mlx_audio.lm.models.rope_utils import initialize_rope
from mlx_audio.stt.models.granite_speech import copy_supporting_files, write_model_card
from mlx_audio.stt.models.granite_speech.config import ModelConfig
from mlx_audio.stt.models.granite_speech.granite_speech import (
    TASK_PROMPTS,
    Model,
    _resolve_prompt,
)
from mlx_audio.stt.utils import MODEL_REMAPPING
from mlx_audio.utils import get_model_class


def tiny_config():
    return ModelConfig.from_dict(
        {
            "model_type": "granite_speech",
            "encoder_config": {
                "input_dim": 160,
                "hidden_dim": 8,
                "num_layers": 2,
                "num_heads": 2,
                "dim_head": 4,
                "output_dim": 348,
                "context_size": 8,
                "max_pos_emb": 16,
                "conv_kernel_size": 3,
            },
            "projector_config": {
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 16,
                "encoder_hidden_size": 8,
            },
            "text_config": {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "max_position_embeddings": 16384,
                "rope_scaling": {
                    "factor": 4.0,
                    "original_max_position_embeddings": 4096,
                    "rope_type": "yarn",
                },
            },
        }
    )


def test_native_config_routes_to_autoregressive_granite(tmp_path):
    config = {
        "model_type": "granite_speech",
        "architectures": ["GraniteSpeechForConditionalGeneration"],
        "encoder_config": {"num_layers": 16, "output_dim": 348},
        "projector_config": {"encoder_hidden_size": 1024},
        "audio_token_index": 100352,
        "text_config": {
            "max_position_embeddings": 16384,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 4096,
            },
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    domain = detect_model_domain(config, tmp_path)
    assert domain == Domain.STT
    assert get_model_type(config, tmp_path, domain) == "granite_speech"
    module, _ = get_model_class(
        model_type="granite_speech",
        model_name=None,
        category="stt",
        model_remapping=MODEL_REMAPPING,
    )
    assert module.Model is Model
    parsed = module.ModelConfig.from_dict(config)
    assert parsed.encoder_config.output_dim == 348
    assert parsed.encoder_config.num_layers == 16
    assert parsed.text_config.rope_scaling == config["text_config"]["rope_scaling"]
    assert parsed.text_config.max_position_embeddings == 16384


@pytest.mark.parametrize("with_scales", [False, True])
def test_sanitize_handles_auxiliary_head_and_repeated_loading(with_scales):
    model = Model(tiny_config())
    native = dict(tree_flatten(model.parameters()))
    source = dict(native)
    for key, value in native.items():
        if key.endswith(".weight") and value.ndim == 3:
            source[key] = value.transpose(0, 2, 1)
    source["weight"] = mx.zeros((32, 8))
    source["bias"] = mx.zeros((32,))
    source["encoder.layers.0.conv.batch_norm.num_batches_tracked"] = mx.array(0)
    if with_scales:
        source["language_model.lm_head.scales"] = mx.ones((32, 1))

    sanitized = Model.sanitize(source)
    assert "weight" not in sanitized and "bias" not in sanitized
    assert not any("num_batches_tracked" in key for key in sanitized)
    for key, expected in native.items():
        np.testing.assert_array_equal(sanitized[key], expected)
        np.testing.assert_array_equal(Model.sanitize(sanitized)[key], expected)
    sanitized.pop("language_model.lm_head.scales", None)
    model.load_weights(list(sanitized.items()), strict=True)


def test_audio_projector_and_cached_decoder_forward():
    model = Model(tiny_config())
    model.eval()
    features, count = model._extract_features(mx.zeros((4800,)))
    projected = model.get_audio_features(features)
    assert features.shape == (1, 15, 160)
    assert projected.shape == (1, count, 16)
    assert count == 3
    cache = model.make_cache()
    logits = model(mx.array([[1, 2, 3]]), cache=cache, input_embeddings=projected)
    next_logits = model(mx.array([[4]]), cache=cache)
    mx.eval(logits, next_logits)
    assert logits.shape == (1, 3, 32)
    assert next_logits.shape == (1, 1, 32)
    assert mx.all(mx.isfinite(next_logits)).item()
    assert all(layer.offset == 4 for layer in cache)


def test_yarn_matches_reference_equations_beyond_original_context():
    dim, factor, original = 128, 4.0, 4096
    rope = initialize_rope(
        dim,
        10000.0,
        False,
        {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": original,
        },
        16384,
    )
    low = np.floor(dim * np.log(original / (32 * 2 * np.pi)) / (2 * np.log(10000)))
    high = np.ceil(dim * np.log(original / (2 * np.pi)) / (2 * np.log(10000)))
    ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
    frequencies = 10000 ** (np.arange(0, dim, 2) / dim)
    inv_freq = (1 - ramp) / frequencies + ramp / (factor * frequencies)
    np.testing.assert_allclose(1 / np.asarray(rope._freqs), inv_freq, rtol=1e-6)
    x = mx.ones((1, 1, 2, dim))
    result = np.asarray(rope(x, offset=8192))
    angles = np.arange(8192, 8194)[:, None] * inv_freq[None, :]
    scale = 1 + 0.1 * np.log(factor)
    expected = scale * np.concatenate(
        [np.cos(angles) - np.sin(angles), np.sin(angles) + np.cos(angles)], axis=-1
    )
    np.testing.assert_allclose(result[0, 0], expected, atol=2e-3)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, TASK_PROMPTS["asr"]),
        (
            {"source_language": "ja"},
            "The speech is in Japanese. Transcribe it exactly; do not translate. "
            + TASK_PROMPTS["asr"],
        ),
        (
            {"hotwords": [" IBM ", None, "", "Granite"]},
            "transcribe the speech to text. Keywords: IBM, Granite",
        ),
        ({"hotwords": [" "]}, TASK_PROMPTS["asr"]),
        (
            {"punctuate": True},
            "transcribe the speech with proper punctuation and capitalization.",
        ),
        (
            {"language": "zh", "punctuate": True, "hotwords": ["IBM"]},
            "Translate the speech to Mandarin with proper punctuation and capitalization. Keywords: IBM",
        ),
        ({"language": "it"}, "Translate the speech to Italian."),
        ({"prompt": "custom", "hotwords": ["IBM"]}, "custom Keywords: IBM"),
    ],
)
def test_five_point_zero_prompts(kwargs, expected):
    params = {"prompt": None, "language": None, **kwargs}
    assert _resolve_prompt("asr", **params) == expected


def test_prompt_options_reject_conflicting_tasks():
    with pytest.raises(ValueError, match="Pass only one"):
        _resolve_prompt("asr", None, "fr", source_language="en")
    with pytest.raises(ValueError, match="require task='asr'"):
        _resolve_prompt("timestamps", None, None, hotwords=["IBM"])


def test_streaming_receives_same_prompt_options():
    forwarded = {}

    def stream_generate(audio, **kwargs):
        forwarded.update(kwargs)
        return iter(())

    stub = SimpleNamespace(is_plus=False, _stream_generate=stream_generate)
    list(
        Model.generate(
            stub,
            mx.zeros((1,)),
            stream=True,
            source_language="de",
            hotwords=["IBM"],
            punctuate=True,
        )
    )
    assert forwarded["prompt"] == (
        "The speech is in German. Transcribe it exactly; do not translate. "
        "transcribe the speech with proper punctuation and capitalization. Keywords: IBM"
    )
    assert forwarded["system_prompt"] is None


def test_conversion_preserves_optional_bpe_head(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "chat_template.jinja",
        "out_llm.safetensors",
        "model.safetensors",
        "model.safetensors.index.json",
    ):
        (source / name).write_text("test")
    copy_supporting_files(source, dest)
    assert {p.name for p in dest.iterdir()} == {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "chat_template.jinja",
        "out_llm.safetensors",
    }


def test_conversion_writes_model_card(tmp_path):
    source = "ibm-granite/granite-speech-5.0-2b"
    target = "mlx-community/granite-speech-5.0-2b-8bit"
    write_model_card(tmp_path, source, target, {})
    content = (tmp_path / "README.md").read_text()
    assert f"base_model: {source}" in content
    assert f"model = load({target!r})" in content
    assert "16 kHz" in content
