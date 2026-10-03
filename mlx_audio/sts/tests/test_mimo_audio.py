import json
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.convert import Domain, detect_model_domain, get_model_type
from mlx_audio.registry import classify_model, model_type_from_config
from mlx_audio.sts.models.mimo_audio import MiMoAudioProcessor, Model, ModelConfig
from mlx_audio.sts.models.mimo_audio.generation import (
    Sampler,
    generate_tokens,
    split_reasoning,
)
from mlx_audio.sts.models.mimo_audio.model import ModelOutput


class Tokenizer:
    def __init__(self):
        self.special = {
            f"<|{name}|>": i + 1
            for i, name in enumerate(
                ["empty", "sosp", "eosp", "sostm", "eostm", "eot", "im_end", "im_start"]
            )
        }
        self.eos_token_id = 9

    def convert_tokens_to_ids(self, token):
        return self.special.get(token)

    def encode(self, text, add_special_tokens=False):
        result = []
        for token in re.split(r"(<\|[^>]+\|>)", text):
            if token in self.special:
                result.append(self.special[token])
            else:
                result.extend(16 + ord(ch) % 32 for ch in token)
        return result

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def config(**kwargs):
    return replace(
        ModelConfig(
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            local_dim=16,
            local_layers=2,
            local_attn_heads=2,
            local_ffn_dim=32,
            input_local_dim=16,
            input_local_layers=1,
            audio_channels=2,
            speech_vocab_size=[9, 5],
            speech_zeroemb_idx=[8, 4],
            delay_pattern=[0, 1],
            empty_token_id=1,
        ),
        **kwargs,
    )


def processor():
    return MiMoAudioProcessor(Tokenizer(), config())


def test_config_parses_speech_fields_and_local_dimensions():
    c = config(speech_vocab_size="9-5", speech_zeroemb_idx="8-4", delay_pattern="0-1")
    assert c.speech_vocab_sizes == [9, 5]
    assert c.speech_empty_ids == [8, 4]
    assert c.delays == [0, 1]
    assert c.qwen_args().hidden_size == c.hidden_size
    assert c.qwen_args("input").num_hidden_layers == c.input_local_layers
    assert c.qwen_args("output").hidden_size == c.local_dim
    with pytest.raises(ValueError, match="audio_channels"):
        config(audio_channels=3)


def test_global_and_local_forward_with_synthetic_weights():
    c = config()
    model = Model(c)
    ids = processor().segments(["hi", {"codes": np.zeros((2, 8), dtype=np.int32)}])
    embeds = model._prepare_input_embeds(ids)
    output = model(ids)
    local = model.local_forward(output.local_hidden_states, Sampler(0))
    assert embeds.shape == (1, ids.shape[-1] // c.group_size, c.hidden_size)
    assert output.text_logits.shape == (1, 1, c.vocab_size)
    assert output.local_hidden_states.shape == (1, 1, c.local_dim)
    assert bool(mx.all(mx.isfinite(output.text_logits)))
    assert bool(mx.all(mx.isfinite(output.local_hidden_states)))
    assert local.shape == (1, c.group_size, c.audio_channels)
    assert bool(mx.all((local >= 0) & (local < mx.array(c.speech_empty_ids))))


def test_metadata_detection_does_not_claim_plain_qwen():
    for architecture in (True, False):
        source = (
            {"model_type": "qwen2", "architectures": ["MiMoAudioModel"]}
            if architecture
            else vars(config()) | {"model_type": "qwen2"}
        )
        assert model_type_from_config(source) == "mimo_audio"
        assert detect_model_domain(source, Path("arbitrarily-named")) == Domain.STS
        assert (
            get_model_type(source, Path("arbitrarily-named"), Domain.STS)
            == "mimo_audio"
        )
    assert model_type_from_config({"model_type": "qwen2"}) == "qwen2"
    assert model_type_from_config({"architectures": None}) is None
    assert classify_model("mimo_audio") == "sts"
    assert classify_model("mimo_audio_tokenizer") == "codec"


def test_sts_loader_resolves_original_config(monkeypatch):
    import mlx_audio.sts.utils as utils

    calls = []
    monkeypatch.setattr(
        utils,
        "load_config",
        lambda *a, **k: {"model_type": "qwen2", "architectures": ["MiMoAudioModel"]},
    )
    monkeypatch.setattr(
        utils, "base_load_model", lambda **kw: calls.append(kw) or "model"
    )
    assert utils.load("renamed-local-model") == "model"
    assert calls[0]["model_type"] == "mimo_audio"
    assert calls[0]["category"] == "sts"


def test_text_and_audio_packing_and_repeat_padding():
    p = processor()
    text = p.text("ab")
    np.testing.assert_array_equal(text[0, ::4], p.tokenizer.encode("ab"))
    assert np.all(text[0, 1::4] == -100)
    codes = np.array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 0]])
    packed = p.audio_codes(codes, boundaries=False)
    assert packed.shape == (3, 8)
    np.testing.assert_array_equal(packed[1:, -3:], np.repeat(codes[:, -1:], 3, axis=1))
    bounded = p.audio_codes(codes)
    assert bounded[0, 0] == p.ids["sosp"]
    assert bounded[0, -4] == p.ids["eosp"]
    with pytest.raises(ValueError):
        p.audio_codes(np.array([[8], [0]]))


def test_interleaving_keeps_all_text_and_audio():
    p = processor()
    audio = {"codes": np.zeros((2, 44), dtype=np.int32)}
    result = p.interleaved("abcdefghijk", audio)
    text = result[0, ::4]
    np.testing.assert_array_equal(text[text >= 16], p.tokenizer.encode("abcdefghijk"))
    assert np.count_nonzero(text == p.ids["empty"]) == 11
    assert np.count_nonzero(text == p.ids["eot"]) == 1
    assert text[0] == p.ids["sostm"] and text[-1] == p.ids["eostm"]


def test_prompts_are_deterministic_and_do_not_load_codec_for_text():
    p = processor()
    a = p.tts("Hello.")
    np.testing.assert_array_equal(np.asarray(a), np.asarray(p.tts("Hello.")))
    assert a[0, 0, -4].item() == p.ids["sostm"]
    p.dialogue([{"role": "user", "content": "Hello"}])
    assert p._audio_tokenizer is None
    for history in ([], [{"role": "assistant", "content": "Hello"}]):
        with pytest.raises(ValueError):
            p.dialogue(history)


def test_cached_global_forward_matches_full_prompt():
    p = processor()
    ids = p.segments(["hi", {"codes": np.zeros((2, 8), dtype=np.int32)}, "!"])
    model = Model(config())
    with mx.stream(mx.cpu):
        expected = model(ids).text_logits
        cache = model.make_cache()
        for start in range(0, ids.shape[-1], 4):
            actual = model(ids[:, :, start : start + 4], cache=cache).text_logits
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-5
        )
        assert all(c.offset == ids.shape[-1] // 4 for c in cache)


def test_speech_padding_embeddings_contribute_zero():
    model = Model(config())
    ids = mx.array(processor().text("hi"))[None]
    expected = model.model.embed_tokens(ids[:, 0, ::4])
    np.testing.assert_allclose(
        np.asarray(model._prepare_input_embeds(ids)), np.asarray(expected), atol=1e-6
    )


def test_local_delay_schedule_and_cache_reset():
    c = config(
        audio_channels=8,
        speech_vocab_size=[9] * 8,
        speech_zeroemb_idx=[8] * 8,
        delay_pattern=list(range(8)),
    )
    model = Model(c)
    calls = []

    class RecordingSampler:
        def __call__(self, logits, removed):
            calls.append(removed)
            return mx.zeros((1,), dtype=mx.int32)

    x = mx.zeros((1, 1, c.local_dim))
    tokens = model.local_forward(x, RecordingSampler())
    assert tokens.shape == (1, 4, 8)
    assert len(calls) == 32 and all(x == [8] for x in calls)
    first = model.local_forward(x, Sampler(0))
    second = model.local_forward(x, Sampler(0))
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert np.all(np.asarray(first) < 8)


def test_generation_switches_modality_and_retains_last_patch():
    class Stub:
        config = config()
        calls = 0
        speech_calls = 0

        def make_cache(self):
            return []

        def __call__(self, ids, cache):
            token = [1, 20, 9][self.calls]
            self.calls += 1
            logits = mx.full((1, 1, 64), -10.0)
            logits[..., token] = 10
            return ModelOutput(logits, mx.zeros((1, 1, 16)))

        def local_forward(self, hidden, sampler):
            self.speech_calls += 1
            return mx.zeros((1, 4, 2), dtype=mx.int32)

    model = Stub()
    tokens, reason = generate_tokens(
        model,
        mx.array(processor().text("x"))[None],
        max_new_tokens=5,
        global_sampler=Sampler(0),
        stop_tokens=[9],
    )
    assert reason == "stop" and model.speech_calls == 1
    np.testing.assert_array_equal(np.asarray(tokens[0, ::4]), [1, 20, 9])
    np.testing.assert_array_equal(
        np.asarray(tokens[1:, 4:8]), np.repeat([[8], [4]], 4, axis=1)
    )


def test_sampling_never_emits_padding_even_if_top_k_is_padding_only():
    sampler = Sampler(temperature=0.9, top_k=1)
    logits = mx.array([[0.0, 1.0, 100.0]])
    assert sampler(logits, [2]).item() in (0, 1)


def test_context_and_streaming_errors():
    model = Model(config(max_position_embeddings=2))
    prompt = mx.array(processor().text("long"))[None]
    with pytest.raises(ValueError, match="context"):
        generate_tokens(model, prompt)
    with pytest.raises(ValueError, match="streaming"):
        model.generate(stream=True)


def test_quantization_excludes_acoustic_modules():
    model = Model(config())
    assert model.model_quant_predicate("model.layers.0.mlp.up_proj", None)
    assert model.model_quant_predicate("lm_head", None)
    for name in (
        "local_transformer.layers.0.self_attn.q_proj",
        "speech_embeddings.0",
        "local_transformer_lm_heads.0",
    ):
        assert not model.model_quant_predicate(name, None)


@pytest.mark.parametrize(
    "text,prefilled,answer,reasoning",
    [
        ("Hello.", False, "Hello.", None),
        ("Use a happy voice.\n</think>\nHello.", True, "Hello.", "Use a happy voice."),
        (
            "<think>Consider the question.</think>42",
            False,
            "42",
            "Consider the question.",
        ),
        ("An unfinished thought", True, "", "An unfinished thought"),
    ],
)
def test_reasoning_is_separate_from_response(text, prefilled, answer, reasoning):
    assert split_reasoning(text, thinking_prefix=prefilled) == (answer, reasoning)


def test_conversion_records_codec_and_preserves_requested_dtype(tmp_path):
    from mlx_audio.convert import convert

    source, output = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    c = config()
    (source / "config.json").write_text(
        json.dumps(
            {
                **asdict(c),
                "model_type": "qwen2",
                "architectures": ["MiMoAudioModel"],
            }
        )
    )
    (source / "tokenizer.json").write_text("{}")
    (source / "unused.py").write_text("raise RuntimeError('not a runtime dependency')")
    model = Model(c)
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    convert(str(source), str(output), dtype="bfloat16")
    converted_config = json.loads((output / "config.json").read_text())
    assert converted_config["model_type"] == "mimo_audio"
    assert converted_config["audio_tokenizer_path"] == "XiaomiMiMo/MiMo-Audio-Tokenizer"
    assert converted_config["audio_tokenizer_revision"]
    assert (output / "tokenizer.json").exists()
    assert not (output / "unused.py").exists()
    assert "Streaming is not implemented" in (output / "README.md").read_text()
    weights = mx.load(str(output / "model.safetensors"))
    assert all(w.dtype == mx.bfloat16 for w in weights.values())
    restored = Model(ModelConfig.from_dict(converted_config))
    restored.load_weights(list(weights.items()), strict=True)


def test_upload_preserves_model_specific_card(tmp_path, monkeypatch):
    import huggingface_hub

    from mlx_audio.convert import upload_to_hub

    card = tmp_path / "README.md"
    card.write_text(
        "---\nlibrary_name: mlx-audio\n---\n\nA separate pinned audio tokenizer is required.\n"
    )
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(
            create_repo=lambda **kw: calls.append(kw),
            upload_folder=lambda **kw: calls.append(kw),
        ),
    )
    upload_to_hub(
        tmp_path, "test/model", "test/source", Domain.STS, model_card_path=card
    )
    assert "separate pinned audio tokenizer" in card.read_text()
    assert calls[-1]["repo_id"] == "test/model"


@pytest.mark.parametrize("speech", [False, True])
def test_cli_text_and_speech_outputs(tmp_path, monkeypatch, speech):
    import mlx_audio.sts.generate as cli
    import mlx_audio.utils as utils
    from mlx_audio import audio_io

    monkeypatch.setattr(
        utils,
        "load_config",
        lambda *a, **k: {
            "model_type": "qwen2",
            "architectures": ["MiMoAudioModel"],
        },
    )
    calls = []

    def generate(audio, **kwargs):
        calls.append((audio, kwargs))
        return SimpleNamespace(
            text="Hello.",
            audio=mx.zeros(240) if speech else None,
            sample_rate=24000,
            generation_tokens=1,
            processing_time=0,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        Model, "from_pretrained", lambda *a, **k: SimpleNamespace(generate=generate)
    )
    output = tmp_path / "new_directory" / "output.wav"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--model",
            "arbitrary-local-name",
            "--task",
            "tts" if speech else "text",
            "--text",
            "Hello.",
            "--output-path",
            str(output),
        ],
    )
    cli.main()
    assert calls[0][0] is None
    assert calls[0][1]["text"] == "Hello."
    assert output.with_suffix(".txt").read_text() == "Hello.\n"
    assert output.exists() == speech
    if speech:
        wave, sample_rate = audio_io.read(str(output))
        assert wave.shape == (240,) and sample_rate == 24000
