import json
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.registry import classify_model
from mlx_audio.tts.models.maya2 import Model, ModelConfig
from mlx_audio.tts.models.maya2.convert import (
    copy_supporting_files,
    prepare_config,
    write_model_card,
)
from mlx_audio.tts.models.maya2.maya2 import pack_prompt
from mlx_audio.tts.utils import get_model_and_args, load


def tiny_config():
    return ModelConfig.from_dict(
        {
            "backbone_config": {
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "vocab_size": 64,
                "max_position_embeddings": 256,
            },
            "depth_config": {
                "hidden_size": 16,
                "num_attention_heads": 4,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "embedding_rank": 4,
            },
            "codebook_size": 8,
            "speaker_anchor_hidden": 8,
        }
    )


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<|audio|>": 2, "<|lang:en|>": 3, "<|lang:hi|>": 4}[token]

    def encode(self, text, add_special_tokens=False):
        self.last_text = text
        return [5, 6]


def test_registration_and_source_config(tmp_path):
    from mlx_audio.convert import Domain, detect_model_domain, get_model_type

    for name in ("maya2", "maya3"):
        assert classify_model(name) == "tts"
        assert get_model_and_args(name, [])[1] == "maya2"
        assert detect_model_domain({"model_type": name}, tmp_path) == Domain.TTS
        assert get_model_type({"model_type": name}, tmp_path, Domain.TTS) == "maya2"
    (tmp_path / "backbone").mkdir()
    (tmp_path / "backbone/config.json").write_text(
        json.dumps(
            {
                "hidden_size": 16,
                "rope_parameters": {"rope_theta": 500000},
            }
        )
    )
    (tmp_path / "maya3_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "speaker_anchor": {"enabled": True, "hidden": 8, "input_dim": 192}
                },
            }
        )
    )
    config = prepare_config({"model_type": "maya3"}, tmp_path)
    assert config["model_type"] == "maya2"
    assert config["backbone_config"]["rope_theta"] == 500000
    assert config["backbone_config"]["hidden_size"] == 16
    assert config["speaker_anchor_hidden"] == 8
    assert "model_path" not in config


def test_named_prompt_uses_braces_and_no_bos():
    tok = Tokenizer()
    p = pack_prompt(tok, " Hello. ", "en", "Ahana")
    assert tok.last_text == "{Ahana}: Hello."
    assert p["tokens"] == [3, 5, 6, 2]
    assert p["sentinels"] == [3]


@pytest.mark.parametrize(
    "ref_lang,expected",
    [("en", [3, 5, 6, 2, 0, 0, 0, 5, 6, 2]), ("hi", [4, 5, 6, 2, 0, 0, 0, 3, 5, 6, 2])],
)
def test_reference_prompt_language_boundary(ref_lang, expected):
    codes = np.arange(64).reshape(32, 2) % 8
    p = pack_prompt(
        Tokenizer(),
        "Hello.",
        "en",
        ref_text="Reference.",
        ref_codes=codes,
        ref_lang=ref_lang,
    )
    assert p["tokens"] == expected
    assert p["kind"] == [0] * 4 + [1, 1, 3] + [0] * (len(expected) - 7)
    np.testing.assert_array_equal(np.array(p["codes"])[4:6], codes.T)


def test_strict_source_and_converted_load(tmp_path):
    from dataclasses import asdict

    model = Model(tiny_config())
    weights = dict(tree_flatten(model.parameters()))
    source = {"model." + k: v for k, v in weights.items()}
    source["model.backbone.lm.lm_head.weight"] = weights[
        "backbone.lm.model.embed_tokens.weight"
    ]
    assert set(model.sanitize(source)) == set(weights)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), source)
    (tmp_path / "config.json").write_text(json.dumps(asdict(model.config)))
    with patch.object(Model, "post_load_hook", side_effect=lambda model, path: model):
        restored = load(tmp_path)
    assert set(dict(tree_flatten(restored.parameters()))) == set(weights)
    assert model.model_quant_predicate("backbone.lm.model.layers.0.mlp.up_proj", None)
    assert not model.model_quant_predicate("depth.heads.0", None)
    assert not model.model_quant_predicate("audio_in.table", None)


def test_incremental_depth_matches_full_prefix():
    mx.random.seed(7)
    model = Model(tiny_config())
    hidden = mx.random.normal((1, 16))
    cb0 = mx.array([2])
    for memory in (model.depth.null_mem, mx.random.normal((1, 2, 16))):
        actual = model.depth.generate(hidden, cb0, memory, temperature=0)
        expected = [cb0]
        for level in range(1, 32):
            codes = mx.stack(expected, axis=1)
            from mlx_audio.tts.models.maya2.depth import GROUP_OF_LEVEL

            h = model.depth.run(
                model.depth.prefix(hidden, codes, level), GROUP_OF_LEVEL[level], memory
            )
            expected.append(mx.argmax(model.depth.heads[level - 1](h), axis=-1))
        np.testing.assert_array_equal(
            np.array(actual), np.array(mx.stack(expected, axis=1))
        )


def test_conversion_artifact_contains_runtime_files_and_licenses(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "converted"
    (source / "backbone").mkdir(parents=True)
    (source / "voices").mkdir()
    dest.mkdir()
    (source / "backbone/tokenizer.json").write_text("{}")
    (source / "voices/voices.json").write_text("{}")
    (source / "LICENSE").write_text("Apache-2.0")
    (source / "unused.yaml").write_text("unused")
    copy_supporting_files(source, dest)
    write_model_card(dest, "maya-research/maya2-native", "local/maya2", {})
    assert (dest / "backbone/tokenizer.json").is_file()
    assert not (dest / "unused.yaml").exists()
    card = (dest / "README.md").read_text()
    assert "CC-BY-4.0" in card and "Apache-2.0" in card
    assert (dest / "LICENSE").read_text() == "Apache-2.0"


class Codec:
    def __init__(self):
        self.decoder = self
        self.upsample = self
        self.decoder_cache = []

    def reset_state(self):
        self.offset = 0

    def decode_step(self, codes):
        self.offset += 1
        return mx.full((1, 1, 1920), self.offset / 10)


class ScriptedModel(Model):
    def __call__(self, embeddings, cache=None):
        count = getattr(self, "_steps", 0)
        self._steps = count + 1
        eos = self.config.codebook_size
        logits = mx.full((1, 1, eos + 1), -100.0)
        logits[:, :, eos if count >= 6 else 1] = 100
        return mx.zeros((1, 1, 16)), logits


def test_streaming_matches_offline_and_resets_on_cancellation():
    model = ScriptedModel(tiny_config())
    model._tokenizer, model._codec = Tokenizer(), Codec()
    with patch.object(
        model.depth, "generate", return_value=mx.ones((1, 32), dtype=mx.int32)
    ):
        offline = list(model.generate("Hello.", max_tokens=10, temperature=0))
        assert offline[0].prompt["tokens-per-sec"] > 0
        model._steps = 0
        chunks = list(
            model.generate(
                "Hello.",
                max_tokens=10,
                temperature=0,
                stream=True,
                streaming_interval=0.16,
            )
        )
        assert len(chunks) == 3
        assert [c.is_final_chunk for c in chunks] == [False, False, True]
        np.testing.assert_array_equal(
            np.array(mx.concatenate([c.audio for c in chunks])),
            np.array(offline[0].audio),
        )
        assert model._codec.offset == 0
        model._steps = 0
        generator = model.generate(
            "Hello.", max_tokens=10, stream=True, streaming_interval=0.08
        )
        next(generator)
        generator.close()
        assert model._codec.offset == 0


def test_generation_normalizes_only_target_text_and_can_bypass():
    model = ScriptedModel(tiny_config())
    model._tokenizer, model._codec = Tokenizer(), Codec()
    with (
        patch.object(
            model.depth, "generate", return_value=mx.ones((1, 32), dtype=mx.int32)
        ),
        patch.object(
            model, "_reference", return_value=(np.zeros((32, 2), dtype=np.int32), None)
        ),
        patch(
            "mlx_audio.tts.models.maya2.maya2.pack_prompt", wraps=pack_prompt
        ) as pack,
    ):
        list(
            model.generate(
                "I have 2 cats.",
                max_tokens=1,
                ref_audio="unused.wav",
                ref_text="I have 3 cats.",
            )
        )
        assert pack.call_args.args[1] == "I have two cats."
        assert pack.call_args.args[4] == "I have 3 cats."
        list(model.generate("I have 2 cats.", max_tokens=1, normalize=False))
        assert pack.call_args.args[1] == "I have 2 cats."


def test_reference_audio_uses_upstream_resampling():
    from scipy.signal import resample_poly

    model = Model(tiny_config())
    model._codec = MagicMock()
    model._codec.encode.return_value = mx.zeros((1, 32, 3), dtype=mx.int32)
    model._speaker_encoder = MagicMock()
    model._speaker_encoder.encode.return_value = mx.ones((192,))
    audio = np.random.default_rng(7).normal(0, 0.1, (9600, 2)).astype(np.float32)
    with patch(
        "mlx_audio.tts.models.maya2.maya2.audio_read", return_value=(audio, 48000)
    ):
        codes, refs = model._reference("reference.wav", "icl_spk")
    expected24 = resample_poly(audio.mean(axis=1), 1, 2)
    expected16 = resample_poly(expected24, 2, 3)
    np.testing.assert_array_equal(
        np.asarray(model._codec.encode.call_args.args[0])[0, 0], expected24
    )
    np.testing.assert_array_equal(
        model._speaker_encoder.encode.call_args.args[0], expected16
    )
    assert codes.shape == (32, 3) and refs.shape == (1, 1, 192)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voice": "unknown"},
        {"lang_code": "fr"},
        {"ref_audio": mx.zeros(2400)},
        {"voice": "Ahana", "ref_audio": mx.zeros(2400)},
        {"top_p": 0},
        {"max_tokens": 0},
        {"speed": 2},
        {"temperature": -1},
    ],
)
def test_generation_rejects_invalid_requests(kwargs):
    model = Model(tiny_config())
    model._tokenizer = Tokenizer()
    with pytest.raises(ValueError):
        next(model.generate("Hello.", **kwargs))
