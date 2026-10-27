"""Weight-free checks for pooling, exact verification, and cache rollback."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.lm.generate import generate_step
from mlx_audio.stt.models.granite_speech import copy_supporting_files
from mlx_audio.stt.models.granite_speech.granite_speech import Model
from mlx_audio.stt.models.granite_speech.speculative import (
    DraftLookup,
    collapse_bpe_tokens,
    posterior_weighted_pool,
    speculative_generate_step,
)
from mlx_audio.stt.tests.test_granite_speech_5_2b import tiny_config


def test_posterior_pooling_pads_with_zero_importance():
    hidden = mx.array([[[1.0], [3.0], [9.0], [11.0], [7.0]]])
    importance = mx.array([[1.0, 3.0, 0.0, 0.0, 0.5]])
    pooled = posterior_weighted_pool(hidden, importance)
    np.testing.assert_allclose(pooled, [[[2.5], [7.0]]])
    np.testing.assert_array_equal(
        posterior_weighted_pool(hidden, mx.zeros_like(importance)),
        mx.zeros((1, 2, 1)),
    )


def test_bpe_ctc_blank_resets_repetition_and_labels_are_shifted():
    assert collapse_bpe_tokens([0, 3, 3, 0, 3, 5, 5, 0]) == [2, 2, 4]


def test_lookup_recovers_after_punctuation_insertion():
    lookup = DraftLookup([1, 2, 3, 4, 5, 6])
    assert lookup.propose([1, 2], 3) == [3, 4, 5]
    lookup.cursor += 1
    assert lookup.propose([1, 2, 3, 99], 3) == []
    assert lookup.propose([1, 2, 3, 99, 4], 3) == [5, 6]


class ToyCache:
    def __init__(self):
        self.tokens = []

    @property
    def state(self):
        return mx.array(self.tokens)

    def trim(self, count):
        if count:
            del self.tokens[-count:]


class ToyModel:
    """Any untrimmed rejected token poisons every subsequent prediction."""

    _tokenizer = SimpleNamespace(eos_token_id=0)

    def __init__(self, target):
        self.target = target
        self.cache = None

    def make_cache(self):
        self.cache = ToyCache()
        return [self.cache]

    def __call__(self, ids, cache, input_embeddings=None):
        predictions = []
        for token in ids[0].tolist():
            cache[0].tokens.append(token)
            history = cache[0].tokens
            prefix = [30, 31] + self.target
            if history == prefix[: len(history)]:
                next_token = prefix[len(history)] if len(history) < len(prefix) else 0
            else:
                next_token = 29
            predictions.append(next_token)
        return mx.where(
            mx.arange(32)[None, None, :] == mx.array(predictions)[None, :, None],
            10.0,
            -10.0,
        )


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize(
    "draft",
    [
        [],
        [1, 2, 3, 4, 5, 6],
        [9, 2, 3, 4],
        [1, 2, 9, 9, 4, 5, 6],
        [1, 2],
        [1, 2, 3, 4, 5, 6, 0],
    ],
)
def test_verification_returns_target_with_rejected_cache_suffix_removed(
    draft, block_size
):
    target = [1, 2, 3, 4, 5, 6, 0]
    model = ToyModel(target)
    stats = {}
    result = list(
        speculative_generate_step(
            mx.array([30, 31]),
            model,
            input_embeddings=mx.zeros((2, 4)),
            draft_tokens=draft,
            max_tokens=20,
            num_draft_tokens=block_size,
            prefill_step_size=1,
            stats=stats,
        )
    )
    assert [token for token, _ in result] == target
    assert stats["accepted_tokens"] <= stats["proposed_tokens"]
    assert 29 not in model.cache.tokens


@pytest.mark.parametrize("limit", [0, 1, 3, 6])
def test_max_tokens_is_honored_inside_accepted_block(limit):
    result = list(
        speculative_generate_step(
            mx.array([30, 31]),
            ToyModel([1, 2, 3, 4, 5, 6, 0]),
            input_embeddings=mx.zeros((2, 4)),
            draft_tokens=[1, 2, 3, 4, 5, 6],
            max_tokens=limit,
        )
    )
    assert [token for token, _ in result] == [1, 2, 3, 4, 5, 6][:limit]


@pytest.mark.parametrize("prompt_length", [3, 250, 254])
def test_real_transformer_verification_matches_sequential_decoding(prompt_length):
    mx.random.seed(7)
    model = Model(tiny_config())
    model._tokenizer = SimpleNamespace(eos_token_id=-1)
    model.eval()
    prompt = (mx.arange(prompt_length) + 1) % 32
    embeds = model.language_model.model.embed_tokens(prompt)
    baseline = [
        token
        for token, _ in generate_step(
            prompt, model, input_embeddings=embeds, max_tokens=16
        )
    ]
    draft = baseline[:6] + [(baseline[6] + 1) % 32] + baseline[7:]
    stats = {}
    actual = [
        token
        for token, _ in speculative_generate_step(
            prompt,
            model,
            input_embeddings=embeds,
            draft_tokens=draft,
            max_tokens=16,
            num_draft_tokens=4,
            stats=stats,
        )
    ]
    assert actual == baseline
    assert stats["accepted_tokens"] > 0


def test_encoder_importance_uses_midpoint_before_self_conditioning():
    model = Model(tiny_config())
    features = mx.random.normal((1, 15, 160))
    ordinary = model.encoder(features)
    hidden, importance = model.encoder(features, return_importance=True)
    mid = model.encoder.layers[0](
        model.encoder.input_linear(features), model.encoder._attention_dists
    )
    expected = 1 - mx.softmax(model.encoder.out(mid), axis=-1)[:, :, 0]
    np.testing.assert_allclose(importance, expected, atol=1e-6)
    np.testing.assert_allclose(hidden, ordinary, atol=1e-6)


def test_optional_bpe_head_loads_real_weights_lazily(tmp_path):
    model = Model(tiny_config())
    assert not model.supports_speculative
    with pytest.raises(ValueError, match="requires out_llm.safetensors"):
        model._load_bpe_head()
    weights = {"weight": mx.ones((32, 8)), "bias": mx.ones((32,))}
    path = tmp_path / "out_llm.safetensors"
    mx.save_safetensors(str(path), weights)
    model._bpe_head_path = path
    assert model.supports_speculative
    head = model._load_bpe_head()
    np.testing.assert_array_equal(head(mx.ones((1, 8))), mx.full((1, 32), 9.0))
    assert model._load_bpe_head() is head


def test_conversion_removes_stale_bpe_head_when_source_has_none(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (dest / "out_llm.safetensors").write_text("stale head")
    copy_supporting_files(source, dest)
    assert not (dest / "out_llm.safetensors").exists()


def test_public_generate_and_stream_report_speculative_stats():
    model = ToyModel([1, 2, 3, 0])
    model._tokenizer = SimpleNamespace(
        eos_token_id=0, decode=lambda tokens, **kwargs: "".join(map(str, tokens))
    )
    model.is_plus = False
    model._load_bpe_head = lambda: None
    model._load_audio = lambda audio: audio
    model._extract_features = lambda audio: (mx.zeros((1, 1, 160)), 2)
    model._get_speculative_audio_features = lambda features, stats: (
        mx.zeros((1, 2, 4)),
        [1, 2, 3],
    )
    model._build_prompt = lambda *args, **kwargs: mx.array([30, 31])
    model._build_inputs_embeds = lambda *args, **kwargs: mx.zeros((1, 2, 4))
    output = Model.generate(model, mx.zeros((1,)), speculative=True)
    chunks = list(Model._stream_generate(model, mx.zeros((1,)), speculative=True))
    assert output.text == "".join(chunk.text for chunk in chunks) == "123"
    assert output.generation_tokens == chunks[-1].generation_tokens == 3
    assert output.speculative_stats["accepted_tokens"] == 2
    assert chunks[-1].speculative_stats["accepted_tokens"] == 2
    assert chunks[-1].is_final


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": 0.5}, {"repetition_penalty": 1.1}, {"num_draft_tokens": 0}],
)
def test_unsupported_sampling_options_fail_before_audio_work(kwargs):
    with pytest.raises(ValueError):
        Model(tiny_config()).generate("unused.wav", speculative=True, **kwargs)
