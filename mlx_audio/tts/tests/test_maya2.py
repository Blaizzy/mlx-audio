"""Tests for the Maya-2 TTS model. All but the last need no weights or network (tiny random configs)."""

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.tts.models.maya2 import Model, ModelConfig
from mlx_audio.tts.models.maya2 import guards as G
from mlx_audio.tts.models.maya2.depth import GROUP_OF_LEVEL
from mlx_audio.tts.models.maya2.prompt import (
    ADDED_TOKENS,
    KIND_AEOS,
    KIND_AUDIO,
    KIND_TEXT,
    N_Q,
    Prompter,
    Voice,
    extend_tokenizer,
    load_frames_prior,
    load_named_voices,
    split_text,
)

N_CODES = 2048


def tiny_config(**overrides):
    kw = dict(
        backbone=dict(
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            vocab_size=300,
            max_position_embeddings=512,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        ),
        depth_dim=16,
        depth_heads=2,
        depth_ffn=32,
        depth_rank=4,
        anchor_hidden=8,
    )
    kw.update(overrides)
    return ModelConfig(**kw)


def tiny_model():
    mx.random.seed(0)
    model = Model(tiny_config())
    # random (not zero) values everywhere, so the numeric tests are meaningful
    model.update(
        nn.utils.tree_map(
            lambda p: mx.random.normal(p.shape) * 0.05, model.parameters()
        )
    )
    mx.eval(model.parameters())
    return model


class FakeTokenizer:
    """Stands in for the HF tokenizer: one id per character, special tokens appended after the base vocab."""

    def __init__(self):
        self.vocab = {}

    def add_tokens(self, tokens, special_tokens=True):
        for t in tokens:
            self.vocab.setdefault(t, 1000 + len(self.vocab))

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 500 for c in text]}


def prompter():
    tok, ids = extend_tokenizer(FakeTokenizer())
    return Prompter(tok, ids), ids


# ----------------------------------------------------------------------------- config / weights
def test_config_from_dict_ignores_unknown_keys():
    cfg = ModelConfig.from_dict(
        {
            "model_type": "maya2",
            "backbone": {"hidden_size": 32},
            "n_q": 32,
            "unknown": 1,
        }
    )
    assert (
        cfg.backbone == {"hidden_size": 32}
        and cfg.n_q == 32
        and not hasattr(cfg, "unknown")
    )


def test_sanitize_maps_official_checkpoint_keys_and_loads_strictly():
    model = tiny_model()
    params = dict(tree_flatten(model.parameters()))
    official = {}
    for k, v in params.items():  # the release's key layout
        if k.startswith("backbone."):
            k = "backbone.lm.model." + k[len("backbone.") :]
        elif k == "cb0_head.weight":
            k = "backbone.cb0_head.weight"
        official[k] = v
    official["backbone.lm.lm_head.weight"] = params[
        "backbone.embed_tokens.weight"
    ]  # tied alias, dropped
    sanitized = model.sanitize(official)
    assert set(sanitized) == set(params)
    model.load_weights(list(sanitized.items()), strict=True)
    assert model.sanitize(params).keys() == params.keys()  # converted keys pass through


def test_quant_predicate_quantizes_backbone_projections_only():
    model = tiny_model()
    nn.quantize(
        model,
        group_size=32,
        bits=8,
        class_predicate=lambda p, m: hasattr(m, "to_quantized")
        and m.weight.shape[-1] % 32 == 0
        and model.model_quant_predicate(p, m),
    )
    quantized = {
        p
        for p, m in model.named_modules()
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    }
    assert quantized and all(p.startswith("backbone.layers.") for p in quantized)
    assert not isinstance(model.backbone.embed_tokens, nn.QuantizedEmbedding)


# ----------------------------------------------------------------------------- depth decoder
def test_depth_kv_reuse_matches_full_recompute_per_level():
    """The serving path (shared-group K/V reuse, last-layer last-position only) equals recomputing each level's full
    prefix through its group, teacher-forced on the same codes."""
    model = tiny_model()
    d = model.depth
    d.to_serving_dtype(mx.float32)
    h = mx.random.normal((1, 32))
    codes = mx.array(np.random.default_rng(1).integers(0, N_CODES, (1, N_Q)), mx.int32)
    cross = d.speaker_constants(np.ones(192, np.float32) / np.sqrt(192))
    _, fast = d.frame(h, codes[:, 0], cross, None, temp=1.0, force=codes)
    table = d.prefix_table()
    prefix = [d.in_proj(h) + d.level_emb.weight[0]] + [
        table[codes[:, k] + k * N_CODES] for k in range(N_Q - 1)
    ]
    for level in range(1, N_Q):
        x = mx.stack(prefix[: level + 1], 1)
        for li, block in enumerate(d.groups[GROUP_OF_LEVEL[level]]):
            x, _ = block(x, causal=True)
            x = x + cross[li]
        ref = d.heads[level - 1](d.norm(x[:, -1]))
        assert mx.allclose(fast[level - 1], ref, atol=1e-4).item(), f"level {level}"


def test_depth_greedy_codes_are_valid_and_seeded_sampling_is_reproducible():
    model = tiny_model()
    model.depth.to_serving_dtype(mx.float32)
    h, cb0, cross = (
        mx.random.normal((1, 32)),
        mx.array([5]),
        model.depth.speaker_constants(None),
    )
    greedy, _ = model.depth.frame(h, cb0, cross, None)
    assert greedy.shape == (1, N_Q) and int(greedy[0, 0]) == 5
    a, _ = model.depth.frame(h, cb0, cross, mx.random.key(3))
    b, _ = model.depth.frame(h, cb0, cross, mx.random.key(3))
    assert mx.array_equal(a, b).item() and int(mx.max(a)) < N_CODES


# ----------------------------------------------------------------------------- prompts
def test_extend_tokenizer_appends_contract_in_order_and_is_idempotent():
    tok, ids = extend_tokenizer(FakeTokenizer())
    assert list(ids) == ADDED_TOKENS and ids[
        "<|lang:ta|>"
    ] == 1000 + ADDED_TOKENS.index("<|lang:ta|>")
    assert extend_tokenizer(tok)[1] == ids


def test_named_voice_prompt_layout():
    p, ids = prompter()
    voice = load_named_voices({"voices": [{"name": "Kayal", "gender": "female"}]})[
        "kayal"
    ]
    prompt = p.build(voice, "hi", "ta")
    text = "{Kayal}: hi"
    assert prompt.tokens.tolist() == [
        ids["<|lang:ta|>"],
        *[ord(c) % 500 for c in text],
        ids["<|audio|>"],
    ]
    assert (prompt.kind == KIND_TEXT).all() and prompt.sentinels == [
        len(prompt.tokens) - 1
    ]
    assert prompt.prefix_len == 0 and (prompt.codes == -1).all()


def test_cloned_voice_prompt_has_reference_span_and_crosslingual_tag():
    p, ids = prompter()
    codes = np.arange(N_Q * 3).reshape(N_Q, 3) % N_CODES
    voice = Voice(
        "me",
        "icl_spk",
        clips=[{"text": "ab", "lang": "en", "codes": codes}],
        ecapa=np.ones(192),
    )
    prompt = p.build(voice, "c", "hi")
    # reference span: <|lang:en|> a b <|audio|> 3 frames AEOS ; target: <|lang:hi|> c <|audio|> (cut here)
    assert prompt.prefix_len == 8
    assert (
        prompt.tokens[0] == ids["<|lang:en|>"]
        and prompt.tokens[8] == ids["<|lang:hi|>"]
    )
    assert prompt.kind[4:7].tolist() == [KIND_AUDIO] * 3 and prompt.kind[7] == KIND_AEOS
    assert np.array_equal(prompt.codes[4:7], codes.T)
    assert prompt.sentinels == [3, 10] and len(prompt.tokens) == 11
    same_lang = p.build(
        voice, "c", "en"
    )  # same language: the target span carries no tag
    assert same_lang.tokens[8] == ord("c") % 500 and same_lang.sentinels == [3, 9]


def test_compose_adds_anchor_only_at_sentinels_and_frames_at_audio():
    model = tiny_model()
    p, _ = prompter()
    codes = np.zeros((N_Q, 2), np.int64)
    voice = Voice(
        "me",
        "icl_spk",
        clips=[{"text": "ab", "lang": "en", "codes": codes}],
        ecapa=np.ones(192) / 14,
    )
    prompt = p.build(voice, "c", "en")
    emb, base = model._compose(prompt, voice), model.backbone.embed_tokens(
        mx.array(prompt.tokens)
    )
    anchor = model.speaker_anchor(voice.ecapa)
    for c in range(len(prompt.tokens)):
        if c in prompt.sentinels:
            assert mx.allclose(emb[c], base[c] + anchor, atol=1e-5).item()
        elif prompt.kind[c] == KIND_AUDIO:
            assert mx.allclose(
                emb[c], model.audio_in.frames(mx.array(codes[:, :1].T))[0], atol=1e-5
            ).item()
        elif prompt.kind[c] == KIND_AEOS:
            assert mx.allclose(emb[c], model.audio_in.aeos(), atol=1e-5).item()
        else:
            assert mx.array_equal(emb[c], base[c]).item()


def test_voice_validation():
    with pytest.raises(ValueError):
        Voice("x", "icl_spk")  # needs a clip and an ECAPA embedding
    with pytest.raises(ValueError):
        Voice("x", "nope")


# ----------------------------------------------------------------------------- serving controls
def test_controlled_stop_rules_and_aeos_never_sampled():
    model = tiny_model()
    aeos = N_CODES

    def run(aeos_logit, frames, minimum=4, hazard=0.0, top_code_logit=0.0):
        logits = np.zeros((1, aeos + 1), np.float32)
        logits[0, 0], logits[0, aeos] = top_code_logit, aeos_logit
        # set the head so that cb0_head(h) returns exactly these logits for h = e_0
        model.cb0_head.weight = (
            mx.zeros_like(model.cb0_head.weight).at[:, 0].add(mx.array(logits[0]))
        )
        h = mx.zeros((1, 32)).at[0, 0].add(1.0)
        st = {
            "temp": 1.0,
            "top_k": aeos + 1,
            "top_p": 1.0,
            "rep": 0.0,
            "bias": 0.0,
            "recent": [],
            "frames": frames,
            "minimum": minimum,
            "hazard": mx.array([hazard]),
        }
        code, cb0, hz = model._cb0(h, st, mx.random.key(0))
        return int(code.item()), int(cb0.item()), float(hz.item())

    assert run(50.0, frames=10)[0] == 2  # argmax is AEOS
    assert run(50.0, frames=2)[0] == 0  # before the minimum length nothing stops
    code, cb0, hz = run(
        9.8, frames=10, top_code_logit=10.0
    )  # code 0 is argmax, p(AEOS) = 0.43 >= tau 0.35
    assert code == 3 and cb0 < aeos and 0.35 <= hz < 0.6
    code, cb0, hz = run(0.0, frames=10, hazard=0.7)  # accumulated hazard >= 0.6
    assert code == 4 and hz >= 0.6 and cb0 < aeos


def test_resolve_lang_prefers_script_unless_codes_share_it():
    r = Model.resolve_lang
    assert (
        r(None, "en", "வணக்கம்!") == "ta"
    )  # the CLI's default "en" must not override Tamil script
    assert r(None, "mr", "नमस्कार") == "mr" and r(None, "en", "नमस्कार") == "hi"
    assert (
        r(None, "as", "মই ভালে আছোঁ") == "as"
        and r(None, "tamil", "Tanglish la pesalam") == "ta"
    )
    assert r(None, None, "Hello") == "en" and r(None, "xx", "Hello") == "en"


def test_split_text_respects_budget_and_keeps_terminals():
    prior = load_frames_prior(
        {"en": {"affine_a": 0.0, "affine_b": 1.0, "resid_p99": 10.0}}
    )
    text = "One two three. Four five six. Seven eight nine."
    assert split_text(text, "en", prior, 100.0) == [text]
    pieces = split_text(text, "en", prior, 2.4)  # 30 frames ~ 30 characters per piece
    assert (
        len(pieces) > 1
        and " ".join(pieces) == text
        and all(p.endswith(".") for p in pieces)
    )


def test_loop_guard_flags_stall_and_period_but_not_varied_codes():
    g = G.GuardConfig()
    stall = np.full((1, g.hist), 7)
    period = np.tile(np.arange(4), g.hist // 4)[None]
    varied = np.random.default_rng(0).integers(0, N_CODES, (1, g.hist))
    hits = G.loop_detect_batch(
        np.concatenate([stall, period, varied]), np.array([g.hist] * 3), g
    )
    assert hits[0] is not None and hits[1] is not None and hits[2] is None


def test_vendored_normaliser_expands_numbers():
    from mlx_audio.tts.models.maya2.text import normalise_text

    assert normalise_text("I have 2 cats.", "en") == "I have two cats."
    assert "ஐந்நூறு" in normalise_text("₹500 கொண்டு வா.", "ta")


# ----------------------------------------------------------------------------- real weights
@pytest.mark.requires_weights
@pytest.mark.skipif(
    not os.environ.get("MAYA2_MLX_PATH"),
    reason="set MAYA2_MLX_PATH to a converted model",
)
def test_generate_with_real_weights():
    from mlx_audio.tts.utils import load_model

    model = load_model(os.environ["MAYA2_MLX_PATH"])
    results = list(model.generate("வணக்கம்!", voice="Kayal", seed=1))
    audio = np.array(results[-1].audio)
    assert (
        results[-1].sample_rate == 24000
        and 0.3 < len(audio) / 24000 < 5
        and np.isfinite(audio).all()
    )
    assert np.abs(audio).max() > 0.01
