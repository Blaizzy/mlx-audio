"""Unit tests for the rumik-oss 1 TTS model that need no weights or network."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_audio.tts.models.rumik_oss import Model, ModelConfig

# A tiny config with the real audio-vocabulary layout scaled down:
# 4 quantizers x 8 codes = 32 unit tokens, then <text>, <audio>, </audio>.
FIRST_UNIT = 40
NUM_Q = 4
CODEBOOK = 8
LAST_UNIT = FIRST_UNIT + NUM_Q * CODEBOOK - 1  # 71
TEXT_START, AUDIO_START, AUDIO_END = 72, 73, 74
VOCAB = 80


def tiny_config(**overrides):
    kwargs = dict(
        hidden_size=16,
        head_dim=8,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=VOCAB,
        sliding_window=4,
        sliding_window_pattern=4,
        logit_scale=1.0,
        num_quantizers=NUM_Q,
        codebook_size=CODEBOOK,
        first_unit_id=FIRST_UNIT,
        text_start_token_id=TEXT_START,
        audio_start_token_id=AUDIO_START,
        audio_end_token_id=AUDIO_END,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def unit_id(code, q):
    return FIRST_UNIT + code * NUM_Q + q


def test_config_from_dict_derives_last_unit_id_and_ignores_unknown_keys():
    raw = {
        "model_type": "rumik_oss",
        "hidden_size": 16,
        "head_dim": 8,
        "num_hidden_layers": 4,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": VOCAB,
        "first_unit_id": FIRST_UNIT,
        "num_quantizers": NUM_Q,
        "codebook_size": CODEBOOK,
        "audio_end_token_id": AUDIO_END,
        "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
        "torch_dtype": "bfloat16",
        "auto_map": {},
    }
    cfg = ModelConfig.from_dict(raw)
    assert cfg.last_unit_id == LAST_UNIT
    assert cfg.model_type == "rumik_oss"
    assert cfg.speakers == ["Ira", "Aisha", "Siya", "Zoya"]


def test_stop_predictor_shapes_and_quant_exclusion():
    model = Model(tiny_config())
    shapes = dict(tree_flatten(model.parameters()))
    # LayerNorm(16) -> Linear(16, 64) -> GELU -> Linear(64, 1)
    assert shapes["stop_predictor.layers.0.weight"].shape == (16,)
    assert shapes["stop_predictor.layers.1.weight"].shape == (64, 16)
    assert shapes["stop_predictor.layers.3.weight"].shape == (1, 64)
    # No lm_head: embeddings are tied.
    assert not any(k.startswith("lm_head") for k in shapes)
    assert model.model_quant_predicate("stop_predictor.layers.1", None) is False
    assert model.model_quant_predicate("model.layers.0.mlp.up_proj", None) is True


def test_sanitize_maps_pytorch_sequential_keys():
    model = Model(tiny_config())
    weights = {
        "stop_predictor.0.weight": mx.ones((16,)),
        "stop_predictor.0.bias": mx.zeros((16,)),
        "stop_predictor.1.weight": mx.ones((64, 16)),
        "stop_predictor.1.bias": mx.zeros((64,)),
        "stop_predictor.3.weight": mx.ones((1, 64)),
        "stop_predictor.3.bias": mx.zeros((1,)),
        "model.layers.0.self_attn.rotary_emb.inv_freq": mx.ones((4,)),
        "lm_head.weight": mx.ones((VOCAB, 16)),
        "model.embed_tokens.weight": mx.ones((VOCAB, 16)),
    }
    out = model.sanitize(weights)
    assert "stop_predictor.layers.3.bias" in out
    assert "stop_predictor.3.bias" not in out
    assert "lm_head.weight" not in out
    assert not any("inv_freq" in k for k in out)
    assert "model.embed_tokens.weight" in out


def test_forward_returns_logits_and_stashes_last_hidden():
    model = Model(tiny_config())
    cache = model.make_cache()
    logits = model(mx.array([[1, 2, 3]]), cache=cache)
    assert logits.shape == (1, 3, VOCAB)
    assert model._last_hidden.shape == (1, 16)
    p = model.stop_probability(model._last_hidden)
    assert p.shape == (1, 1)
    assert 0.0 <= p.item() <= 1.0


def test_audio_tokens_to_codes_matches_reference_semantics():
    model = Model(tiny_config())
    frame_a = [unit_id(c, q) for q, c in enumerate([1, 2, 3, 4])]
    frame_b = [unit_id(c, q) for q, c in enumerate([5, 6, 7, 0])]
    # A stray non-unit token mid-frame drops that partial frame and resyncs.
    broken = [unit_id(1, 0), unit_id(1, 1), TEXT_START]
    # An off-round-robin q=0 token starts a fresh frame.
    resync = [unit_id(2, 0), unit_id(2, 1), unit_id(9 % CODEBOOK, 0)] + [
        unit_id(3, q) for q in (1, 2, 3)
    ]
    tokens = frame_a + broken + frame_b + resync + [AUDIO_END, unit_id(0, 0)]
    codes = model.audio_tokens_to_codes(tokens)
    assert codes.shape == (1, NUM_Q, 3)
    assert codes[0, :, 0].tolist() == [1, 2, 3, 4]
    assert codes[0, :, 1].tolist() == [5, 6, 7, 0]
    assert codes[0, :, 2].tolist() == [1, 3, 3, 3]


def test_audio_tokens_to_codes_requires_a_full_frame():
    model = Model(tiny_config())
    with pytest.raises(ValueError):
        model.audio_tokens_to_codes([unit_id(0, 0), unit_id(0, 1)])


def test_logits_processor_masks_vocab_and_min_tokens():
    model = Model(tiny_config())
    model._last_hidden = mx.zeros((1, 16))
    proc = model.make_audio_logits_processor(min_tokens=2)
    logits = mx.zeros((1, VOCAB))
    tokens = mx.array([0])

    # Steps 0 and 1: units allowed, </audio> and text tokens masked.
    out = proc(tokens, logits)
    assert mx.isfinite(out[0, FIRST_UNIT]).item()
    assert mx.isfinite(out[0, LAST_UNIT]).item()
    assert out[0, AUDIO_END].item() == -float("inf")
    assert out[0, TEXT_START].item() == -float("inf")
    assert out[0, 0].item() == -float("inf")
    proc(tokens, logits)

    # Step 2 onwards: </audio> becomes legal (stop head untrained -> not forced
    # unless it fires; check the allowed set).
    out = proc(tokens, logits)
    assert mx.isfinite(out[0, AUDIO_END]).item()
    assert out[0, TEXT_START].item() == -float("inf")


def test_logits_processor_forces_audio_end_when_stop_fires():
    model = Model(tiny_config())
    # Bias the final linear so sigmoid(...) > 0.5 for any input.
    final = model.stop_predictor.layers[3]
    final.weight = mx.zeros_like(final.weight)
    final.bias = mx.array([10.0])
    model._last_hidden = mx.zeros((1, 16))
    proc = model.make_audio_logits_processor(min_tokens=0)
    out = proc(mx.array([0]), mx.zeros((1, VOCAB)))
    assert mx.argmax(out, axis=-1).item() == AUDIO_END
    assert out[0, FIRST_UNIT].item() == -float("inf")


def test_prompt_format_and_voice_resolution():
    model = Model(tiny_config())
    assert model.resolve_voice(None) == "Ira"
    assert model.resolve_voice("zoya") == "Zoya"
    with pytest.raises(ValueError):
        model.resolve_voice("Bob")
    p = model.build_prompt("नमस्ते", "Aisha", "happy, Hindi accent, steady pace")
    assert p == '<text>Aisha: <description="happy, Hindi accent, steady pace"> नमस्ते<audio>'
    # An inline description in the text is kept as-is.
    p2 = model.build_prompt('<description="sad"> hi', None, "happy")
    assert p2 == '<text>Ira: <description="sad"> hi<audio>'
