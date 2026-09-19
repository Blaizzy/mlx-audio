"""Depth-cache regressions for Breeze TTS 2.

The depth decoder used to re-run its full stack over the growing prefix at every
one of the 15 codebook steps of a frame. It now walks one position per step
against a per-frame KV cache, which is ~2.7x faster end to end; these tests pin
the arithmetic that made the change safe. The reference walk below keeps the
original prefix-recompute form, so both live in the same test file.
"""

import pytest

try:
    import mlx.core as mx
except (ImportError, RuntimeError) as exc:  # pragma: no cover - CI without Metal
    pytest.skip(f"MLX device unavailable: {exc}", allow_module_level=True)

from mlx_audio.tts.models.breeze_tts.breeze_tts import Model
from mlx_audio.tts.models.breeze_tts.config import ModelConfig


def _tiny_config(**overrides):
    values = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_codebooks=4,
        vocab_size=8,
        text_vocab_size=32,
        text_encoder_config={
            "hidden_size": 12,
            "num_hidden_layers": 1,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 6,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "layer_types": ["full_attention"],
        },
        depth_decoder_config={
            "hidden_size": 12,
            "num_hidden_layers": 1,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 6,
            "rms_norm_eps": 1e-5,
            "num_codebooks": 4,
            "vocab_size": 8,
            "audio_embed_size": 16,
        },
    )
    values.update(overrides)
    return ModelConfig(**values)


def _randomized_model(seed: int = 0) -> Model:
    """A tiny model with a randomised output head.

    ``_CodebooksHead`` is initialised to zeros, so a fixture that leaves it alone
    produces all-zero logits: every comparison would pass and every greedy pick
    would return the same id, which is a test that cannot fail. Randomising the
    head is what gives these tests something to disagree about.
    """
    model = Model(_tiny_config())
    mx.random.seed(seed)
    head = model.depth_decoder.codebooks_head
    head.weight = mx.random.normal(head.weight.shape) * 0.5
    return model


def _next_token(model: Model, logits: mx.array) -> int:
    """Greedy pick through the same reserved-id mask the real loop applies."""
    return int(mx.argmax(model._mask_reserved_codec_logits(logits)))


def _reference_walk(model: Model, hidden: mx.array, first_codebook: int):
    """The original arithmetic: one full-prefix forward per codebook step."""
    decoder = model.depth_decoder
    logits_list = []
    tokens = [0, first_codebook]
    for _ in range(model.num_codebooks - 1):
        token_ids = mx.array(tokens, dtype=mx.int32)[None, :]
        step = decoder.next_logits(token_ids, hidden)
        mx.eval(step)
        logits_list.append(step)
        tokens.append(_next_token(model, step))
    return logits_list, tokens[1:]


def _cached_walk(model: Model, hidden: mx.array, first_codebook: int):
    """The cached walk: one cache position per codebook step."""
    decoder = model.depth_decoder
    cache = decoder.model.make_cache()
    decoder.start_frame(hidden, cache)
    logits_list = []
    tokens = [0, first_codebook]
    for head_idx in range(model.num_codebooks - 1):
        step = decoder.step_logits(cache, head_idx=head_idx, token_id=tokens[-1])
        mx.eval(step)
        logits_list.append(step)
        tokens.append(_next_token(model, step))
    return logits_list, tokens[1:]


def test_cached_walk_reproduces_prefix_recompute_logits():
    model = _randomized_model()
    hidden = mx.random.normal((1, 16))
    reference, _ = _reference_walk(model, hidden, 1)
    cached, _ = _cached_walk(model, hidden, 1)

    assert len(cached) == len(reference) == model.num_codebooks - 1
    # Guard against a vacuous comparison: a zero head would make every logits
    # tensor agree with every other one.
    assert max(float(mx.abs(step).max()) for step in reference) > 0.1
    for expected, actual in zip(reference, cached):
        # The walks share the same arithmetic and differ only in the order the
        # accumulation happens in, so the tolerated gap is fp32 rounding, not a
        # design allowance (fp64 agreement is exact).
        assert mx.allclose(expected, actual, atol=1e-5).item()
        assert int(mx.argmax(expected)) == int(mx.argmax(actual))


def test_cached_walk_picks_the_same_codebooks_as_recompute():
    model = _randomized_model()
    for seed in range(4):
        mx.random.seed(seed)
        hidden = mx.random.normal((1, 16))
        for first_codebook in (1, 2, 3):
            assert (
                _reference_walk(model, hidden, first_codebook)[1]
                == _cached_walk(model, hidden, first_codebook)[1]
            )


def test_depth_tokens_uses_the_cached_walk():
    model = _randomized_model()
    hidden = mx.random.normal((1, 16))
    expected = _reference_walk(model, hidden, 1)[1]
    tokens = model._depth_tokens(
        1,
        hidden,
        unconditional_hidden=None,
        cfg_scale=1.0,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
    )
    assert tokens == expected


def test_depth_tokens_cfg_matches_the_recompute_path():
    model = _randomized_model()
    hidden = mx.random.normal((1, 16))
    unconditional = mx.random.normal((1, 16))
    tokens = model._depth_tokens(
        1,
        hidden,
        unconditional_hidden=unconditional,
        cfg_scale=2.0,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
    )
    decoder = model.depth_decoder
    cond_cache = decoder.model.make_cache()
    uncond_cache = decoder.model.make_cache()
    decoder.start_frame(hidden, cond_cache)
    decoder.start_frame(unconditional, uncond_cache)
    expected = [0, 1]
    for head_idx in range(model.num_codebooks - 1):
        cond = decoder.step_logits(cond_cache, head_idx=head_idx, token_id=expected[-1])
        uncond = decoder.step_logits(
            uncond_cache, head_idx=head_idx, token_id=expected[-1]
        )
        logits = uncond + 2.0 * (cond - uncond)
        expected.append(_next_token(model, logits))
    assert tokens == expected[1:]


def test_each_step_extends_the_frame_cache_once():
    model = _randomized_model()
    hidden = mx.random.normal((1, 16))
    decoder = model.depth_decoder
    cache = decoder.model.make_cache()

    assert [int(layer.offset) for layer in cache] == [0] * len(cache)
    decoder.start_frame(hidden, cache)
    assert {int(layer.offset) for layer in cache} == {1}
    for head_idx in range(model.num_codebooks - 1):
        decoder.step_logits(cache, head_idx=head_idx, token_id=1)
        assert {int(layer.offset) for layer in cache} == {head_idx + 2}


def test_frames_do_not_share_cache_state():
    model = _randomized_model()
    mx.random.seed(11)
    first_hidden = mx.random.normal((1, 16))
    other_hidden = mx.random.normal((1, 16))

    def frame(hidden):
        return model._depth_tokens(
            1,
            hidden,
            unconditional_hidden=None,
            cfg_scale=1.0,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )

    expected = frame(first_hidden)
    frame(other_hidden)  # interleave an unrelated frame
    assert frame(first_hidden) == expected
    assert frame(other_hidden) == frame(other_hidden)
