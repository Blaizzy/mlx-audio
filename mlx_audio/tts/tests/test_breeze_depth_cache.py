"""Cache equivalence and device-side sampling tests for Breeze TTS 2."""

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
    mx.random.seed(seed)
    model = Model(_tiny_config())
    head = model.depth_decoder.codebooks_head
    # The zero-initialized head would make every logit comparison pass.
    head.weight = mx.random.normal(head.weight.shape) * 0.5
    return model


def _next_token(model: Model, logits: mx.array) -> int:
    return int(mx.argmax(model._mask_reserved_codec_logits(logits)))


def _reference_walk(model: Model, hidden: mx.array, first_codebook: int):
    """Recompute the full prefix at each step as an independent reference."""
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


def _synced_walk(
    model, hidden, first_codebook, *, unconditional, cfg_scale, **sampling
):
    """Reference sampling with a host readback after each token."""
    decoder = model.depth_decoder
    cond_cache = decoder.model.make_cache()
    decoder.start_frame(hidden, cond_cache)
    uncond_cache = None
    if unconditional is not None:
        uncond_cache = decoder.model.make_cache()
        decoder.start_frame(unconditional, uncond_cache)
    tokens = [0, first_codebook]
    for head_idx in range(model.num_codebooks - 1):
        logits = decoder.step_logits(cond_cache, head_idx=head_idx, token_id=tokens[-1])
        if uncond_cache is not None:
            uncond = decoder.step_logits(
                uncond_cache, head_idx=head_idx, token_id=tokens[-1]
            )
            logits = uncond + cfg_scale * (logits - uncond)
        logits = model._mask_reserved_codec_logits(logits)
        tokens.append(model._sample(logits, **sampling))
    return tokens[1:]


def test_cached_walk_reproduces_prefix_recompute_logits():
    model = _randomized_model()
    hidden = mx.random.normal((1, 16))
    # GPU attention kernels differ in precision for single-query and full-prefix
    # execution; use CPU for the strict logit comparison.
    with mx.stream(mx.cpu):
        reference, _ = _reference_walk(model, hidden, 1)
        cached, _ = _cached_walk(model, hidden, 1)

    assert len(cached) == len(reference) == model.num_codebooks - 1
    assert max(float(mx.abs(step).max()) for step in reference) > 0.1
    for expected, actual in zip(reference, cached):
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
    expected = [0, 1]
    for _ in range(model.num_codebooks - 1):
        token_ids = mx.array(expected, dtype=mx.int32)[None, :]
        cond = decoder.next_logits(token_ids, hidden)
        uncond = decoder.next_logits(token_ids, unconditional)
        logits = uncond + 2.0 * (cond - uncond)
        expected.append(_next_token(model, logits))
    assert tokens == expected[1:]


@pytest.mark.parametrize("cfg_scale", [None, 2.0])
def test_depth_tokens_stay_on_device_and_keep_the_rng_order(monkeypatch, cfg_scale):
    model = _randomized_model()
    hidden = mx.random.normal((1, 16))
    unconditional = mx.random.normal((1, 16)) if cfg_scale else None
    kwargs = dict(
        unconditional=unconditional,
        cfg_scale=cfg_scale or 1.0,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )
    expected = []
    for seed in range(4):
        mx.random.seed(seed)
        expected.append(_synced_walk(model, hidden, 1, **kwargs))
    # Ensure this exercises stochastic sampling.
    assert len({tuple(tokens) for tokens in expected}) > 1

    def synced_sample(*_args, **_kwargs):
        raise AssertionError("the depth loop read a token back to the host")

    monkeypatch.setattr(model, "_sample", synced_sample)
    kwargs["unconditional_hidden"] = kwargs.pop("unconditional")
    for seed, tokens in enumerate(expected):
        mx.random.seed(seed)
        assert model._depth_tokens(1, hidden, **kwargs) == tokens


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
