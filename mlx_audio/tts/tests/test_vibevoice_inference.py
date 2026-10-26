from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from mlx_audio.lm.models.cache import KVCache
from mlx_audio.tts.models.vibevoice.config import (
    AcousticTokenizerConfig,
    DiffusionHeadConfig,
    ModelConfig,
    Qwen2DecoderConfig,
    SemanticTokenizerConfig,
)
from mlx_audio.tts.models.vibevoice.language_model import Attention
from mlx_audio.tts.models.vibevoice.vibevoice import Model


def tiny_model():
    model = Model(
        ModelConfig(
            model_type="vibevoice",
            acoustic_vae_dim=4,
            semantic_vae_dim=8,
            decoder_config=Qwen2DecoderConfig(
                hidden_size=64,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_hidden_layers=1,
                vocab_size=64,
            ),
            diffusion_head_config=DiffusionHeadConfig(
                hidden_size=64,
                head_layers=1,
                head_ffn_ratio=2,
                latent_size=4,
                ddpm_num_steps=100,
            ),
            acoustic_tokenizer_config=AcousticTokenizerConfig(
                vae_dim=4,
                encoder_n_filters=2,
                encoder_ratios=[2, 2],
                encoder_depths="1-1-1",
                decoder_ratios=[2, 2],
                decoder_depths="1-1-1",
            ),
            semantic_tokenizer_config=SemanticTokenizerConfig(
                vae_dim=8,
                encoder_n_filters=2,
                encoder_ratios=[2, 2],
                encoder_depths="1-1-1",
            ),
        )
    )
    return model


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("cfg_scale", [1.0, 1.5])
def test_diffusion_feedback_preserves_model_dtype(dtype, cfg_scale):
    model = tiny_model()
    model.update(tree_map(lambda value: value.astype(dtype), model.parameters()))
    condition = mx.ones((1, 64), dtype=dtype)
    latent = model.sample_speech_tokens(
        condition,
        condition,
        cfg_scale=cfg_scale,
        ddpm_steps=2,
    )
    acoustic = model.acoustic_connector(latent[:, None, :])
    wave = model.acoustic_tokenizer.decode(latent[:, None, :])
    semantic = model.semantic_tokenizer.encode(wave)
    feedback = acoustic + model.semantic_connector(semantic)
    hidden, cache = model.language_model(
        inputs_embeds=feedback, cache=model.language_model.make_cache()
    )
    mx.eval(hidden)
    assert (
        latent.dtype == feedback.dtype == hidden.dtype == cache[0].keys.dtype == dtype
    )


def control_model():
    model = Model.__new__(Model)
    model.speech_start_id, model.speech_end_id = 1, 2
    model.speech_diffusion_id, model.eos_id, model.bos_id = 3, 4, 5
    return model


@pytest.mark.parametrize(
    "temperature,top_k,top_p,expected",
    [
        (1.0, None, 0.8, [1, 2]),
        (1.0, 2, 0.65, [1]),
        (0.5, None, 0.8, [1]),
        (0.5, 99, 0.8, [1]),
        (1.0, 1, 0.99, [1]),
    ],
)
def test_control_sampling_filters_in_temperature_top_k_top_p_order(
    temperature, top_k, top_p, expected
):
    model = control_model()
    logits = mx.log(mx.array([[1e-10, 1e-10, 0.6, 0.25, 0.1, 0.05]]))
    with patch(
        "mlx_audio.tts.models.vibevoice.vibevoice.mx.random.categorical",
        return_value=mx.array([1]),
    ) as sampler:
        token = model._select_non_streaming_next_token(
            logits, do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p
        )
    filtered = np.array(sampler.call_args.args[0][0])
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(filtered)), expected)
    assert token == 2


def test_negative_reset_discards_old_utterance_and_restarts_positions():
    config = Qwen2DecoderConfig(
        hidden_size=4, num_attention_heads=1, num_key_value_heads=1, head_dim=4
    )
    attention = Attention(config)
    for name in ("q_proj", "k_proj", "v_proj"):
        layer = getattr(attention, name)
        layer.weight, layer.bias = mx.zeros_like(layer.weight), mx.zeros_like(
            layer.bias
        )
    attention.o_proj.weight = mx.eye(4)
    cache = KVCache()
    cache.update_and_fetch(
        mx.zeros((1, 1, 3, 4)),
        mx.broadcast_to(mx.array([1.0, 2.0, 3.0])[None, None, :, None], (1, 1, 3, 4)),
    )
    cache = control_model()._reset_negative_cache_on_speech_start([cache])[0]
    assert cache.offset == 1
    actual, cache = attention(
        mx.zeros((1, 1, 4)), mx.ones((1, 1, 4)), mx.zeros((1, 1, 4)), cache=cache
    )
    np.testing.assert_allclose(np.array(actual), np.full((1, 1, 4), 0.5))
    assert cache.offset == 2


def test_compact_negative_cache_matches_masked_reference_positions():
    from mlx_audio.tts.models.vibevoice.language_model import Qwen2Model

    model = Qwen2Model(
        Qwen2DecoderConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
        )
    )
    prefix = mx.arange(5)[None, :]
    caches = [model.make_cache(), model.make_cache()]
    for cache in caches:
        hidden, _ = model(input_ids=prefix, cache=cache)
        mx.eval(hidden)
    compact = control_model()._reset_negative_cache_on_speech_start(caches[0])
    masked = caches[1]
    for layer in masked:
        last = layer.offset - 1
        layer.keys[..., last : last + 1, :] = layer.keys[..., :1, :]
        layer.values[..., last : last + 1, :] = layer.values[..., :1, :]
    token = mx.array([[6]])
    actual, _ = model(input_ids=token, cache=compact)
    rotary = model.rotary_emb
    model.rotary_emb = lambda positions: rotary(mx.array([1]))
    mask = mx.array([[[[False, False, False, False, True, True]]]])
    expected, _ = model(input_ids=token, cache=masked, mask=mask)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=1e-5, rtol=1e-5
    )


def test_config_keeps_input_and_accepts_checkpoint_aliases():
    params = {
        "model_type": "vibevoice",
        "acostic_vae_dim": 4,
        "sampling_rate": 16000,
        "decoder_config": {"hidden_size": 64},
    }
    original = deepcopy(params)
    config = ModelConfig.from_dict(params)
    assert params == original
    assert config.acoustic_vae_dim == 4
    assert config.sample_rate == 16000
    assert config.decoder_config.hidden_size == 64


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_growing_cache_matches_full_causal_forward(dtype):
    model = tiny_model().language_model
    model.update(tree_map(lambda value: value.astype(dtype), model.parameters()))
    ids = mx.array([[0, 1, 2, 3, 4]])
    full, _ = model(input_ids=ids)
    cache = model.make_cache()
    prefix, cache = model(input_ids=ids[:, :3], cache=cache)
    next_hidden, cache = model(input_ids=ids[:, 3:], cache=cache)
    actual = mx.concatenate([prefix, next_hidden], axis=1)
    assert cache[0].offset == 5
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(full.astype(mx.float32)),
        atol=0.02 if dtype == mx.bfloat16 else 1e-5,
        rtol=0.02 if dtype == mx.bfloat16 else 1e-5,
    )


def test_reference_padding_keeps_partial_final_frame():
    model = tiny_model()
    audio = mx.ones((1, 1, 13))
    encoded = model.acoustic_tokenizer.encode(audio)
    assert encoded.mean.shape == (1, 4, 4)


def test_streaming_diffusion_preserves_float32_precision():
    config = tiny_model().config
    config.model_type = "vibevoice_streaming"
    config.decoder_config.num_hidden_layers = 2
    config.tts_backbone_num_hidden_layers = 1
    model = Model(config)
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    condition = mx.ones((1, 64), dtype=mx.bfloat16)
    latent = model.sample_speech_tokens(condition, condition, ddpm_steps=2)
    assert latent.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(latent)).item())


def test_generation_resets_segment_caches_and_stops_at_eos():
    model = tiny_model()
    ids = {"<|vision_start|>": 1, "<|vision_end|>": 2, "<|vision_pad|>": 3}
    model.tokenizer = SimpleNamespace(
        convert_tokens_to_ids=ids.get,
        bos_token_id=None,
        eos_token_id=4,
        encode=lambda text, **kwargs: [0],
    )
    tokens = [3, 2, 1, 3, 4]
    with (
        patch.object(
            model, "_select_non_streaming_next_token", side_effect=tokens
        ) as select,
        patch.object(
            model,
            "_reset_negative_cache_on_speech_start",
            wraps=model._reset_negative_cache_on_speech_start,
        ) as reset,
    ):
        result = next(model.generate("Hello", max_tokens=20, ddpm_steps=2))
    assert select.call_count == len(tokens)
    assert reset.call_count == 1
    assert reset.call_args.args[0][0].offset == 2
    assert result.samples == 8
    assert result.audio.shape == (8,)
    assert bool(mx.all(mx.isfinite(result.audio)).item())


@pytest.mark.parametrize("encoder", ["acoustic", "semantic"])
def test_cached_encoder_matches_full_frames(encoder):
    model = tiny_model()
    encode = (
        model.acoustic_tokenizer.encoder
        if encoder == "acoustic"
        else model.semantic_tokenizer.encoder
    )
    audio = mx.arange(16, dtype=mx.float32).reshape(1, 1, 16) / 16
    expected = encode(audio)
    cache = {}
    actual = mx.concatenate(
        [encode(audio[..., :8], cache=cache), encode(audio[..., 8:], cache=cache)],
        axis=1,
    )
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=1e-5, rtol=1e-5
    )


def test_multiple_speaker_references_fill_only_voice_slots():
    model = control_model()
    model.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [0])
    script = model._parse_non_streaming_script("Speaker 1: Hello\nSpeaker 2: Hi")
    tokens, mask = model._build_non_streaming_prompt_tokens(script, [2, 1])
    embeddings = mx.zeros((1, len(tokens), 4))
    refs = [mx.ones((2, 4)), mx.full((1, 4), 2.0)]
    actual = model._inject_prefill_speech_embeddings(embeddings, mask, refs)
    expected = np.zeros((1, len(tokens), 4))
    expected[0, np.array(mask)] = np.concatenate([np.ones((2, 4)), np.full((1, 4), 2)])
    np.testing.assert_array_equal(np.array(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_reference_prefill_replaces_nonzero_values_exactly(dtype):
    model = control_model()
    original = mx.ones((1, 3, 4), dtype=dtype)
    reference = mx.full((1, 4), 0.001, dtype=dtype)
    result = model._inject_prefill_speech_embeddings(
        original, [False, True, False], [reference]
    )
    np.testing.assert_array_equal(
        np.array(result[0, 1].astype(mx.float32)),
        np.array(reference[0].astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(result[0, [0, 2]].astype(mx.float32)), np.ones((2, 4))
    )
    np.testing.assert_array_equal(
        np.array(original.astype(mx.float32)), np.ones((1, 3, 4))
    )


def test_stereo_reference_loaded_as_array_matches_mono(tmp_path):
    from mlx_audio.audio_io import read, write

    model = tiny_model()
    t = np.arange(160, dtype=np.float32)
    stereo = np.column_stack([np.sin(t / 10), np.cos(t / 10)]) / 4
    path = tmp_path / "stereo.wav"
    write(str(path), stereo, model.sample_rate)
    samples, _ = read(str(path), dtype="float32")
    assert samples.shape == (160, 2)
    mx.random.seed(42)
    actual = model._prepare_single_non_streaming_ref_embedding(mx.array(samples))
    mx.random.seed(42)
    expected = model._prepare_single_non_streaming_ref_embedding(
        mx.array(samples.mean(axis=-1))
    )
    assert actual.shape == (40, 64)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5
    )
