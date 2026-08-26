import pytest

try:
    import mlx.core as mx
except (ImportError, RuntimeError) as exc:  # pragma: no cover - no Metal host
    pytest.skip(f"MLX device unavailable: {exc}", allow_module_level=True)

from mlx_audio.tts.models.breeze_tts.breeze_tts import (
    Model,
    _t5gemma2_attention_mask,
)
from mlx_audio.tts.models.breeze_tts.config import ModelConfig
from mlx_audio.tts.utils import get_model_and_args
from mlx_audio.utils import get_model_name_parts


def tiny_config():
    return ModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_codebooks=3,
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
        },
        depth_decoder_config={
            "hidden_size": 12,
            "num_hidden_layers": 1,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 6,
            "rms_norm_eps": 1e-5,
            "num_codebooks": 3,
            "vocab_size": 8,
            "audio_embed_size": 16,
        },
    )


def test_config_reads_breeze_nested_sample_rate():
    config = ModelConfig.from_dict(
        {
            "model_type": "breeze",
            "codec_config": {"sampling_rate": 24000},
            "backbone_config": {
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000,
                "max_position_embeddings": 40960,
            },
        }
    )
    assert config.model_type == "breeze"
    assert config.sample_rate == 24000
    assert config.backbone_config["rms_norm_eps"] == 1e-6


def test_backbone_uses_nested_qwen_config_not_top_level_rope_settings():
    config = tiny_config()
    config.rms_norm_eps = 1e-5
    config.rope_theta = 500000
    config.max_position_embeddings = 2048
    config.rope_scaling = {"rope_type": "llama3", "factor": 32.0}
    config.backbone_config = {
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "rope_scaling": None,
        "max_position_embeddings": 40960,
    }

    model = Model(config)
    assert model.backbone_model.args.rms_norm_eps == 1e-6
    assert model.backbone_model.args.rope_theta == 1_000_000
    assert model.backbone_model.args.max_position_embeddings == 40960
    assert model.backbone_model.args.rope_scaling is None


def test_t5gemma2_masks_are_bidirectional_with_symmetric_local_window():
    assert _t5gemma2_attention_mask(5, "full_attention", None) is None
    mask = _t5gemma2_attention_mask(5, "sliding_attention", 4)
    assert mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, False],
        [False, True, True, True, True],
        [False, False, True, True, True],
        [False, False, False, True, True],
    ]


def test_depth_decoder_predicts_each_remaining_codebook():
    model = Model(tiny_config())
    logits = model.depth_decoder.next_logits(
        mx.array([[0, 1]], dtype=mx.int32), mx.zeros((1, 16))
    )
    assert logits.shape == (1, 8)


def test_sanitize_keeps_main_checkpoint_weights_only():
    weights = {
        "lm_head.weight": mx.zeros((9, 16)),
        "text_encoder.layers.0.pre_self_attn_layernorm.weight": mx.zeros((16,)),
        "codec_model.decoder.weight": mx.zeros((1,)),
        "codec_model.quantizer.initialized": mx.zeros((1,)),
        "backbone_model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((1,)),
    }
    sanitized = Model.sanitize(weights)
    assert list(sanitized) == [
        "lm_head.weight",
        "text_encoder.layers.0.pre_self_attn_layernorm.weight",
    ]


def test_codec_key_validation_allows_only_generated_quantizer_flags():
    expected = {
        "decoder.weight",
        "encoder_model.quantizer.rvq_first.vq.layers.0.codebook.initialized",
    }
    Model._validate_codec_weight_keys(expected, {"decoder.weight"})
    with pytest.raises(ValueError, match="unexpected"):
        Model._validate_codec_weight_keys(expected, {"decoder.weight", "other.weight"})
    with pytest.raises(ValueError, match="missing"):
        Model._validate_codec_weight_keys(expected, set())


def test_default_generic_voice_maps_to_breeze_s0():
    model = Model(tiny_config())
    assert model._speaker("af_heart") == "[S0]"
    assert model._speaker("S2") == "[S2]"
    assert model._speaker("[S3]") == "[S3]"


def test_depth_cfg_applies_to_every_remaining_codebook(monkeypatch):
    model = Model(tiny_config())
    sampled_logits = []

    def next_logits(_token_ids, hidden):
        return mx.full((1, 8), hidden[0, 0])

    def sample(logits, **_kwargs):
        sampled_logits.append(logits)
        return 1

    monkeypatch.setattr(model.depth_decoder, "next_logits", next_logits)
    monkeypatch.setattr(model, "_sample", sample)
    tokens = model._depth_tokens(
        1,
        mx.ones((1, 16)),
        unconditional_hidden=mx.zeros((1, 16)),
        cfg_scale=3.0,
        temperature=0,
        top_p=1,
        top_k=0,
    )

    assert tokens == [1, 1, 1]
    assert len(sampled_logits) == 2
    assert all(float(logits[0, 0].item()) == 3.0 for logits in sampled_logits)


def test_depth_cfg_masks_reserved_tokens_at_every_step(monkeypatch):
    model = Model(tiny_config())
    sampled_logits = []

    def next_logits(_token_ids, _hidden):
        return mx.arange(8, dtype=mx.float32)[None, :]

    def sample(logits, **_kwargs):
        sampled_logits.append(logits)
        return 1

    monkeypatch.setattr(model.depth_decoder, "next_logits", next_logits)
    monkeypatch.setattr(model, "_sample", sample)
    model._depth_tokens(
        1,
        mx.ones((1, 16)),
        unconditional_hidden=mx.zeros((1, 16)),
        cfg_scale=2.0,
        temperature=0,
        top_p=1,
        top_k=0,
    )

    assert len(sampled_logits) == 2
    for logits in sampled_logits:
        assert logits[0, :5].tolist() == [0, 1, 2, 3, 4]
        assert all(value == float("-inf") for value in logits[0, 5:].tolist())


def test_repetition_penalty_is_sign_aware():
    logits = mx.array([[4.0, -2.0, 7.0, -5.0]])
    penalized = Model._apply_repetition_penalty(logits, [0, 1, 1, 3], 2.0)
    assert penalized.tolist() == [[2.0, -4.0, 7.0, -10.0]]


def test_codec_vocab_boundary_comes_from_config():
    config = tiny_config()
    config.codec_config = {"codebook_size": 5}
    model = Model(config)
    token = model._sample(
        mx.array([[0.0, 0.0, 1.0, 0.0, 2.0, 99.0, 98.0, 97.0, 96.0]]),
        temperature=0,
        top_p=1,
        top_k=0,
    )
    assert token == 4


def test_reference_audio_batch_is_rejected_before_codec_encode():
    model = Model(tiny_config())
    model.audio_tokenizer = object()
    with pytest.raises(ValueError, match="exactly one reference audio"):
        model._encode_reference(mx.zeros((2, 10)))


def test_multiple_reference_paths_are_rejected_before_prompt_encoding():
    model = Model(tiny_config())
    with pytest.raises(ValueError, match="exactly one reference audio"):
        model._prompt_embeddings(
            "target",
            voice=None,
            instruct=None,
            ref_audio=["a.wav", "b.wav"],
            ref_text=["a", "b"],
        )


def test_exact_prompt_template_for_cloning_and_direction(monkeypatch):
    model = Model(tiny_config())
    prompts = []

    def text_ids(value):
        prompts.append(value)
        return mx.array([1, 2], dtype=mx.int32)

    monkeypatch.setattr(model, "_text_ids", text_ids)
    monkeypatch.setattr(
        model, "_encode_reference", lambda _audio: mx.zeros((1, 1, 3), mx.int32)
    )
    model._prompt_embeddings(
        "target",
        voice="S2",
        instruct="warm",
        ref_audio="reference.wav",
        ref_text="reference text",
    )
    assert prompts == ["[S2]reference text", "[S2]<ins_bos>warm<ins_eos>target"]


def test_stream_flushes_at_exact_interval_and_resets_state(monkeypatch):
    model = Model(tiny_config())

    class Decoder:
        decode_upsample_rate = 24000

        def __init__(self):
            self.reset_calls = 0

        def reset_streaming_state(self):
            self.reset_calls += 1

        def streaming_step(self, codes):
            return mx.zeros((1, 1, codes.shape[-1]))

    class AudioTokenizer:
        decode_upsample_rate = 24000

        def __init__(self):
            self.decoder = Decoder()

    class Head:
        def __init__(self):
            self.calls = 0

        def __call__(self, _hidden):
            self.calls += 1
            logits = mx.full((1, 9), -100.0)
            logits[..., 1 if self.calls <= 3 else 8] = 100.0
            return logits

    model.audio_tokenizer = AudioTokenizer()
    monkeypatch.setattr(
        model, "_prompt_embeddings", lambda *_args, **_kwargs: mx.zeros((1, 1, 16))
    )
    monkeypatch.setattr(
        model, "_depth_tokens", lambda first, *_args, **_kwargs: [first, 2, 3]
    )
    model.lm_head = Head()

    chunks = list(
        model.generate(
            "test", temperature=0, top_k=0, stream=True, streaming_interval=2.0
        )
    )
    assert len(chunks) == 2
    assert chunks[0].token_count == 2
    assert chunks[0].prompt["tokens"] == 2
    assert chunks[0].prompt["tokens-per-sec"] >= 0
    assert chunks[0].audio_samples["samples-per-sec"] >= 0
    assert not chunks[0].is_final_chunk
    assert chunks[1].token_count == 1
    assert chunks[1].prompt["tokens"] == 1
    assert chunks[1].prompt["tokens-per-sec"] >= 0
    assert chunks[1].audio_samples["samples-per-sec"] >= 0
    assert chunks[1].is_final_chunk
    assert model.audio_tokenizer.decoder.reset_calls == 2


def test_early_eos_yields_silent_result_instead_of_raising(monkeypatch):
    model = Model(tiny_config())

    class Decoder:
        decode_upsample_rate = 24000

        def reset_streaming_state(self):
            pass

    class AudioTokenizer:
        decode_upsample_rate = 24000
        decoder = Decoder()

        def decode(self, codes):
            assert codes.shape == (1, 1, 3)
            return mx.zeros((1, 4)), mx.array([4], dtype=mx.int32)

    class Backbone:
        def make_cache(self):
            return []

        def __call__(self, input_embeddings=None, input_ids=None, cache=None):
            del cache
            length = (
                input_embeddings.shape[1]
                if input_embeddings is not None
                else input_ids.shape[1]
            )
            return mx.zeros((1, length, 16))

    class Head:
        def __call__(self, _hidden):
            logits = mx.full((1, 9), -100.0)
            logits[..., 8] = 100.0
            return logits

    model.audio_tokenizer = AudioTokenizer()
    model.backbone_model = Backbone()
    model.lm_head = Head()
    monkeypatch.setattr(
        model, "_prompt_embeddings", lambda *_args, **_kwargs: mx.zeros((1, 1, 16))
    )

    results = list(model.generate("test", temperature=0, top_k=0))
    assert len(results) == 1
    assert results[0].is_final_chunk
    assert results[0].token_count == 0
    assert results[0].samples == 4
    assert results[0].prompt == {"tokens": 0, "tokens-per-sec": 0.0}
    assert results[0].audio_samples["samples"] == 4
    assert results[0].audio_samples["samples-per-sec"] >= 0
    assert results[0].audio.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_breeze_registry_loads_model_module():
    module, model_type = get_model_and_args(
        "breeze", get_model_name_parts("BreezeBlue/breeze-tts-2")
    )
    assert model_type == "breeze_tts"
    assert module.Model is Model
