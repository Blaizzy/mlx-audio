"""Download-free architecture and checkpoint-loading regressions for Breeze."""

import pytest

try:
    import mlx.core as mx
except (ImportError, RuntimeError) as exc:  # pragma: no cover - CI without Metal
    pytest.skip(f"MLX device unavailable: {exc}", allow_module_level=True)

from mlx_audio.tts.models.breeze_tts.breeze_tts import (
    Model,
    _t5gemma2_attention_mask,
    _TextEncoder,
)
from mlx_audio.tts.models.breeze_tts.config import ModelConfig


def _tiny_config(**overrides):
    values = dict(
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
            "num_codebooks": 3,
            "vocab_size": 8,
            "audio_embed_size": 16,
        },
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_config_keeps_audio_wrapper_fields_distinct_from_nested_qwen():
    config = ModelConfig.from_dict(
        {
            "model_type": "breeze",
            "vocab_size": 151936,
            "num_codebooks": 32,
            "audio_vocab_size": 2051,
            "audio_num_codebooks": 16,
            "backbone_config": {
                "model_type": "qwen3",
                "vocab_size": 151936,
                "hidden_size": 2048,
                "rms_norm_eps": 1e-6,
            },
            "codec_config": {"sampling_rate": 24000},
        }
    )

    assert config.wrapper_vocab_size == 2051
    assert config.wrapper_num_codebooks == 16
    assert config.backbone_value("vocab_size") == 151936
    assert config.backbone_hidden_size == 2048
    assert config.sample_rate == 24000


def test_backbone_honors_nested_qwen_architecture_and_wrapper_embedding_shape():
    config = _tiny_config(
        rms_norm_eps=1e-5,
        rope_theta=500000,
        max_position_embeddings=2048,
        backbone_config={
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
            "vocab_size": 151936,
            "tie_word_embeddings": True,
        },
    )
    model = Model(config)

    assert model.backbone_model.args.rms_norm_eps == 1e-6
    assert model.backbone_model.args.rope_theta == 1_000_000
    assert model.backbone_model.args.max_position_embeddings == 40960
    assert model.backbone_model.args.vocab_size == 151936
    assert model.backbone_model.args.tie_word_embeddings is True
    assert model.backbone_model.embed_tokens.embed_audio_tokens.weight.shape == (
        3 * 8,
        16,
    )


def test_audio_embedding_projection_matches_wrapper_audio_embed_size():
    config = _tiny_config(audio_embed_size=10)
    model = Model(config)

    assert model.backbone_model.embed_tokens.embed_audio_tokens.weight.shape == (
        3 * 8,
        10,
    )
    assert model.backbone_model.embed_tokens.audio_embeds_projector.weight.shape == (
        16,
        10,
    )


def test_t5gemma2_masks_are_bidirectional_and_padding_aware():
    assert _t5gemma2_attention_mask(5, "full_attention", None) is None
    sliding = _t5gemma2_attention_mask(5, "sliding_attention", 4)
    assert sliding.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, False],
        [False, True, True, True, True],
        [False, False, True, True, True],
        [False, False, False, True, True],
    ]

    padding = mx.array([[1, 1, 1, 0, 0]], dtype=mx.int32)
    full_padded = _t5gemma2_attention_mask(
        5, "full_attention", None, attention_mask=padding
    )
    sliding_padded = _t5gemma2_attention_mask(
        5, "sliding_attention", 4, attention_mask=padding
    )
    assert full_padded.shape == (1, 1, 1, 5)
    assert full_padded.tolist() == [[[[True, True, True, False, False]]]]
    assert sliding_padded.shape == (1, 1, 5, 5)
    assert sliding_padded.tolist()[0][0][0] == [True, True, True, False, False]
    assert sliding_padded.tolist()[0][0][4] == [False, False, False, False, False]


def test_text_encoder_accepts_padded_batches_without_causal_mask():
    config = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 16,
        "layer_types": ["full_attention"],
    }
    encoder = _TextEncoder(config)
    hidden = encoder(
        mx.array([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=mx.int32),
        attention_mask=mx.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=mx.int32),
    )
    assert hidden.shape == (2, 4, 8)


def test_depth_heads_follow_nested_depth_configuration():
    config = _tiny_config(
        num_codebooks=5,
        vocab_size=7,
        depth_decoder_config={
            "hidden_size": 12,
            "num_hidden_layers": 1,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 6,
            "rms_norm_eps": 1e-5,
            "num_codebooks": 4,
            "vocab_size": 7,
            "audio_embed_size": 16,
        },
    )
    model = Model(config)
    assert model.depth_decoder.model.embed_tokens.weight.shape == (4 * 7, 16)
    assert model.depth_decoder.codebooks_head.weight.shape == (3, 12, 7)


def test_sanitize_materializes_checkpoint_tied_embedding_and_drops_codec():
    depth = mx.zeros((24, 8))
    sanitized = Model.sanitize(
        {
            "depth_decoder.model.embed_tokens.weight": depth,
            "backbone_model.embed_tokens.embed_audio_tokens.weight": mx.ones((24, 8)),
            "codec_model.decoder.weight": mx.zeros((1,)),
            "text_encoder.rotary_emb.inv_freq": mx.zeros((1,)),
        }
    )
    assert sanitized["depth_decoder.model.embed_tokens.weight"] is depth
    assert sanitized["backbone_model.embed_tokens.embed_audio_tokens.weight"] is depth
    assert "codec_model.decoder.weight" not in sanitized
    assert "text_encoder.rotary_emb.inv_freq" not in sanitized


def test_codec_key_validation_is_exact_except_runtime_initialized_flags():
    expected = {
        "decoder.weight",
        "encoder_model.quantizer.rvq_first.vq.layers.0.codebook.initialized",
    }
    Model._validate_codec_weight_keys(expected, {"decoder.weight"})
    with pytest.raises(ValueError, match="unexpected"):
        Model._validate_codec_weight_keys(expected, {"decoder.weight", "extra"})
    with pytest.raises(ValueError, match="missing"):
        Model._validate_codec_weight_keys(expected, set())
