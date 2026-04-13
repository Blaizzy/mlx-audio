import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_audio.stt.models.sensevoice.config import (
    EncoderConfig,
    FrontendConfig,
    ModelConfig,
)
from mlx_audio.stt.models.sensevoice.sensevoice import (
    EncoderLayerSANM,
    MultiHeadedAttentionSANM,
    PositionwiseFeedForward,
    SenseVoiceEncoder,
    SenseVoiceSmall,
    SinusoidalPositionEncoder,
    _apply_cmvn,
    _apply_lfr,
)


class TestConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.model_type == "sensevoice"
        assert config.vocab_size == 25055
        assert config.input_size == 560
        assert config.encoder_conf.output_size == 512
        assert config.encoder_conf.num_blocks == 50
        assert config.encoder_conf.tp_blocks == 20

    def test_from_dict(self):
        d = {
            "model_type": "sensevoice",
            "vocab_size": 25055,
            "input_size": 560,
            "encoder_conf": {
                "output_size": 512,
                "attention_heads": 4,
                "linear_units": 2048,
                "num_blocks": 50,
                "tp_blocks": 20,
                "kernel_size": 11,
                "sanm_shfit": 0,  # upstream typo
            },
            "frontend_conf": {
                "fs": 16000,
                "n_mels": 80,
                "lfr_m": 7,
                "lfr_n": 6,
            },
        }
        config = ModelConfig.from_dict(d)
        assert config.encoder_conf.sanm_shift == 0
        assert config.frontend_conf.lfr_m == 7


class TestSinusoidalPositionEncoder:
    def test_shape(self):
        enc = SinusoidalPositionEncoder()
        x = mx.zeros((2, 10, 512))
        out = enc(x)
        assert out.shape == (2, 10, 512)

    def test_adds_positional_info(self):
        enc = SinusoidalPositionEncoder()
        x = mx.zeros((1, 5, 64))
        out = enc(x)
        # Output should not be all zeros since position encoding is added
        assert mx.any(out != 0).item()


class TestPositionwiseFeedForward:
    def test_shape(self):
        ff = PositionwiseFeedForward(512, 2048)
        x = mx.zeros((2, 10, 512))
        out = ff(x)
        assert out.shape == (2, 10, 512)


class TestMultiHeadedAttentionSANM:
    def test_shape_same_dim(self):
        attn = MultiHeadedAttentionSANM(
            n_head=4, in_feat=512, n_feat=512, kernel_size=11
        )
        x = mx.zeros((2, 20, 512))
        out = attn(x)
        assert out.shape == (2, 20, 512)

    def test_shape_projection(self):
        """Test first layer where in_feat != n_feat (560 -> 512)."""
        attn = MultiHeadedAttentionSANM(
            n_head=4, in_feat=560, n_feat=512, kernel_size=11
        )
        x = mx.zeros((1, 15, 560))
        out = attn(x)
        assert out.shape == (1, 15, 512)


class TestEncoderLayerSANM:
    def test_shape_same_dim(self):
        attn = MultiHeadedAttentionSANM(
            n_head=4, in_feat=512, n_feat=512, kernel_size=11
        )
        ff = PositionwiseFeedForward(512, 2048)
        layer = EncoderLayerSANM(in_size=512, size=512, self_attn=attn, feed_forward=ff)
        x = mx.zeros((2, 10, 512))
        out = layer(x)
        assert out.shape == (2, 10, 512)

    def test_shape_projection(self):
        attn = MultiHeadedAttentionSANM(
            n_head=4, in_feat=560, n_feat=512, kernel_size=11
        )
        ff = PositionwiseFeedForward(512, 2048)
        layer = EncoderLayerSANM(in_size=560, size=512, self_attn=attn, feed_forward=ff)
        x = mx.zeros((1, 10, 560))
        out = layer(x)
        assert out.shape == (1, 10, 512)


class TestLFR:
    def test_basic(self):
        feats = mx.ones((100, 80))
        lfr = _apply_lfr(feats, lfr_m=7, lfr_n=6)
        expected_t = math.ceil(100 / 6)
        assert lfr.shape == (expected_t, 560)

    def test_short_input(self):
        feats = mx.ones((10, 80))
        lfr = _apply_lfr(feats, lfr_m=7, lfr_n=6)
        assert lfr.shape[1] == 560
        assert lfr.shape[0] == math.ceil(10 / 6)

    def test_left_padding(self):
        feats = mx.arange(20).reshape(20, 1).astype(mx.float32)
        lfr = _apply_lfr(feats, lfr_m=7, lfr_n=6)
        # first output frame should start with 3 copies of feats[0] then feats[0:4]
        first_frame = lfr[0]
        assert first_frame[0].item() == 0.0  # left-padded with feats[0]
        assert first_frame[1].item() == 0.0
        assert first_frame[2].item() == 0.0
        assert first_frame[3].item() == 0.0  # feats[0]
        assert first_frame[4].item() == 1.0  # feats[1]


class TestCMVN:
    def test_basic(self):
        feats = mx.ones((10, 560))
        means = mx.zeros((560,))
        istd = mx.ones((560,))
        out = _apply_cmvn(feats, means, istd)
        assert out.shape == (10, 560)
        np.testing.assert_allclose(np.array(out), np.ones((10, 560)), atol=1e-5)


class TestSenseVoiceEncoder:
    @pytest.fixture
    def small_config(self):
        return ModelConfig(
            input_size=560,
            encoder_conf=EncoderConfig(
                output_size=64,
                attention_heads=2,
                linear_units=128,
                num_blocks=3,
                tp_blocks=2,
                kernel_size=5,
            ),
        )

    def test_shape(self, small_config):
        encoder = SenseVoiceEncoder(small_config)
        x = mx.zeros((1, 10, 560))
        mx.eval(encoder.parameters())
        out = encoder(x)
        assert out.shape == (1, 10, 64)


class TestSenseVoiceSmall:
    @pytest.fixture
    def small_model(self):
        config = ModelConfig(
            input_size=560,
            vocab_size=100,
            encoder_conf=EncoderConfig(
                output_size=64,
                attention_heads=2,
                linear_units=128,
                num_blocks=3,
                tp_blocks=2,
                kernel_size=5,
            ),
        )
        model = SenseVoiceSmall(config)
        mx.eval(model.parameters())
        return model

    def test_forward_shape(self, small_model):
        feats = mx.zeros((1, 10, 560))
        log_probs = small_model(feats, language="en")
        # 4 query tokens + 10 feature frames = 14
        assert log_probs.shape == (1, 14, 100)

    def test_sanitize(self):
        weights = {
            "ctc.ctc_lo.weight": mx.zeros((100, 64)),
            "ctc.ctc_lo.bias": mx.zeros((100,)),
            "encoder.encoders.0.self_attn.fsmn_block.weight": mx.zeros((64, 1, 11)),
            "embed.weight": mx.zeros((16, 560)),
        }
        sanitized = SenseVoiceSmall.sanitize(weights)
        assert "ctc_lo.weight" in sanitized
        assert "ctc.ctc_lo.weight" not in sanitized
        # Conv1d weights transposed
        assert sanitized["encoder.encoders.0.self_attn.fsmn_block.weight"].shape == (
            64,
            11,
            1,
        )
