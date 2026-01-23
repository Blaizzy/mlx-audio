# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.vibevoice_asr.config import (
    AcousticTokenizerConfig,
    ModelConfig,
    Qwen2Config,
    SemanticTokenizerConfig,
)
from mlx_audio.stt.models.vibevoice_asr.audio_encoder import (
    AcousticTokenizerEncoder,
    Block1D,
    ConvRMSNorm,
    SConv1d,
    SemanticTokenizerEncoder,
    TokenizerEncoder,
)
from mlx_audio.stt.models.vibevoice_asr.vibevoice_asr import (
    LanguageModel,
    Model,
    SpeechConnector,
)


def _small_acoustic_config():
    """Create a small acoustic tokenizer config for fast testing."""
    return AcousticTokenizerConfig(
        channels=1,
        vae_dim=8,
        encoder_n_filters=4,
        encoder_ratios=[2, 2],
        encoder_depths="2-2-2",
        causal=True,
    )


def _small_semantic_config():
    """Create a small semantic tokenizer config for fast testing."""
    return SemanticTokenizerConfig(
        channels=1,
        vae_dim=16,
        encoder_n_filters=4,
        encoder_ratios=[2, 2],
        encoder_depths="2-2-2",
        causal=True,
    )


def _small_qwen2_config():
    """Create a small Qwen2 config for fast testing."""
    return Qwen2Config(
        vocab_size=1000,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        tie_word_embeddings=False,
        layer_types=["full_attention"] * 2,
    )


def _small_model_config():
    """Create a small model config for fast testing."""
    return ModelConfig(
        acoustic_tokenizer_config=_small_acoustic_config(),
        semantic_tokenizer_config=_small_semantic_config(),
        decoder_config=_small_qwen2_config(),
        acoustic_vae_dim=8,
        semantic_vae_dim=16,
    )


class TestConfig(unittest.TestCase):
    """Tests for VibeVoice-ASR configuration classes."""

    def test_acoustic_tokenizer_config_from_dict(self):
        config_dict = {
            "channels": 1,
            "vae_dim": 64,
            "encoder_n_filters": 32,
            "encoder_ratios": [8, 5, 5, 4, 2, 2],
            "encoder_depths": "3-3-3-3-3-3-8",
            "causal": True,
            "fix_std": 0.5,
            "std_dist_type": "gaussian",
        }
        config = AcousticTokenizerConfig.from_dict(config_dict)

        self.assertEqual(config.channels, 1)
        self.assertEqual(config.vae_dim, 64)
        self.assertEqual(config.encoder_n_filters, 32)
        self.assertEqual(config.encoder_ratios, [8, 5, 5, 4, 2, 2])
        self.assertEqual(config.fix_std, 0.5)
        self.assertEqual(config.std_dist_type, "gaussian")

    def test_acoustic_tokenizer_parsed_depths(self):
        config = AcousticTokenizerConfig(encoder_depths="3-3-3-3-3-3-8")
        self.assertEqual(config.parsed_encoder_depths, [3, 3, 3, 3, 3, 3, 8])

    def test_acoustic_tokenizer_parsed_depths_list(self):
        config = AcousticTokenizerConfig(encoder_depths=[3, 3, 3])
        self.assertEqual(config.parsed_encoder_depths, [3, 3, 3])

    def test_semantic_tokenizer_config_from_dict(self):
        config_dict = {
            "channels": 1,
            "vae_dim": 128,
            "encoder_n_filters": 32,
            "encoder_ratios": [8, 5, 5, 4, 2, 2],
            "encoder_depths": "3-3-3-3-3-3-8",
            "std_dist_type": "none",
        }
        config = SemanticTokenizerConfig.from_dict(config_dict)

        self.assertEqual(config.vae_dim, 128)
        self.assertEqual(config.std_dist_type, "none")

    def test_qwen2_config_from_dict(self):
        config_dict = {
            "vocab_size": 152064,
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "intermediate_size": 18944,
        }
        config = Qwen2Config.from_dict(config_dict)

        self.assertEqual(config.vocab_size, 152064)
        self.assertEqual(config.hidden_size, 3584)
        self.assertEqual(config.head_dim, 128)

    def test_qwen2_config_head_dim(self):
        config = Qwen2Config(hidden_size=256, num_attention_heads=8)
        self.assertEqual(config.head_dim, 32)

    def test_model_config_from_dict(self):
        config_dict = {
            "model_type": "vibevoice_asr",
            "acoustic_tokenizer_config": {
                "vae_dim": 64,
                "encoder_n_filters": 32,
            },
            "semantic_tokenizer_config": {
                "vae_dim": 128,
                "encoder_n_filters": 32,
            },
            "decoder_config": {
                "vocab_size": 152064,
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
            },
            "acoustic_vae_dim": 64,
            "semantic_vae_dim": 128,
        }
        config = ModelConfig.from_dict(config_dict)

        self.assertEqual(config.model_type, "vibevoice_asr")
        self.assertIsInstance(config.acoustic_tokenizer_config, AcousticTokenizerConfig)
        self.assertIsInstance(config.semantic_tokenizer_config, SemanticTokenizerConfig)
        self.assertIsInstance(config.decoder_config, Qwen2Config)
        self.assertEqual(config.acoustic_tokenizer_config.vae_dim, 64)
        self.assertEqual(config.semantic_tokenizer_config.vae_dim, 128)
        self.assertEqual(config.decoder_config.hidden_size, 3584)

    def test_model_config_ignores_extra_keys(self):
        config_dict = {
            "model_type": "vibevoice_asr",
            "acoustic_tokenizer_config": {"vae_dim": 64, "unknown_key": 999},
            "semantic_tokenizer_config": {},
            "decoder_config": {"vocab_size": 1000, "extra_field": True},
        }
        config = ModelConfig.from_dict(config_dict)
        self.assertEqual(config.acoustic_tokenizer_config.vae_dim, 64)
        self.assertEqual(config.decoder_config.vocab_size, 1000)


class TestSConv1d(unittest.TestCase):
    """Tests for SConv1d (causal convolution)."""

    def test_output_shape_stride1(self):
        conv = SConv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1)
        x = mx.random.normal((1, 100, 1))  # [B, T, C]
        y = conv(x)
        # Stride 1, causal padding preserves length
        self.assertEqual(y.shape[0], 1)
        self.assertEqual(y.shape[1], 100)
        self.assertEqual(y.shape[2], 32)

    def test_output_shape_stride2(self):
        conv = SConv1d(in_channels=4, out_channels=8, kernel_size=4, stride=2)
        x = mx.random.normal((1, 100, 4))
        y = conv(x)
        self.assertEqual(y.shape[0], 1)
        self.assertEqual(y.shape[1], 50)  # 100 / 2
        self.assertEqual(y.shape[2], 8)

    def test_groups_parameter(self):
        conv = SConv1d(in_channels=8, out_channels=8, kernel_size=7, stride=1, groups=8)
        x = mx.random.normal((1, 50, 8))
        y = conv(x)
        self.assertEqual(y.shape, (1, 50, 8))


class TestConvRMSNorm(unittest.TestCase):
    """Tests for ConvRMSNorm."""

    def test_output_shape(self):
        norm = ConvRMSNorm(dim=16)
        x = mx.random.normal((1, 16, 50))  # [B, C, T]
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_normalization(self):
        norm = ConvRMSNorm(dim=4, elementwise_affine=False)
        x = mx.ones((1, 4, 10)) * 5.0
        y = norm(x)
        mx.eval(y)
        # RMSNorm of a constant should normalize it
        self.assertTrue(y.shape == x.shape)


class TestBlock1D(unittest.TestCase):
    """Tests for Block1D transformer block."""

    def test_output_shape(self):
        block = Block1D(dim=16, kernel_size=7, causal=True)
        x = mx.random.normal((1, 50, 16))  # [B, T, C]
        y = block(x)
        self.assertEqual(y.shape, x.shape)

    def test_residual_connection(self):
        block = Block1D(dim=8, kernel_size=3, causal=True, layer_scale_init_value=0.0)
        x = mx.zeros((1, 20, 8))
        y = block(x)
        mx.eval(y)
        # With zero input, output should be close to zero (residual + small FFN output)
        self.assertEqual(y.shape, x.shape)


class TestTokenizerEncoder(unittest.TestCase):
    """Tests for TokenizerEncoder."""

    def test_output_shape(self):
        encoder = TokenizerEncoder(
            channels=1,
            vae_dim=8,
            n_filters=4,
            ratios=[2, 2],
            depths=[2, 2, 2],
            causal=True,
        )
        # Input: [B, 1, T] - 1 channel, T samples
        x = mx.random.normal((1, 1, 160))
        y = encoder(x)
        mx.eval(y)

        # Total downsampling: product of ratios = 2*2 = 4
        # T' = 160 / 4 = 40
        self.assertEqual(y.shape[0], 1)  # batch
        self.assertEqual(y.shape[1], 40)  # time (downsampled)
        self.assertEqual(y.shape[2], 8)  # vae_dim

    def test_2d_input(self):
        encoder = TokenizerEncoder(
            channels=1, vae_dim=8, n_filters=4, ratios=[2], depths=[2, 2]
        )
        # [B, T] input without channel dim
        x = mx.random.normal((1, 80))
        y = encoder(x)
        mx.eval(y)

        self.assertEqual(y.shape[0], 1)
        self.assertEqual(y.shape[1], 40)  # 80 / 2
        self.assertEqual(y.shape[2], 8)

    def test_hop_length(self):
        encoder = TokenizerEncoder(
            channels=1, vae_dim=8, n_filters=4, ratios=[4, 2, 2], depths=[1, 1, 1, 1]
        )
        # hop_length should be product of ratios
        self.assertEqual(encoder.hop_length, 16)


class TestAcousticTokenizerEncoder(unittest.TestCase):
    """Tests for AcousticTokenizerEncoder."""

    def test_encode_output_shape(self):
        config = _small_acoustic_config()
        encoder = AcousticTokenizerEncoder(config)

        x = mx.random.normal((1, 1, 160))  # [B, 1, T]
        y = encoder.encode(x)
        mx.eval(y)

        # Downsampled by product of ratios (2*2=4): 160/4 = 40
        self.assertEqual(y.shape[0], 1)
        self.assertEqual(y.shape[1], 40)
        self.assertEqual(y.shape[2], config.vae_dim)

    def test_sample_gaussian(self):
        config = _small_acoustic_config()
        config.fix_std = 0.5
        config.std_dist_type = "gaussian"
        encoder = AcousticTokenizerEncoder(config)

        mean = mx.zeros((1, 10, config.vae_dim))
        sampled = encoder.sample(mean)
        mx.eval(sampled)

        self.assertEqual(sampled.shape, mean.shape)

    def test_sample_none(self):
        config = _small_acoustic_config()
        config.std_dist_type = "none"
        encoder = AcousticTokenizerEncoder(config)

        mean = mx.ones((1, 10, config.vae_dim))
        sampled = encoder.sample(mean)
        mx.eval(sampled)

        # With "none" dist type, output should equal input
        self.assertTrue(mx.allclose(sampled, mean).item())


class TestSemanticTokenizerEncoder(unittest.TestCase):
    """Tests for SemanticTokenizerEncoder."""

    def test_encode_output_shape(self):
        config = _small_semantic_config()
        encoder = SemanticTokenizerEncoder(config)

        x = mx.random.normal((1, 1, 160))
        y = encoder.encode(x)
        mx.eval(y)

        self.assertEqual(y.shape[0], 1)
        self.assertEqual(y.shape[1], 40)  # 160 / (2*2)
        self.assertEqual(y.shape[2], config.vae_dim)


class TestSpeechConnector(unittest.TestCase):
    """Tests for SpeechConnector (MLP projection)."""

    def test_output_shape(self):
        connector = SpeechConnector(input_dim=8, output_dim=32)
        x = mx.random.normal((1, 10, 8))
        y = connector(x)
        mx.eval(y)

        self.assertEqual(y.shape, (1, 10, 32))

    def test_different_dims(self):
        connector = SpeechConnector(input_dim=16, output_dim=64)
        x = mx.random.normal((2, 5, 16))
        y = connector(x)
        mx.eval(y)

        self.assertEqual(y.shape, (2, 5, 64))


class TestLanguageModel(unittest.TestCase):
    """Tests for LanguageModel (Qwen2 wrapper)."""

    def test_forward_pass(self):
        config = _small_qwen2_config()
        lm = LanguageModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = lm(input_ids)
        mx.eval(logits)

        self.assertEqual(logits.shape, (1, 5, config.vocab_size))

    def test_forward_with_embeddings(self):
        config = _small_qwen2_config()
        lm = LanguageModel(config)

        embeddings = mx.random.normal((1, 5, config.hidden_size))
        logits = lm(input_embeddings=embeddings)
        mx.eval(logits)

        self.assertEqual(logits.shape, (1, 5, config.vocab_size))

    def test_embed_tokens_property(self):
        config = _small_qwen2_config()
        lm = LanguageModel(config)

        self.assertIsInstance(lm.embed_tokens, nn.Embedding)

    def test_layers_property(self):
        config = _small_qwen2_config()
        lm = LanguageModel(config)

        self.assertEqual(len(lm.layers), config.num_hidden_layers)


class TestModelSanitize(unittest.TestCase):
    """Tests for Model.sanitize weight processing."""

    def setUp(self):
        self.config = _small_model_config()
        self.model = Model(self.config)

    def test_removes_model_prefix(self):
        weights = {
            "model.acoustic_tokenizer.encoder.head.conv.weight": mx.zeros((8, 7, 16)),
        }
        sanitized = self.model.sanitize(weights)

        self.assertIn("acoustic_tokenizer.encoder.head.conv.weight", sanitized)
        self.assertNotIn(
            "model.acoustic_tokenizer.encoder.head.conv.weight", sanitized
        )

    def test_skips_decoder_weights(self):
        weights = {
            "model.acoustic_tokenizer.decoder.layers.0.weight": mx.zeros((10, 10)),
            "model.acoustic_tokenizer.encoder.head.conv.weight": mx.zeros((8, 7, 16)),
        }
        sanitized = self.model.sanitize(weights)

        self.assertNotIn(
            "acoustic_tokenizer.decoder.layers.0.weight", sanitized
        )
        self.assertIn("acoustic_tokenizer.encoder.head.conv.weight", sanitized)

    def test_downsample_layers_remap(self):
        weights = {
            "model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight": mx.zeros(
                (4, 7, 1)
            ),
            "model.acoustic_tokenizer.encoder.downsample_layers.1.0.conv.conv.weight": mx.zeros(
                (8, 4, 4)
            ),
        }
        sanitized = self.model.sanitize(weights)

        self.assertIn(
            "acoustic_tokenizer.encoder.downsample_layers.0.conv.weight", sanitized
        )
        self.assertIn(
            "acoustic_tokenizer.encoder.downsample_layers.1.conv.weight", sanitized
        )

    def test_head_conv_remap(self):
        weights = {
            "model.acoustic_tokenizer.encoder.head.conv.conv.weight": mx.zeros(
                (8, 7, 16)
            ),
        }
        sanitized = self.model.sanitize(weights)

        self.assertIn("acoustic_tokenizer.encoder.head.conv.weight", sanitized)

    def test_mixer_conv_remap(self):
        weights = {
            "model.acoustic_tokenizer.encoder.stages.0.0.mixer.conv.conv.conv.weight": mx.zeros(
                (4, 7, 1)
            ),
        }
        sanitized = self.model.sanitize(weights)

        self.assertIn(
            "acoustic_tokenizer.encoder.stages.0.0.mixer.conv.conv.weight", sanitized
        )

    def test_language_model_layers_remap(self):
        weights = {
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.zeros((32, 32)),
        }
        sanitized = self.model.sanitize(weights)

        self.assertIn(
            "language_model.model.layers.0.self_attn.q_proj.weight", sanitized
        )

    def test_language_model_embed_tokens_remap(self):
        weights = {
            "model.language_model.embed_tokens.weight": mx.zeros((1000, 32)),
        }
        sanitized = self.model.sanitize(weights)

        self.assertIn("language_model.model.embed_tokens.weight", sanitized)

    def test_language_model_norm_remap(self):
        weights = {
            "model.language_model.norm.weight": mx.zeros((32,)),
        }
        sanitized = self.model.sanitize(weights)

        self.assertIn("language_model.model.norm.weight", sanitized)

    def test_lm_head_remap(self):
        weights = {
            "model.lm_head.weight": mx.zeros((1000, 32)),
        }
        sanitized = self.model.sanitize(weights)

        self.assertIn("language_model.lm_head.weight", sanitized)

    def test_conv_transpose_pytorch_weights(self):
        """Test that PyTorch conv weights (with model. prefix) are transposed."""
        # PyTorch format: [out, in, kernel] -> MLX format: [out, kernel, in]
        pytorch_weight = mx.zeros((32, 1, 7))  # out=32, in=1, kernel=7
        weights = {
            "model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight": pytorch_weight,
        }
        sanitized = self.model.sanitize(weights)

        key = "acoustic_tokenizer.encoder.downsample_layers.0.conv.weight"
        self.assertEqual(sanitized[key].shape, (32, 7, 1))  # Transposed

    def test_no_transpose_already_converted(self):
        """Test that already-converted MLX weights are NOT transposed."""
        # MLX format weight (no "model." prefix = already converted)
        mlx_weight = mx.zeros((32, 7, 1))  # out=32, kernel=7, in=1
        weights = {
            "acoustic_tokenizer.encoder.downsample_layers.0.conv.weight": mlx_weight,
        }
        sanitized = self.model.sanitize(weights)

        key = "acoustic_tokenizer.encoder.downsample_layers.0.conv.weight"
        self.assertEqual(sanitized[key].shape, (32, 7, 1))  # Unchanged

    def test_skips_position_ids(self):
        weights = {
            "model.language_model.layers.0.position_ids": mx.zeros((1, 128)),
            "model.language_model.norm.weight": mx.zeros((32,)),
        }
        sanitized = self.model.sanitize(weights)

        self.assertNotIn("language_model.model.layers.0.position_ids", sanitized)
        self.assertIn("language_model.model.norm.weight", sanitized)

    def test_skips_fix_std(self):
        weights = {
            "model.acoustic_tokenizer.fix_std": mx.array([0.5]),
        }
        sanitized = self.model.sanitize(weights)

        self.assertNotIn("acoustic_tokenizer.fix_std", sanitized)


class TestModelQuantPredicate(unittest.TestCase):
    """Tests for Model.model_quant_predicate."""

    def setUp(self):
        self.config = _small_model_config()
        self.model = Model(self.config)

    def test_quantize_language_model(self):
        self.assertTrue(
            self.model.model_quant_predicate(
                "language_model.model.layers.0.self_attn.q_proj", None
            )
        )

    def test_skip_acoustic_tokenizer(self):
        self.assertFalse(
            self.model.model_quant_predicate(
                "acoustic_tokenizer.encoder.downsample_layers.0.conv", None
            )
        )

    def test_skip_semantic_tokenizer(self):
        self.assertFalse(
            self.model.model_quant_predicate(
                "semantic_tokenizer.encoder.stages.0.0.ffn.linear1", None
            )
        )

    def test_skip_connectors(self):
        self.assertFalse(
            self.model.model_quant_predicate("acoustic_connector.fc1", None)
        )
        self.assertFalse(
            self.model.model_quant_predicate("semantic_connector.fc2", None)
        )


class TestModel(unittest.TestCase):
    """Tests for the full VibeVoice-ASR Model."""

    def setUp(self):
        self.config = _small_model_config()
        self.model = Model(self.config)

    def test_model_init(self):
        self.assertIsInstance(self.model.acoustic_tokenizer, AcousticTokenizerEncoder)
        self.assertIsInstance(self.model.semantic_tokenizer, SemanticTokenizerEncoder)
        self.assertIsInstance(self.model.acoustic_connector, SpeechConnector)
        self.assertIsInstance(self.model.semantic_connector, SpeechConnector)
        self.assertIsInstance(self.model.language_model, LanguageModel)

    def test_get_input_embeddings(self):
        emb = self.model.get_input_embeddings()
        self.assertIsInstance(emb, nn.Embedding)

    def test_encode_speech_1d(self):
        """Test encode_speech with 1D input [T]."""
        # Small audio: 160 samples
        audio = mx.random.normal((160,))
        features = self.model.encode_speech(audio)
        mx.eval(features)

        # Output should be [1, T', hidden_size]
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[2], self.config.decoder_config.hidden_size)

    def test_encode_speech_2d(self):
        """Test encode_speech with 2D input [B, T]."""
        audio = mx.random.normal((1, 160))
        features = self.model.encode_speech(audio)
        mx.eval(features)

        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[2], self.config.decoder_config.hidden_size)

    def test_encode_speech_3d(self):
        """Test encode_speech with 3D input [B, 1, T]."""
        audio = mx.random.normal((1, 1, 160))
        features = self.model.encode_speech(audio)
        mx.eval(features)

        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[2], self.config.decoder_config.hidden_size)

    def test_language_model_forward(self):
        """Test that language model produces correct logit shape."""
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = self.model.language_model(input_ids)
        mx.eval(logits)

        self.assertEqual(
            logits.shape, (1, 5, self.config.decoder_config.vocab_size)
        )

    def test_merge_speech_text_embeddings(self):
        """Test embedding merging with speech features."""
        seq_len = 10
        speech_len = 4
        hidden = self.config.decoder_config.hidden_size

        input_ids = mx.ones((1, seq_len), dtype=mx.int32)
        speech_features = mx.random.normal((1, speech_len, hidden))
        # Mask: positions 2-5 are speech
        mask = mx.array([[False, False, True, True, True, True, False, False, False, False]])

        embeddings = self.model._merge_speech_text_embeddings(
            input_ids, speech_features, mask
        )
        mx.eval(embeddings)

        self.assertEqual(embeddings.shape, (1, seq_len, hidden))

    def test_merge_no_speech_features(self):
        """Test embedding merging without speech features returns text embeddings."""
        input_ids = mx.array([[1, 2, 3]])
        embeddings = self.model._merge_speech_text_embeddings(input_ids, None, None)
        mx.eval(embeddings)

        self.assertEqual(
            embeddings.shape, (1, 3, self.config.decoder_config.hidden_size)
        )


if __name__ == "__main__":
    unittest.main()
