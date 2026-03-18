"""Tests for Granite Speech model components."""

import unittest

import mlx.core as mx
import numpy as np


class TestGraniteSpeechConfig(unittest.TestCase):
    """Tests for config parsing."""

    def test_model_config_from_dict(self):
        from mlx_audio.stt.models.granite_speech.config import ModelConfig

        config_dict = {
            "model_type": "granite_speech",
            "audio_token_index": 100352,
            "downsample_rate": 5,
            "window_size": 15,
            "encoder_config": {
                "model_type": "granite_speech_encoder",
                "num_layers": 16,
                "hidden_dim": 1024,
                "input_dim": 160,
                "output_dim": 348,
            },
            "projector_config": {
                "model_type": "blip_2_qformer",
                "num_hidden_layers": 2,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
            },
            "text_config": {
                "model_type": "granite",
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "vocab_size": 100353,
            },
            "extra_field": "should_be_ignored",
        }

        config = ModelConfig.from_dict(config_dict)
        self.assertEqual(config.model_type, "granite_speech")
        self.assertEqual(config.audio_token_index, 100352)
        self.assertEqual(config.encoder_config.num_layers, 16)
        self.assertEqual(config.encoder_config.hidden_dim, 1024)
        self.assertEqual(config.projector_config.num_hidden_layers, 2)
        self.assertEqual(config.projector_config.hidden_size, 1024)
        self.assertEqual(config.text_config.hidden_size, 2048)
        self.assertEqual(config.text_config.vocab_size, 100353)

    def test_nested_config_defaults(self):
        from mlx_audio.stt.models.granite_speech.config import ModelConfig

        config = ModelConfig()
        self.assertIsNotNone(config.encoder_config)
        self.assertIsNotNone(config.projector_config)
        self.assertIsNotNone(config.text_config)
        self.assertEqual(config.encoder_config.hidden_dim, 1024)


class TestBatchNorm1d(unittest.TestCase):
    """Tests for inference-only BatchNorm1d."""

    def test_output_shape(self):
        from mlx_audio.stt.models.granite_speech.conformer import BatchNorm1d

        bn = BatchNorm1d(64)
        x = mx.random.normal((2, 10, 64))
        out = bn(x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_normalization(self):
        from mlx_audio.stt.models.granite_speech.conformer import BatchNorm1d

        bn = BatchNorm1d(4)
        bn.running_mean = mx.array([1.0, 2.0, 3.0, 4.0])
        bn.running_var = mx.array([1.0, 1.0, 1.0, 1.0])
        bn.weight = mx.ones((4,))
        bn.bias = mx.zeros((4,))

        x = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
        out = bn(x)
        mx.eval(out)  # noqa: S307 - MLX array materialization
        np.testing.assert_allclose(np.array(out[0, 0]), [0.0, 0.0, 0.0, 0.0], atol=1e-5)


class TestConformerComponents(unittest.TestCase):
    """Tests for conformer building blocks."""

    def _small_config(self):
        from mlx_audio.stt.models.granite_speech.config import EncoderConfig

        return EncoderConfig(
            num_layers=2,
            hidden_dim=64,
            input_dim=16,
            output_dim=32,
            num_heads=4,
            dim_head=16,
            feedforward_mult=2,
            conv_kernel_size=3,
            conv_expansion_factor=2,
            context_size=20,
            max_pos_emb=10,
        )

    def test_feedforward_shape(self):
        from mlx_audio.stt.models.granite_speech.conformer import ConformerFeedForward

        ff = ConformerFeedForward(64, mult=2)
        x = mx.random.normal((1, 10, 64))
        out = ff(x)
        self.assertEqual(out.shape, (1, 10, 64))

    def test_attention_shape(self):
        from mlx_audio.stt.models.granite_speech.conformer import ConformerAttention

        config = self._small_config()
        attn = ConformerAttention(config)
        x = mx.random.normal((1, 10, 64))
        out = attn(x)
        self.assertEqual(out.shape, (1, 10, 64))

    def test_attention_block_mode(self):
        from mlx_audio.stt.models.granite_speech.conformer import ConformerAttention

        config = self._small_config()
        config.context_size = 5
        attn = ConformerAttention(config)
        # Sequence longer than context_size triggers block attention
        x = mx.random.normal((1, 12, 64))
        out = attn(x)
        self.assertEqual(out.shape, (1, 12, 64))

    def test_conv_module_shape(self):
        from mlx_audio.stt.models.granite_speech.conformer import ConformerConvModule

        config = self._small_config()
        conv = ConformerConvModule(config)
        x = mx.random.normal((1, 10, 64))
        out = conv(x)
        self.assertEqual(out.shape, (1, 10, 64))

    def test_conformer_block_shape(self):
        from mlx_audio.stt.models.granite_speech.conformer import ConformerBlock

        config = self._small_config()
        block = ConformerBlock(config)
        x = mx.random.normal((1, 10, 64))
        out = block(x)
        self.assertEqual(out.shape, (1, 10, 64))

    def test_ctc_encoder_shape(self):
        from mlx_audio.stt.models.granite_speech.conformer import CTCEncoder

        config = self._small_config()
        encoder = CTCEncoder(config)
        x = mx.random.normal((1, 20, 16))
        out = encoder(x)
        self.assertEqual(out.shape, (1, 20, 64))


class TestQFormerComponents(unittest.TestCase):
    """Tests for Q-Former building blocks."""

    def _small_config(self):
        from mlx_audio.stt.models.granite_speech.config import ProjectorConfig

        return ProjectorConfig(
            num_hidden_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            intermediate_size=128,
            encoder_hidden_size=64,
        )

    def test_qformer_attention_self(self):
        from mlx_audio.stt.models.granite_speech.qformer import QFormerAttention

        config = self._small_config()
        attn = QFormerAttention(config)
        x = mx.random.normal((1, 3, 64))
        out = attn(x)
        self.assertEqual(out.shape, (1, 3, 64))

    def test_qformer_attention_cross(self):
        from mlx_audio.stt.models.granite_speech.qformer import QFormerAttention

        config = self._small_config()
        attn = QFormerAttention(config)
        q = mx.random.normal((1, 3, 64))
        kv = mx.random.normal((1, 15, 64))
        out = attn(q, kv)
        self.assertEqual(out.shape, (1, 3, 64))

    def test_qformer_layer_shape(self):
        from mlx_audio.stt.models.granite_speech.qformer import QFormerLayer

        config = self._small_config()
        layer = QFormerLayer(config)
        q = mx.random.normal((1, 3, 64))
        enc = mx.random.normal((1, 15, 64))
        out = layer(q, enc)
        self.assertEqual(out.shape, (1, 3, 64))

    def test_encoder_projector_shape(self):
        from mlx_audio.stt.models.granite_speech.qformer import EncoderProjector

        config = self._small_config()
        proj = EncoderProjector(
            config=config,
            window_size=10,
            downsample_rate=5,
            num_queries=2,
            output_dim=32,
        )
        enc_out = mx.random.normal((1, 20, 64))
        out = proj(enc_out)
        mx.eval(out)  # noqa: S307 - MLX array materialization
        # num_windows = ceil(20 / 5) = 4, output tokens = 4 * 2 = 8
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[2], 32)
        self.assertTrue(out.shape[1] > 0)


class TestWeightSanitization(unittest.TestCase):
    """Tests for weight key remapping and conv transposition."""

    def _make_model(self):
        from mlx_audio.stt.models.granite_speech.config import (
            EncoderConfig,
            ModelConfig,
            ProjectorConfig,
            TextConfig,
        )

        config = ModelConfig(
            encoder_config=EncoderConfig(
                num_layers=1,
                hidden_dim=32,
                input_dim=8,
                output_dim=16,
                num_heads=2,
                dim_head=16,
                feedforward_mult=2,
                conv_kernel_size=3,
                conv_expansion_factor=2,
                context_size=10,
                max_pos_emb=5,
            ),
            projector_config=ProjectorConfig(
                num_hidden_layers=1,
                hidden_size=32,
                num_attention_heads=2,
                intermediate_size=64,
                encoder_hidden_size=32,
            ),
            text_config=TextConfig(
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=128,
                vocab_size=100,
                logits_scaling=1.0,
                attention_multiplier=1.0,
                embedding_multiplier=1.0,
                residual_multiplier=1.0,
                max_position_embeddings=128,
            ),
        )
        from mlx_audio.stt.models.granite_speech.granite_speech import Model

        return Model(config)

    def test_skip_num_batches_tracked(self):
        model = self._make_model()
        weights = {
            "encoder.layers.0.conv.batch_norm.num_batches_tracked": mx.array(100),
            "encoder.layers.0.conv.batch_norm.weight": mx.ones((64,)),
        }
        sanitized = model.sanitize(weights)
        self.assertNotIn(
            "encoder.layers.0.conv.batch_norm.num_batches_tracked", sanitized
        )
        self.assertIn("encoder.layers.0.conv.batch_norm.weight", sanitized)

    def test_conv_weight_transposition(self):
        model = self._make_model()
        # PyTorch conv weight: [O, I, K] = (32, 16, 3)
        pt_weight = mx.random.normal((32, 16, 3))
        weights = {"encoder.layers.0.conv.up_conv.weight": pt_weight}
        sanitized = model.sanitize(weights)
        result = sanitized["encoder.layers.0.conv.up_conv.weight"]
        # MLX conv weight: [O, K, I] = (32, 3, 16)
        self.assertEqual(result.shape, (32, 3, 16))

    def test_non_conv_weights_unchanged(self):
        model = self._make_model()
        weight = mx.random.normal((64, 32))
        weights = {"encoder.layers.0.ff1.up_proj.weight": weight}
        sanitized = model.sanitize(weights)
        np.testing.assert_array_equal(
            np.array(sanitized["encoder.layers.0.ff1.up_proj.weight"]),
            np.array(weight),
        )


class TestQuantizationPredicate(unittest.TestCase):
    """Tests for model_quant_predicate."""

    def test_language_model_quantized(self):
        from mlx_audio.stt.models.granite_speech.config import (
            EncoderConfig,
            ModelConfig,
            ProjectorConfig,
            TextConfig,
        )
        from mlx_audio.stt.models.granite_speech.granite_speech import Model

        config = ModelConfig(
            encoder_config=EncoderConfig(
                num_layers=1,
                hidden_dim=32,
                input_dim=8,
                output_dim=16,
                num_heads=2,
                dim_head=16,
                feedforward_mult=2,
                conv_kernel_size=3,
                conv_expansion_factor=2,
                context_size=10,
                max_pos_emb=5,
            ),
            projector_config=ProjectorConfig(
                num_hidden_layers=1,
                hidden_size=32,
                num_attention_heads=2,
                intermediate_size=64,
                encoder_hidden_size=32,
            ),
            text_config=TextConfig(
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=128,
                vocab_size=100,
                logits_scaling=1.0,
                attention_multiplier=1.0,
                embedding_multiplier=1.0,
                residual_multiplier=1.0,
                max_position_embeddings=128,
            ),
        )
        model = Model(config)

        self.assertTrue(
            model.model_quant_predicate("language_model.model.layers.0", None)
        )
        self.assertTrue(model.model_quant_predicate("language_model.lm_head", None))
        self.assertFalse(model.model_quant_predicate("encoder.layers.0", None))
        self.assertFalse(model.model_quant_predicate("projector.linear", None))


class TestFullModelForward(unittest.TestCase):
    """Test full model forward pass with small config."""

    def test_forward_pass(self):
        from mlx_audio.stt.models.granite_speech.config import (
            EncoderConfig,
            ModelConfig,
            ProjectorConfig,
            TextConfig,
        )
        from mlx_audio.stt.models.granite_speech.granite_speech import Model

        config = ModelConfig(
            window_size=10,
            downsample_rate=5,
            audio_token_index=99,
            encoder_config=EncoderConfig(
                num_layers=1,
                hidden_dim=32,
                input_dim=8,
                output_dim=16,
                num_heads=2,
                dim_head=16,
                feedforward_mult=2,
                conv_kernel_size=3,
                conv_expansion_factor=2,
                context_size=10,
                max_pos_emb=5,
            ),
            projector_config=ProjectorConfig(
                num_hidden_layers=1,
                hidden_size=32,
                num_attention_heads=2,
                intermediate_size=64,
                encoder_hidden_size=32,
            ),
            text_config=TextConfig(
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=128,
                vocab_size=100,
                logits_scaling=1.0,
                attention_multiplier=1.0,
                embedding_multiplier=1.0,
                residual_multiplier=1.0,
                max_position_embeddings=128,
            ),
        )
        model = Model(config)

        # Simulate: encoder input -> projector -> merge with text -> LLM forward
        enc_input = mx.random.normal((1, 20, 8))
        encoder_out = model.encoder(enc_input)
        self.assertEqual(encoder_out.shape, (1, 20, 32))

        audio_embeds = model.projector(encoder_out)
        mx.eval(audio_embeds)  # noqa: S307 - MLX array materialization
        self.assertEqual(audio_embeds.shape[0], 1)
        self.assertEqual(audio_embeds.shape[2], 64)  # LLM hidden size
        num_audio_tokens = audio_embeds.shape[1]

        # Build input_ids with audio placeholders
        prefix = [1, 2, 3]
        audio_placeholder = [99] * num_audio_tokens
        suffix = [4, 5]
        input_ids = mx.array([prefix + audio_placeholder + suffix])

        logits = model(input_ids, audio_embeds=audio_embeds)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], len(prefix) + num_audio_tokens + len(suffix))
        self.assertEqual(logits.shape[2], 100)  # vocab_size


if __name__ == "__main__":
    unittest.main()
