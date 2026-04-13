import unittest

import mlx.core as mx

from mlx_audio.stt.models.moonshine.config import ModelConfig
from mlx_audio.stt.models.moonshine.moonshine import (
    Model,
    MoonshineAttention,
    MoonshineDecoder,
    MoonshineDecoderLayer,
    MoonshineDecoderMLP,
    MoonshineEncoder,
    MoonshineEncoderLayer,
    MoonshineEncoderMLP,
)


def _small_config():
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        encoder_num_hidden_layers=2,
        decoder_num_hidden_layers=2,
        encoder_num_attention_heads=4,
        decoder_num_attention_heads=4,
        partial_rotary_factor=0.5,
    )


class TestConfig(unittest.TestCase):

    def test_defaults(self):
        config = ModelConfig()
        self.assertEqual(config.vocab_size, 32768)
        self.assertEqual(config.hidden_size, 288)
        self.assertEqual(config.encoder_num_hidden_layers, 6)
        self.assertEqual(config.decoder_num_hidden_layers, 6)

    def test_kv_heads_default(self):
        config = ModelConfig(encoder_num_attention_heads=8)
        self.assertEqual(config.encoder_num_key_value_heads, 8)

    def test_from_dict(self):
        d = {"vocab_size": 1000, "hidden_size": 64, "encoder_num_hidden_layers": 4}
        config = ModelConfig.from_dict(d)
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.encoder_num_hidden_layers, 4)

    def test_from_dict_ignores_extra(self):
        d = {"vocab_size": 1000, "unknown_field": True}
        config = ModelConfig.from_dict(d)
        self.assertEqual(config.vocab_size, 1000)


class TestEncoderMLP(unittest.TestCase):

    def test_output_shape(self):
        mlp = MoonshineEncoderMLP(32, 64)
        x = mx.random.normal((1, 10, 32))
        out = mlp(x)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 10, 32))


class TestDecoderMLP(unittest.TestCase):

    def test_output_shape(self):
        mlp = MoonshineDecoderMLP(32, 64)
        x = mx.random.normal((1, 10, 32))
        out = mlp(x)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 10, 32))


class TestAttention(unittest.TestCase):

    def test_self_attention(self):
        attn = MoonshineAttention(32, 4, 4)
        x = mx.random.normal((1, 10, 32))
        out, cache = attn(x)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 10, 32))

    def test_cross_attention(self):
        attn = MoonshineAttention(32, 4, 4)
        x = mx.random.normal((1, 5, 32))
        enc = mx.random.normal((1, 20, 32))
        out, cache = attn(x, encoder_hidden_states=enc)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 5, 32))

    def test_causal_self_attention(self):
        attn = MoonshineAttention(32, 4, 4, is_causal=True)
        x = mx.random.normal((1, 10, 32))
        out, cache = attn(x)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 10, 32))

    def test_cache_update(self):
        attn = MoonshineAttention(32, 4, 4, is_causal=True)
        x1 = mx.random.normal((1, 5, 32))
        out1, cache1 = attn(x1)
        mx.eval(out1)

        x2 = mx.random.normal((1, 1, 32))
        out2, cache2 = attn(x2, cache=cache1)
        mx.eval(out2)
        self.assertEqual(out2.shape, (1, 1, 32))
        self.assertEqual(cache2[0].shape[2], 6)


class TestEncoderLayer(unittest.TestCase):

    def test_output_shape(self):
        config = _small_config()
        layer = MoonshineEncoderLayer(config)
        x = mx.random.normal((1, 10, 32))
        out = layer(x)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 10, 32))


class TestDecoderLayer(unittest.TestCase):

    def test_output_shape(self):
        config = _small_config()
        layer = MoonshineDecoderLayer(config)
        x = mx.random.normal((1, 5, 32))
        enc = mx.random.normal((1, 20, 32))
        out, self_cache, cross_cache = layer(x, enc)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 5, 32))


class TestEncoder(unittest.TestCase):

    def test_output_shape(self):
        config = _small_config()
        encoder = MoonshineEncoder(config)
        audio = mx.random.normal((1, 16000))
        out = encoder(audio)
        mx.eval(out)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[2], 32)
        self.assertTrue(out.shape[1] > 0)

    def test_1d_input(self):
        config = _small_config()
        encoder = MoonshineEncoder(config)
        audio = mx.random.normal((16000,))
        out = encoder(audio)
        mx.eval(out)
        self.assertEqual(out.shape[0], 1)


class TestDecoder(unittest.TestCase):

    def test_output_shape(self):
        config = _small_config()
        decoder = MoonshineDecoder(config)
        tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
        enc = mx.random.normal((1, 20, 32))
        out, cache = decoder(tokens, enc)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 3, 32))
        self.assertEqual(len(cache), 2)

    def test_autoregressive(self):
        config = _small_config()
        decoder = MoonshineDecoder(config)
        enc = mx.random.normal((1, 20, 32))

        t1 = mx.array([[1]], dtype=mx.int32)
        out1, cache1 = decoder(t1, enc)
        mx.eval(out1)

        t2 = mx.array([[2]], dtype=mx.int32)
        out2, cache2 = decoder(t2, enc, cache=cache1)
        mx.eval(out2)
        self.assertEqual(out2.shape, (1, 1, 32))


class TestModelSanitize(unittest.TestCase):

    def setUp(self):
        self.config = _small_config()
        self.model = Model(self.config)

    def test_encoder_key_mapping(self):
        weights = {"model.encoder.layers.0.self_attn.q_proj.weight": mx.zeros((32, 32))}
        sanitized = self.model.sanitize(weights)
        self.assertIn("encoder.layers.0.self_attn.q_proj.weight", sanitized)

    def test_decoder_key_mapping(self):
        weights = {"model.decoder.layers.0.self_attn.q_proj.weight": mx.zeros((32, 32))}
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.layers.0.self_attn.q_proj.weight", sanitized)

    def test_cross_attn_mapping(self):
        weights = {
            "model.decoder.layers.0.encoder_attn.q_proj.weight": mx.zeros((32, 32))
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("decoder.layers.0.encoder_attn.q_proj.weight", sanitized)

    def test_conv_transpose(self):
        weights = {"model.encoder.conv1.weight": mx.zeros((32, 1, 127))}
        sanitized = self.model.sanitize(weights)
        self.assertEqual(sanitized["encoder.conv1.weight"].shape, (32, 127, 1))

    def test_tied_weights_skip_proj(self):
        config = _small_config()
        config.tie_word_embeddings = True
        model = Model(config)
        weights = {"proj_out.weight": mx.zeros((64, 32))}
        sanitized = model.sanitize(weights)
        self.assertNotIn("proj_out.weight", sanitized)

    def test_encoder_group_norm(self):
        weights = {"model.encoder.groupnorm.weight": mx.zeros((32,))}
        sanitized = self.model.sanitize(weights)
        self.assertIn("encoder.groupnorm.weight", sanitized)


class TestModel(unittest.TestCase):

    def setUp(self):
        self.config = _small_config()
        self.model = Model(self.config)

    def test_init(self):
        self.assertIsInstance(self.model.encoder, MoonshineEncoder)
        self.assertIsInstance(self.model.decoder, MoonshineDecoder)

    def test_sample_rate(self):
        self.assertEqual(self.model.sample_rate, 16000)

    def test_logits_tied(self):
        config = _small_config()
        config.tie_word_embeddings = True
        model = Model(config)
        hidden = mx.random.normal((1, 1, 32))
        logits = model._get_logits(hidden)
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 1, 64))


if __name__ == "__main__":
    unittest.main()
