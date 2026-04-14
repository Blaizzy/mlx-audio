import unittest

import mlx.core as mx

from mlx_audio.stt.models.mms.mms import Model
from mlx_audio.stt.models.wav2vec.wav2vec import ModelConfig


def _small_config():
    return ModelConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        conv_dim=(16, 16),
        conv_stride=(2, 2),
        conv_kernel=(4, 3),
        num_feat_extract_layers=2,
        num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=4,
    )


class TestConfig(unittest.TestCase):

    def test_defaults(self):
        config = ModelConfig()
        self.assertEqual(config.model_type, "wav2vec2")
        self.assertEqual(config.hidden_size, 768)

    def test_from_dict(self):
        d = {"hidden_size": 1024, "num_hidden_layers": 24}
        config = ModelConfig.from_dict(d)
        self.assertEqual(config.hidden_size, 1024)


class TestCTCDecode(unittest.TestCase):

    def test_greedy_decode(self):
        config = _small_config()
        model = Model(config)
        logits = mx.zeros((1, 10, 32))
        logits = logits.at[0, 0, 5].add(10.0)
        logits = logits.at[0, 1, 5].add(10.0)
        logits = logits.at[0, 2, 8].add(10.0)
        logits = logits.at[0, 3, 0].add(10.0)  # blank
        logits = logits.at[0, 4, 8].add(10.0)
        decoded = model._ctc_decode(logits)
        self.assertEqual(decoded[0], [5, 8, 8])

    def test_all_blanks(self):
        config = _small_config()
        model = Model(config)
        logits = mx.zeros((1, 5, 32))
        decoded = model._ctc_decode(logits)
        self.assertEqual(decoded[0], [])


class TestTokensToText(unittest.TestCase):

    def test_with_vocab(self):
        config = _small_config()
        model = Model(config)
        model._vocab = {1: "h", 2: "e", 3: "l", 4: "o", 5: "|"}
        text = model._tokens_to_text([1, 2, 3, 3, 4, 5, 1, 2])
        self.assertEqual(text, "hello he")

    def test_without_vocab(self):
        config = _small_config()
        model = Model(config)
        text = model._tokens_to_text([1, 2, 3])
        self.assertEqual(text, "1 2 3")


class TestModelSanitize(unittest.TestCase):

    def setUp(self):
        self.config = _small_config()
        self.model = Model(self.config)

    def test_keeps_lm_head(self):
        weights = {"lm_head.weight": mx.zeros((32, 32))}
        sanitized = self.model.sanitize(weights)
        self.assertIn("lm_head.weight", sanitized)

    def test_keeps_wav2vec2_prefix(self):
        weights = {
            "wav2vec2.encoder.layers.0.attention.q_proj.weight": mx.zeros((32, 32))
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("wav2vec2.encoder.layers.0.attention.q_proj.weight", sanitized)

    def test_conv_transpose(self):
        weights = {
            "wav2vec2.feature_extractor.conv_layers.0.conv.weight": mx.zeros((16, 1, 4))
        }
        sanitized = self.model.sanitize(weights)
        key = "wav2vec2.feature_extractor.conv_layers.0.conv.weight"
        self.assertEqual(sanitized[key].shape, (16, 4, 1))

    def test_skips_quantizer(self):
        weights = {"quantizer.weight_proj.weight": mx.zeros((32, 32))}
        sanitized = self.model.sanitize(weights)
        self.assertEqual(len(sanitized), 0)

    def test_skips_masked_spec(self):
        weights = {"masked_spec_embed": mx.zeros((32,))}
        sanitized = self.model.sanitize(weights)
        self.assertEqual(len(sanitized), 0)


class TestModel(unittest.TestCase):

    def setUp(self):
        self.config = _small_config()
        self.model = Model(self.config)

    def test_init(self):
        self.assertIsNotNone(self.model.wav2vec2)
        self.assertIsNotNone(self.model.lm_head)

    def test_sample_rate(self):
        self.assertEqual(self.model.sample_rate, 16000)

    def test_forward(self):
        audio = mx.random.normal((1, 320))
        logits = self.model(audio)
        mx.eval(logits)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[2], 32)


if __name__ == "__main__":
    unittest.main()
