import unittest

import mlx.core as mx

from mlx_audio.sts.models.moshi.moshi import MoshiConfig, MoshiSTSModel
from mlx_audio.sts.models.moshi_backend import models


class TestMoshi(unittest.TestCase):
    def test_moshi_config(self):
        config = MoshiConfig(quantized=4)
        self.assertEqual(config.quantized, 4)
        self.assertEqual(config.hf_repo, "kyutai/moshiko-mlx-bf16")

    def test_moshi_instantiation(self):
        config = MoshiConfig(quantized=4)
        model = MoshiSTSModel(config)

        # Verify it created the backend LM with correct types
        self.assertIsInstance(model.model, models.Lm)
        self.assertEqual(model.model.cfg.text_out_vocab_size, 32000)
        self.assertEqual(model.model.cfg.audio_codebooks, 16)

        # We can't easily test generation without the weights/tokenizers loaded,
        # but we can verify the API signature exists
        self.assertTrue(hasattr(model, "generate"))
        self.assertTrue(hasattr(model, "load_weights"))
        self.assertTrue(hasattr(model, "from_pretrained"))


if __name__ == "__main__":
    unittest.main()
