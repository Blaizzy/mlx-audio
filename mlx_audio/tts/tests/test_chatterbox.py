import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np


class TestChatterboxConfig(unittest.TestCase):
    def test_t3_config_defaults(self):
        """Test T3Config default values and factory methods."""
        from mlx_audio.tts.models.chatterbox.config import T3Config

        # Test defaults
        config = T3Config()
        self.assertEqual(config.text_tokens_dict_size, 704)
        self.assertEqual(config.speech_tokens_dict_size, 8194)
        self.assertEqual(config.llama_config_name, "Llama_520M")
        self.assertEqual(config.n_channels, 1024)
        self.assertFalse(config.is_multilingual)

        # Test factory methods
        self.assertFalse(T3Config.english_only().is_multilingual)
        self.assertTrue(T3Config.multilingual().is_multilingual)

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        from mlx_audio.tts.models.chatterbox.config import ModelConfig

        config = ModelConfig()

        self.assertEqual(config.model_type, "chatterbox")
        self.assertEqual(config.s3_sr, 16000)
        self.assertEqual(config.s3gen_sr, 24000)
        self.assertEqual(config.sample_rate, 24000)
        self.assertIsNotNone(config.t3_config)

    def test_model_config_from_dict(self):
        """Test ModelConfig.from_dict method."""
        from mlx_audio.tts.models.chatterbox.config import ModelConfig

        config_dict = {
            "model_type": "chatterbox",
            "t3_config": {
                "text_tokens_dict_size": 2454,
            },
        }

        config = ModelConfig.from_dict(config_dict)

        self.assertEqual(config.model_type, "chatterbox")
        self.assertTrue(config.t3_config.is_multilingual)


class TestChatterboxModel(unittest.TestCase):
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.T3")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3Token2Wav")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3TokenizerV2")
    def test_init(self, mock_s3_tokenizer, mock_ve, mock_s3gen, mock_t3):
        """Test Model initialization with config."""
        from mlx_audio.tts.models.chatterbox.chatterbox import Model
        from mlx_audio.tts.models.chatterbox.config import ModelConfig

        config = ModelConfig()
        model = Model(config)

        self.assertIsNotNone(model.t3)
        self.assertIsNotNone(model.s3gen)
        self.assertIsNotNone(model.ve)
        self.assertEqual(model.sr, 24000)
        self.assertEqual(model.sample_rate, 24000)

    @patch("mlx_audio.tts.models.chatterbox.chatterbox.T3")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3Token2Wav")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3TokenizerV2")
    def test_sanitize(
        self, mock_s3_tokenizer, mock_ve_class, mock_s3gen_class, mock_t3_class
    ):
        """Test weight sanitization routes to correct components."""
        from mlx_audio.tts.models.chatterbox.chatterbox import Model

        # Mock components to have sanitize methods that pass through weights
        for mock_class in [
            mock_ve_class,
            mock_t3_class,
            mock_s3gen_class,
            mock_s3_tokenizer,
        ]:
            mock_class.return_value.sanitize.side_effect = lambda w: w

        model = Model()

        # Test that prefixed weights are routed and re-prefixed
        weights = {
            "ve.lstm.weight": mx.zeros((10, 10)),
            "t3.tfmr.weight": mx.zeros((10, 10)),
            "s3gen.flow.weight": mx.zeros((10, 10)),
        }

        result = model.sanitize(weights)

        # Verify weights keep their prefixes
        self.assertIn("ve.lstm.weight", result)
        self.assertIn("t3.tfmr.weight", result)
        self.assertIn("s3gen.flow.weight", result)


if __name__ == "__main__":
    unittest.main()
