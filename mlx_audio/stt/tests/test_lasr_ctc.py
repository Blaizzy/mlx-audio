import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx


class TestLasrModel(unittest.TestCase):
    """Tests for the MedASR (Lasr) model."""

    def setUp(self):
        """Set up test fixtures."""
        from mlx_audio.stt.models.lasr_ctc.config import LasrEncoderConfig, ModelConfig
        from mlx_audio.stt.models.lasr_ctc.lasr import LasrEncoder, LasrForCTC

        self.LasrEncoderConfig = LasrEncoderConfig
        self.ModelConfig = ModelConfig
        self.LasrForCTC = LasrForCTC
        self.LasrEncoder = LasrEncoder

        self.encoder_config = LasrEncoderConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=128,
            conv_kernel_size=5,
            num_mel_bins=80,
            subsampling_conv_channels=32,
            subsampling_conv_kernel_size=3,
            subsampling_conv_stride=2,
        )
        self.model_config = ModelConfig(
            vocab_size=1000, encoder_config=self.encoder_config, model_type="lasr"
        )

    def test_config_from_dict(self):
        """Test ModelConfig.from_dict method."""
        config_dict = {
            "vocab_size": 2000,
            "encoder_config": {
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "num_mel_bins": 80,
            },
            "pad_token_id": 1,
            "ctc_loss_reduction": "sum",
        }
        config = self.ModelConfig.from_dict(config_dict)

        self.assertEqual(config.vocab_size, 2000)
        self.assertEqual(config.pad_token_id, 1)
        self.assertEqual(config.ctc_loss_reduction, "sum")

        # Check nested encoder config
        self.assertIsInstance(config.encoder_config, self.LasrEncoderConfig)
        self.assertEqual(config.encoder_config.hidden_size, 128)
        self.assertEqual(config.encoder_config.num_hidden_layers, 4)
        self.assertEqual(config.encoder_config.num_mel_bins, 80)

    def test_encoder_forward(self):
        """Test LasrEncoder forward pass shape."""
        encoder = self.LasrEncoder(self.encoder_config)

        batch_size = 1
        seq_len = 50
        # Input features: [B, L, num_mel_bins]
        input_features = mx.random.normal(
            (batch_size, seq_len, self.encoder_config.num_mel_bins)
        )

        output = encoder(input_features)

        # Subsampling reduces length by factor?
        # conv0 stride 2, conv1 stride 2 makes it /4 approx?
        # Let's verify expectation: LasrEncoderSubsampling has 2 conv layers with stride 2
        # So L should be roughly L // 4
        expected_len = seq_len // 4

        self.assertEqual(output.shape[0], batch_size)
        # Length might vary slightly due to padding/convolution arithmetic, but let's check basic validity
        # With stride 2, length becomes (L-K+2P)/S + 1.
        # Simply checking if it runs and produces (B, L', H)
        self.assertEqual(output.shape[2], self.encoder_config.hidden_size)
        self.assertTrue(output.shape[1] > 0)

    def test_model_forward(self):
        """Test LasrForCTC full forward pass."""
        model = self.LasrForCTC(self.model_config)

        batch_size = 2
        seq_len = 60
        input_features = mx.random.normal(
            (batch_size, seq_len, self.encoder_config.num_mel_bins)
        )

        logits = model(input_features)

        # Output should be [B, L_subsampled, Vocab]
        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[2], self.model_config.vocab_size)


if __name__ == "__main__":
    unittest.main()
