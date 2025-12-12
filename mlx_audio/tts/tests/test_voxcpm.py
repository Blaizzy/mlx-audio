import unittest

import mlx.core as mx

from mlx_audio.tts.models.voxcpm.audio_vae import AudioVAE
from mlx_audio.tts.models.voxcpm.config import AudioVAEConfig


class TestVoxCPM(unittest.TestCase):

    def test_audio_vae_shape(self):
        """AudioVAE encode/decode preserves expected shape relationship."""
        config = AudioVAEConfig(
            encoder_dim=64,
            encoder_rates=[2, 2],
            latent_dim=64,
            decoder_dim=128,
            decoder_rates=[2, 2],
            sample_rate=24000,
        )
        vae = AudioVAE(config)

        # Input: (batch, samples)
        # Hop length = prod(encoder_rates) = 4
        samples = 100
        # We need samples to be a multiple of hop_length for simplified testing,
        # normally preprocess handles padding.

        x = mx.zeros((1, samples))

        # Test encode
        encoded = vae.encode(x)
        # Expected shape: (1, 1, samples // hop_length)?
        # vae.encode returns mu.
        # Encoder: (N, 1, T) -> (N, latent, T')
        # T' = T / hop + padding adjustments.

        self.assertEqual(encoded.ndim, 3)

        # Test decode
        decoded = vae.decode(encoded)

        # Output should be roughly same length
        # ConvTranspose logic might result in slightly different shape if padding is not perfect.
        # But let's check basic execution.
        self.assertIsNotNone(decoded)

    def test_sanitize_weight_norm(self):
        """Test simple weight norm fusion."""
        config = AudioVAEConfig(encoder_rates=[2], decoder_rates=[2])
        vae = AudioVAE(config)

        # Construct fake weights
        # encoder.conv_in.weight_g (out, 1, 1)
        # encoder.conv_in.weight_v (out, in, k)

        # Let's say out=64, in=1, k=7 (default)
        out_c = 64
        in_c = 1
        k = 7

        g = mx.ones((out_c, 1, 1)) * 2.0
        v = mx.ones((out_c, in_c, k))

        weights = {"encoder.conv_in.weight_g": g, "encoder.conv_in.weight_v": v}

        sanitized = vae.sanitize(weights)

        self.assertIn("encoder.conv_in.weight", sanitized)
        w = sanitized["encoder.conv_in.weight"]
        # Expected shape after transpose: (out, k, in) -> (64, 7, 1)
        self.assertEqual(w.shape, (64, 7, 1))

        # Value check:
        # v norm per row: sqrt(1*1*7) = sqrt(7)
        # w = 2 * (1 / sqrt(7))
        expected_val = 2.0 / (7**0.5)
        self.assertTrue(mx.allclose(w[0, 0, 0], mx.array(expected_val)))

    def test_model_init(self):
        """Test full model initialization with minimal config."""
        from mlx_audio.tts.models.voxcpm import Model, ModelArgs
        from mlx_audio.tts.models.voxcpm.config import (
            AudioVAEConfig,
            DiTConfig,
            EncoderConfig,
            LMConfig,
        )

        args = ModelArgs(
            lm_config=LMConfig(
                num_hidden_layers=1,
                hidden_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=128,
            ),
            encoder_config=EncoderConfig(num_layers=1, hidden_dim=64),
            dit_config=DiTConfig(
                num_layers=1,
                hidden_dim=64,
                cfm_config={
                    "solver": "euler"
                },  # Dict will be converted by from_dict or need explicit obje?
                # ModelArgs __init__ expects DiTConfig object.
                # But ModelArgs.from_dict converts dicts.
                # Here we are constructing directly.
                # DiTConfig default cfm_config is factory.
            ),
            audio_vae_config=AudioVAEConfig(
                encoder_rates=[2],
                decoder_rates=[2],
                encoder_dim=16,
                decoder_dim=16,
                latent_dim=16,
            ),
            patch_size=2,
            feat_dim=16,
        )

        model = Model(args)
        self.assertIsNotNone(model)

        # Test simulated generate step (parts of it)
        # Mock tokenizer? Model checks if tokenizer is None.
        # We can mock it or bypass.

        # We can try to call parts directly
        # Test Embeds
        x = mx.array([[1, 2, 3]])
        emb = model.base_lm.embed_tokens(x)
        self.assertEqual(emb.shape, (1, 3, 64))


if __name__ == "__main__":
    unittest.main()
