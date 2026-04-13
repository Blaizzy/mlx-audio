import unittest

import mlx.core as mx

from ..models.fish_s1_dac.fish_s1_dac import (
    DAC,
    DownsampleResidualVectorQuantize,
    Identity,
)


class TestFishS1DAC(unittest.TestCase):
    def test_tiny_encode_decode(self):
        quantizer = DownsampleResidualVectorQuantize(
            input_dim=16,
            n_codebooks=2,
            codebook_dim=4,
            codebook_size=16,
            semantic_codebook_size=32,
            downsample_factor=(2,),
            pre_module=Identity(),
            post_module=Identity(),
        )

        model = DAC(
            encoder_dim=4,
            encoder_rates=[2, 2],
            latent_dim=16,
            decoder_dim=16,
            decoder_rates=[2, 2],
            quantizer=quantizer,
            sample_rate=44100,
            causal=True,
            encoder_transformer_layers=[0, 0],
            decoder_transformer_layers=[0, 0],
            transformer_general_config=lambda **kw: None,
        )

        audio = mx.zeros((1, 1, 128), dtype=mx.float32)
        indices, feature_lengths = model.encode(audio)
        self.assertEqual(indices.shape[0], 1)
        self.assertEqual(indices.shape[1], 3)  # semantic + residual quantizers

        decoded, decoded_lengths = model.decode(indices, feature_lengths)
        self.assertEqual(tuple(decoded.shape), (1, 1, 128))
        self.assertEqual(int(decoded_lengths[0]), 128)

        z_q = model.encode_zq(audio)
        self.assertEqual(tuple(z_q.shape), (1, 16, 16))
        recon = model.decode_zq(z_q)
        self.assertEqual(tuple(recon.shape), (1, 1, 128))


if __name__ == "__main__":
    unittest.main()
