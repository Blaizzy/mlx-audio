import unittest

import mlx.core as mx

from ..models.snac import SNAC

config = {
    "sampling_rate": 24000,
    "encoder_dim": 48,
    "encoder_rates": [2, 4, 8, 8],
    "decoder_dim": 1024,
    "decoder_rates": [8, 8, 4, 2],
    "attn_window_size": None,
    "codebook_size": 4096,
    "codebook_dim": 8,
    "vq_strides": [4, 2, 1],
    "noise": True,
    "depthwise": True,
}


class TestSNAC(unittest.TestCase):
    """Test SNAC model encoding and decoding."""

    def test_snac(self):
        audio = mx.zeros((1, 1, 120_000))

        model = SNAC(**config)
        codes = model.encode(audio)
        self.assertEqual(len(codes), 3)
        self.assertEqual(codes[0].shape, (1, 59))
        self.assertEqual(codes[1].shape, (1, 118))
        self.assertEqual(codes[2].shape, (1, 236))

        reconstructed = model.decode(codes).squeeze(-1)
        self.assertEqual(reconstructed.shape, (1, 120_907))


if __name__ == "__main__":
    unittest.main()
