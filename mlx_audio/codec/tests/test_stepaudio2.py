import unittest

import mlx.core as mx

from mlx_audio.codec.models.stepaudio2 import DiT


class TestStepAudio2Codec(unittest.TestCase):
    def test_tiny_dit_forward_shape(self):
        model = DiT(
            in_channels=8,
            out_channels=2,
            mlp_ratio=2.0,
            depth=1,
            num_heads=2,
            head_dim=2,
            hidden_size=4,
        )
        x = mx.zeros((1, 2, 3))
        mu = mx.zeros((1, 2, 3))
        spks = mx.zeros((1, 2))
        cond = mx.zeros((1, 2, 3))
        t = mx.array([0.5])
        mask = mx.ones((1, 1, 3), dtype=mx.bool_)
        out = model(x, mask, mu, t, spks, cond)
        self.assertEqual(out.shape, (1, 2, 3))
