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


class TestCAMPPlusPooling(unittest.TestCase):
    def test_partial_segments_exclude_padding(self):
        from mlx_audio.codec.models.stepaudio2.speaker import FusedCAMLayer

        layer = FusedCAMLayer(2, 2, 3, stride=1, padding=1, dilation=1, bias=False)
        for length in (1, 99, 100, 101, 103, 200, 201):
            x = mx.arange(length, dtype=mx.float32)[None, None]
            actual = layer.seg_pooling(x)
            expected = []
            for start in range(0, length, 100):
                end = min(start + 100, length)
                expected.extend([(start + end - 1) / 2] * (end - start))
            self.assertEqual(actual.shape, x.shape)
            self.assertTrue(bool(mx.allclose(actual, mx.array(expected)[None, None])))

    def test_statistics_use_sample_standard_deviation_without_epsilon(self):
        from mlx_audio.codec.models.stepaudio2.speaker import UnbiasedStatsPool

        x = mx.array([[[1.0, 3.0, 5.0], [7.0, 7.0, 7.0]]])
        actual = UnbiasedStatsPool()(x)
        self.assertTrue(bool(mx.allclose(actual, mx.array([[3.0, 7.0, 2.0, 0.0]]))))
        with self.assertRaisesRegex(ValueError, "two frames"):
            UnbiasedStatsPool()(x[..., :1])
