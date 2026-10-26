import unittest


class TestIndexTTSWeightNorm(unittest.TestCase):
    def test_zero_rows_do_not_produce_nan(self):
        import mlx.core as mx

        from mlx_audio.tts.models.indextts2.indextts2 import Model

        expected = {
            "layer.weight_g": mx.zeros((2, 1, 1), dtype=mx.float16),
            "layer.weight_v": mx.zeros((2, 1, 4), dtype=mx.float16),
        }
        value = mx.zeros((2, 1, 4), dtype=mx.float16)
        rebuilt = Model._split_weightnorm("layer.weight", value, expected)

        self.assertIsNotNone(rebuilt)
        self.assertTrue(bool(mx.all(mx.isfinite(rebuilt["layer.weight_g"]))))
        self.assertTrue(bool(mx.all(mx.isfinite(rebuilt["layer.weight_v"]))))
        self.assertTrue(bool(mx.all(rebuilt["layer.weight_g"] == 0)))

    def test_nonzero_rows_preserve_effective_weight(self):
        import mlx.core as mx

        from mlx_audio.tts.models.indextts2.indextts2 import Model

        value = mx.array([[[1.0, 2.0, 3.0]]], dtype=mx.float32)
        expected = {
            "layer.weight_g": mx.zeros((1, 1, 1), dtype=mx.float32),
            "layer.weight_v": mx.zeros((1, 1, 3), dtype=mx.float32),
        }
        rebuilt = Model._split_weightnorm("layer.weight", value, expected)
        effective = (
            rebuilt["layer.weight_g"]
            * rebuilt["layer.weight_v"]
            / mx.sqrt(
                mx.sum(rebuilt["layer.weight_v"] ** 2, axis=(1, 2), keepdims=True)
            )
        )
        self.assertTrue(bool(mx.allclose(effective, value, atol=1e-6)))


if __name__ == "__main__":
    unittest.main()
