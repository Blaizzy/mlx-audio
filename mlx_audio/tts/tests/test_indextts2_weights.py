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


class TestIndexTTSEmotionControls(unittest.TestCase):
    def test_emotion_weight_normalization(self):
        from mlx_audio.tts.models.indextts2.indextts2 import Model

        weights = Model._normalize_emotion_weights([1.0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(weights["happy"], 1.0)
        self.assertEqual(sum(weights.values()), 1.0)

        weights = Model._normalize_emotion_weights({"高兴": 0.6, "自然": 0.2})
        self.assertAlmostEqual(weights["happy"], 0.6)
        self.assertAlmostEqual(weights["calm"], 0.2)

        with self.assertRaises(ValueError):
            Model._normalize_emotion_weights([1.0, 0.0])

    def test_emotion_matrix_mix_uses_style_nearest_rows(self):
        from types import SimpleNamespace

        import mlx.core as mx

        from mlx_audio.tts.models.indextts2.indextts2 import Model

        stub = SimpleNamespace(
            spk_matrix_split=[
                mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32) for _ in range(8)
            ],
            emo_matrix_split=[
                mx.array([[float(i), float(i)], [10.0, 10.0]], dtype=mx.float32)
                for i in range(8)
            ],
        )
        weights = {
            key: 0.0
            for key in (
                "happy",
                "angry",
                "sad",
                "afraid",
                "disgusted",
                "melancholic",
                "surprised",
                "calm",
            )
        }
        weights["happy"] = 1.0
        style = mx.array([[1.0, 0.0]], dtype=mx.float32)
        vector = Model._compute_emotion_vector(stub, weights, style)
        self.assertEqual(vector.shape, (1, 2))
        self.assertTrue(bool(mx.allclose(vector, mx.array([[0.0, 0.0]]))))


if __name__ == "__main__":
    unittest.main()
