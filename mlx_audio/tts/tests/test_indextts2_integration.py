import os
import unittest


@unittest.skipUnless(
    os.environ.get("MLX_INDEXTTS2_MODEL") and os.environ.get("MLX_INDEXTTS2_REF_AUDIO"),
    "set MLX_INDEXTTS2_MODEL and MLX_INDEXTTS2_REF_AUDIO for the integration test",
)
class TestIndexTTS25Integration(unittest.TestCase):
    def test_generate_finite_audio(self):
        import numpy as np

        from mlx_audio.tts import load

        model = load(os.environ["MLX_INDEXTTS2_MODEL"], strict=True)
        result = next(
            model.generate(
                "Hello from IndexTTS two point five.",
                ref_audio=os.environ["MLX_INDEXTTS2_REF_AUDIO"],
                language="en",
                diffusion_steps=8,
            )
        )
        audio = np.array(result.audio, dtype=np.float32)
        self.assertGreater(audio.size, 0)
        self.assertTrue(np.isfinite(audio).all())
        self.assertEqual(result.sample_rate, 22050)


if __name__ == "__main__":
    unittest.main()
