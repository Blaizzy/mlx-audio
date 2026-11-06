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

    def test_explicit_emotion_vector(self):
        import numpy as np

        from mlx_audio.tts import load

        model = load(os.environ["MLX_INDEXTTS2_MODEL"], strict=True)
        if model.emo_matrix is None:
            self.skipTest("emotion matrices are not converted")
        result = next(
            model.generate(
                "I am very happy to see you.",
                ref_audio=os.environ["MLX_INDEXTTS2_REF_AUDIO"],
                language="en",
                emo_vector=[1.0, 0, 0, 0, 0, 0, 0, 0],
                emo_alpha=0.8,
                diffusion_steps=8,
            )
        )
        audio = np.array(result.audio, dtype=np.float32)
        self.assertGreater(audio.size, 0)
        self.assertTrue(np.isfinite(audio).all())

    def test_segment_streaming(self):
        import numpy as np

        from mlx_audio.tts import load

        model = load(os.environ["MLX_INDEXTTS2_MODEL"], strict=True)
        chunks = list(
            model.generate(
                "Hello from IndexTTS. This is another sentence. And a final one.",
                ref_audio=os.environ["MLX_INDEXTTS2_REF_AUDIO"],
                language="en",
                stream=True,
                max_text_tokens_per_segment=8,
                diffusion_steps=8,
            )
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual(
            [chunk.segment_idx for chunk in chunks], list(range(len(chunks)))
        )
        self.assertTrue(all(chunk.is_streaming_chunk for chunk in chunks))
        self.assertEqual(
            [chunk.is_final_chunk for chunk in chunks],
            [False] * (len(chunks) - 1) + [True],
        )
        for chunk in chunks:
            audio = np.array(chunk.audio, dtype=np.float32)
            self.assertGreater(audio.size, 0)
            self.assertTrue(np.isfinite(audio).all())


if __name__ == "__main__":
    unittest.main()
