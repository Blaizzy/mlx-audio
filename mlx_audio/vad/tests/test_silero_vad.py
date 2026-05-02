import unittest

import mlx.core as mx
import numpy as np

from ..models.silero_vad.config import BranchConfig, ModelConfig
from ..models.silero_vad.silero_vad import Model, SileroVADState, VADOutput


class TestSileroVAD(unittest.TestCase):
    def test_default_config(self):
        cfg = ModelConfig()
        self.assertEqual(cfg.model_type, "silero_vad")
        self.assertEqual(cfg.branch_16k.sample_rate, 16000)
        self.assertEqual(cfg.branch_16k.chunk_size, 512)
        self.assertEqual(cfg.branch_8k.sample_rate, 8000)
        self.assertEqual(cfg.branch_8k.chunk_size, 256)

    def test_config_from_dict(self):
        cfg = ModelConfig.from_dict(
            {
                "dtype": "float16",
                "branch_16k": {"chunk_size": 512, "context_size": 64},
                "branch_8k": {"sample_rate": 8000, "filter_length": 128},
            }
        )
        self.assertEqual(cfg.dtype, "float16")
        self.assertIsInstance(cfg.branch_16k, BranchConfig)
        self.assertEqual(cfg.branch_8k.filter_length, 128)

    def test_forward_shape_and_state_16k(self):
        model = Model(ModelConfig())
        x = mx.zeros((2, 576), dtype=mx.float32)
        out, state = model(x, sample_rate=16000)
        mx.eval(out, state)
        self.assertEqual(out.shape, (2, 1))
        self.assertEqual(state.shape, (2, 2, 128))
        self.assertGreaterEqual(float(out.min().item()), 0.0)
        self.assertLessEqual(float(out.max().item()), 1.0)

    def test_forward_shape_and_state_8k(self):
        model = Model(ModelConfig())
        x = mx.zeros((1, 288), dtype=mx.float32)
        out, state = model(x, sample_rate=8000)
        mx.eval(out, state)
        self.assertEqual(out.shape, (1, 1))
        self.assertEqual(state.shape, (2, 1, 128))

    def test_feed_updates_streaming_context(self):
        model = Model(ModelConfig())
        chunk = np.zeros((512,), dtype=np.float32)
        out, state = model.feed(chunk, sample_rate=16000)
        mx.eval(out, state.context)
        self.assertEqual(out.shape, (1, 1))
        self.assertIsInstance(state, SileroVADState)
        self.assertEqual(state.context.shape, (1, 64))

    def test_predict_proba_chunks(self):
        model = Model(ModelConfig())
        audio = np.zeros((1024,), dtype=np.float32)
        probs = model.predict_proba(audio, sample_rate=16000)
        mx.eval(probs)
        self.assertEqual(probs.shape, (2,))

    def test_generate_returns_output(self):
        model = Model(ModelConfig())
        audio = np.zeros((512,), dtype=np.float32)
        result = model.generate(audio, sample_rate=16000)
        self.assertIsInstance(result, VADOutput)
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.probabilities.shape, (1,))

    def test_probs_to_timestamps(self):
        probs = np.array([0.1, 0.8, 0.85, 0.1, 0.1], dtype=np.float32)
        timestamps = Model._probs_to_timestamps(
            probs,
            audio_len=5 * 512,
            sample_rate=16000,
            threshold=0.5,
            min_speech_duration_ms=30,
            min_silence_duration_ms=30,
            speech_pad_ms=0,
            return_seconds=False,
        )
        self.assertEqual(timestamps, [{"start": 512, "end": 1536}])


if __name__ == "__main__":
    unittest.main()
