import unittest

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from ..models.smart_turn.config import EncoderConfig, ModelConfig, ProcessorConfig
from ..models.smart_turn.smart_turn import EndpointOutput, Model

_TINY_ENCODER = {
    "num_mel_bins": 8,
    "max_source_positions": 64,
    "d_model": 16,
    "encoder_attention_heads": 2,
    "encoder_layers": 1,
    "encoder_ffn_dim": 32,
    "k_proj_bias": False,
}

_TINY_PROCESSOR = {
    "sampling_rate": 16000,
    "max_audio_seconds": 8,
    "n_fft": 400,
    "hop_length": 160,
    "n_mels": 8,
    "normalize_audio": True,
    "threshold": 0.5,
}


def _make_config(dtype="float32"):
    return ModelConfig(
        dtype=dtype,
        encoder_config=EncoderConfig.from_dict(_TINY_ENCODER),
        processor_config=ProcessorConfig.from_dict(_TINY_PROCESSOR),
    )


def _make_model(dtype="float32"):
    model = Model(_make_config(dtype))
    mx.eval(model.parameters())
    if dtype == "float16":
        model.update(tree_map(lambda p: p.astype(mx.float16), model.parameters()))
        mx.eval(model.parameters())
    return model


class TestSmartTurnV3(unittest.TestCase):
    def test_default_config(self):
        cfg = ModelConfig()
        self.assertEqual(cfg.model_type, "smart_turn")
        self.assertEqual(cfg.architecture, "smart_turn")
        self.assertEqual(cfg.dtype, "float32")
        self.assertIsInstance(cfg.encoder_config, EncoderConfig)
        self.assertIsInstance(cfg.processor_config, ProcessorConfig)
        self.assertEqual(cfg.processor_config.sampling_rate, 16000)
        self.assertEqual(cfg.processor_config.max_audio_seconds, 8)

    def test_config_from_dict(self):
        cfg = ModelConfig.from_dict(
            {
                "dtype": "float16",
                "sample_rate": 22050,
                "max_audio_seconds": 6,
                "threshold": 0.42,
                "encoder_config": _TINY_ENCODER,
                "processor_config": _TINY_PROCESSOR,
            }
        )
        self.assertEqual(cfg.dtype, "float16")
        self.assertEqual(cfg.sample_rate, 22050)
        self.assertEqual(cfg.max_audio_seconds, 6)
        self.assertAlmostEqual(cfg.threshold, 0.42)
        self.assertEqual(cfg.encoder_config.d_model, 16)
        self.assertEqual(cfg.processor_config.n_mels, 8)

    def test_forward_output_shape_and_range(self):
        model = _make_model()
        inp = mx.zeros((1, 8, 64), dtype=mx.float32)
        out = model(inp)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 1))
        self.assertGreaterEqual(float(out.min().item()), 0.0)
        self.assertLessEqual(float(out.max().item()), 1.0)

    def test_forward_return_logits(self):
        model = _make_model()
        inp = mx.zeros((1, 8, 64), dtype=mx.float32)
        logits = model(inp, return_logits=True)
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 1))

    def test_forward_batch_dimension(self):
        model = _make_model()
        inp = mx.zeros((2, 8, 64), dtype=mx.float32)
        out = model(inp)
        mx.eval(out)
        self.assertEqual(out.shape, (2, 1))

    def test_dtype_propagation(self):
        for t in [mx.float32, mx.float16]:
            dtype_str = "float16" if t == mx.float16 else "float32"
            model = _make_model(dtype_str)
            inp = mx.zeros((1, 8, 64), dtype=t)
            out = model(inp)
            mx.eval(out)
            self.assertEqual(model.dtype, t)
            self.assertEqual(out.dtype, t)

    def test_prepare_audio_array_lengths(self):
        model = _make_model()
        max_samples = (
            model.config.processor_config.max_audio_seconds
            * model.config.processor_config.sampling_rate
        )

        short = np.ones((16000,), dtype=np.float32)
        short_out = model._prepare_audio_array(short, sample_rate=16000)
        self.assertEqual(short_out.shape[0], max_samples)

        long = np.ones((200000,), dtype=np.float32)
        long_out = model._prepare_audio_array(long, sample_rate=16000)
        self.assertEqual(long_out.shape[0], max_samples)

    def test_prepare_audio_array_resample(self):
        model = _make_model()
        max_samples = (
            model.config.processor_config.max_audio_seconds
            * model.config.processor_config.sampling_rate
        )
        audio_8k = np.ones((8000,), dtype=np.float32)  # 1 second @ 8kHz
        out = model._prepare_audio_array(audio_8k, sample_rate=8000)
        self.assertEqual(out.shape[0], max_samples)

    def test_predict_endpoint_returns_dataclass(self):
        model = _make_model()
        audio = np.zeros((16000,), dtype=np.float32)
        result = model.predict_endpoint(audio, sample_rate=16000, threshold=0.5)
        self.assertIsInstance(result, EndpointOutput)
        self.assertIn(result.prediction, (0, 1))
        self.assertIsInstance(result.probability, float)

    def test_sanitize_drops_val_constants(self):
        sanitized = Model.sanitize(
            {
                "val_17": mx.zeros((16, 16), dtype=mx.float32),
                "val_123": mx.zeros((1,), dtype=mx.float32),
            }
        )
        self.assertEqual(sanitized, {})

    def test_sanitize_remaps_prefixes(self):
        sanitized = Model.sanitize(
            {
                "inner.classifier.0.weight": mx.zeros((16, 16), dtype=mx.float32),
                "inner.pool_attention.2.bias": mx.zeros((1,), dtype=mx.float32),
            }
        )
        self.assertIn("classifier_0.weight", sanitized)
        self.assertIn("pool_attention_2.bias", sanitized)

    def test_sanitize_conv1d_transpose(self):
        weights = {"encoder.conv1.weight": mx.zeros((16, 8, 3), dtype=mx.float32)}
        sanitized = Model.sanitize(weights)
        self.assertEqual(sanitized["encoder.conv1.weight"].shape, (16, 3, 8))

    def test_sanitize_fc_transpose_heuristics(self):
        weights = {
            "encoder.layers.0.fc1.weight": mx.zeros((16, 32), dtype=mx.float32),
            "encoder.layers.0.fc2.weight": mx.zeros((32, 16), dtype=mx.float32),
        }
        sanitized = Model.sanitize(weights)
        self.assertEqual(sanitized["encoder.layers.0.fc1.weight"].shape, (32, 16))
        self.assertEqual(sanitized["encoder.layers.0.fc2.weight"].shape, (16, 32))

    def test_sanitize_pool_transpose_heuristics(self):
        weights = {
            "pool_attention.0.weight": mx.zeros((16, 256), dtype=mx.float32),
            "pool_attention.2.weight": mx.zeros((256, 1), dtype=mx.float32),
        }
        sanitized = Model.sanitize(weights)
        self.assertEqual(sanitized["pool_attention_0.weight"].shape, (256, 16))
        self.assertEqual(sanitized["pool_attention_2.weight"].shape, (1, 256))


if __name__ == "__main__":
    unittest.main()
