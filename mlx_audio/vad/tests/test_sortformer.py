import unittest

import mlx.core as mx
from mlx.utils import tree_map

from ..models.sortformer.config import (
    FCEncoderConfig,
    ModelConfig,
    ModulesConfig,
    ProcessorConfig,
    TFEncoderConfig,
)
from ..models.sortformer.sortformer import (
    DiarizationOutput,
    DiarizationSegment,
    Model,
    StreamingState,
)

# Minimal configs for fast unit tests (tiny model)
_TINY_FC = {
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "num_mel_bins": 16,
    "subsampling_conv_channels": 16,
    "max_position_embeddings": 512,
    "conv_kernel_size": 3,
}

_TINY_TF = {
    "d_model": 24,
    "encoder_layers": 1,
    "encoder_attention_heads": 2,
    "encoder_ffn_dim": 48,
    "max_source_positions": 512,
    "num_mel_bins": 16,
}

_TINY_MOD = {
    "fc_d_model": 32,
    "tf_d_model": 24,
    "num_speakers": 4,
}

_TINY_PROC = {
    "feature_size": 16,
    "hop_length": 160,
    "n_fft": 512,
    "win_length": 400,
}


def _make_config(dtype="float32", use_aosc=False):
    mod = {**_TINY_MOD, "use_aosc": use_aosc}
    return ModelConfig(
        dtype=dtype,
        fc_encoder_config=FCEncoderConfig.from_dict(_TINY_FC),
        tf_encoder_config=TFEncoderConfig.from_dict(_TINY_TF),
        modules_config=ModulesConfig.from_dict(mod),
        processor_config=ProcessorConfig.from_dict(_TINY_PROC),
    )


def _make_model(dtype="float32", use_aosc=False):
    cfg = _make_config(dtype, use_aosc)
    model = Model(cfg)
    mx.eval(model.parameters())
    if dtype == "float16":
        model.update(tree_map(lambda p: p.astype(mx.float16), model.parameters()))
        mx.eval(model.parameters())
    return model


class TestSortformerModel(unittest.TestCase):
    """Sortformer model: config, dtype, forward, streaming, post-processing, sanitize."""

    # -- Config --

    def test_default_config(self):
        cfg = ModelConfig()
        self.assertEqual(cfg.model_type, "sortformer")
        self.assertEqual(cfg.num_speakers, 4)
        self.assertEqual(cfg.dtype, "float32")
        self.assertIsInstance(cfg.fc_encoder_config, FCEncoderConfig)
        self.assertIsInstance(cfg.tf_encoder_config, TFEncoderConfig)
        self.assertIsInstance(cfg.modules_config, ModulesConfig)
        self.assertIsInstance(cfg.processor_config, ProcessorConfig)

    def test_config_from_dict(self):
        cfg = ModelConfig.from_dict(
            {
                "dtype": "float16",
                "fc_encoder_config": _TINY_FC,
                "tf_encoder_config": _TINY_TF,
                "modules_config": _TINY_MOD,
                "processor_config": _TINY_PROC,
            }
        )
        self.assertEqual(cfg.dtype, "float16")
        self.assertEqual(cfg.fc_encoder_config.hidden_size, 32)

    def test_aosc_flag(self):
        self.assertTrue(_make_config(use_aosc=True).modules_config.use_aosc)
        self.assertFalse(_make_config(use_aosc=False).modules_config.use_aosc)

    # -- Dtype propagation --

    def test_dtype_propagation(self):
        for t in [mx.float32, mx.float16]:
            dtype_str = "float16" if t == mx.float16 else "float32"
            model = _make_model(dtype_str)
            self.assertEqual(model.dtype, t)
            n = model.config.fc_encoder_config.num_mel_bins
            preds = model(mx.zeros((1, n, 160), dtype=t), mx.array([160]))
            mx.eval(preds)
            self.assertEqual(preds.dtype, t)

    # -- Forward pass --

    def test_output_shape(self):
        model = _make_model()
        n = model.config.fc_encoder_config.num_mel_bins
        preds = model(mx.zeros((1, n, 320)), mx.array([320]))
        mx.eval(preds)
        self.assertEqual(preds.ndim, 3)
        self.assertEqual(preds.shape[0], 1)
        self.assertEqual(preds.shape[2], 4)

    def test_output_range(self):
        model = _make_model()
        n = model.config.fc_encoder_config.num_mel_bins
        preds = model(mx.random.normal((1, n, 320)), mx.array([320]))
        mx.eval(preds)
        self.assertGreaterEqual(preds.min().item(), 0.0)
        self.assertLessEqual(preds.max().item(), 1.0)

    def test_batch_dimension(self):
        model = _make_model()
        n = model.config.fc_encoder_config.num_mel_bins
        preds = model(mx.zeros((2, n, 320)), mx.array([320, 320]))
        mx.eval(preds)
        self.assertEqual(preds.shape[0], 2)

    # -- Streaming --

    def test_init_streaming_state(self):
        state = _make_model().init_streaming_state()
        self.assertIsInstance(state, StreamingState)
        self.assertEqual(state.spkcache_len, 0)
        self.assertEqual(state.fifo_len, 0)
        self.assertEqual(state.frames_processed, 0)

    def test_streaming_step(self):
        model = _make_model()
        state = model.init_streaming_state()
        n = model.config.fc_encoder_config.num_mel_bins
        preds, new_state = model.streaming_step(
            mx.zeros((1, n, 160)), mx.array([160]), state
        )
        mx.eval(preds)
        self.assertEqual(preds.ndim, 2)
        self.assertEqual(preds.shape[1], 4)
        self.assertIsInstance(new_state, StreamingState)

    def test_streaming_state_accumulates(self):
        model = _make_model()
        state = model.init_streaming_state()
        n = model.config.fc_encoder_config.num_mel_bins
        chunk = mx.zeros((1, n, 160))
        length = mx.array([160])
        _, state = model.streaming_step(chunk, length, state)
        fifo1 = state.fifo_len
        _, state = model.streaming_step(chunk, length, state)
        self.assertGreater(state.fifo_len, 0)
        self.assertGreaterEqual(state.fifo_len, fifo1)

    def test_context_gated_by_aosc(self):
        self.assertFalse(_make_model(use_aosc=False).config.modules_config.use_aosc)
        self.assertTrue(_make_model(use_aosc=True).config.modules_config.use_aosc)

    # -- Post-processing --

    def test_preds_to_segments(self):
        preds = mx.zeros((10, 4))
        preds = preds.at[2:6, 0].add(1.0)
        preds = preds.at[7:10, 1].add(1.0)
        mx.eval(preds)
        segs = Model._preds_to_segments(preds, frame_duration=0.08, threshold=0.5)
        speakers = {s.speaker for s in segs}
        self.assertEqual(speakers, {0, 1})
        for s in segs:
            self.assertGreater(s.end, s.start)

    def test_empty_preds_no_segments(self):
        preds = mx.zeros((10, 4))
        mx.eval(preds)
        segs = Model._preds_to_segments(preds, frame_duration=0.08, threshold=0.5)
        self.assertEqual(len(segs), 0)

    def test_diarization_output_text(self):
        segs = [
            DiarizationSegment(start=0.0, end=1.0, speaker=0),
            DiarizationSegment(start=1.5, end=2.5, speaker=1),
        ]
        out = DiarizationOutput(segments=segs, num_speakers=2)
        self.assertIn("speaker_0", out.text)
        self.assertIn("speaker_1", out.text)
        self.assertEqual(len(out.text.strip().split("\n")), 2)

    # -- Sanitize --

    def test_sanitize_conv2d_transpose(self):
        weights = {"fc_encoder.subsampling.layers.0.weight": mx.zeros((16, 1, 3, 3))}
        w = Model.sanitize(weights)["fc_encoder.subsampling.layers_0.weight"]
        self.assertEqual(w.shape, (16, 3, 3, 1))

    def test_sanitize_conv1d_transpose(self):
        weights = {
            "fc_encoder.layers.0.conv.depthwise_conv.weight": mx.zeros((32, 1, 9))
        }
        w = Model.sanitize(weights)["fc_encoder.layers.0.conv.depthwise_conv.weight"]
        self.assertEqual(w.shape, (32, 9, 1))

    def test_sanitize_already_converted_passthrough(self):
        weights = {"fc_encoder.subsampling.layers_0.weight": mx.zeros((16, 3, 3, 1))}
        w = Model.sanitize(weights)["fc_encoder.subsampling.layers_0.weight"]
        self.assertEqual(w.shape, (16, 3, 3, 1))

    def test_sanitize_skips_batchnorm_tracking(self):
        sanitized = Model.sanitize(
            {
                "fc_encoder.layers.0.conv.norm.running_mean": mx.zeros((32,)),
                "fc_encoder.layers.0.conv.norm.num_batches_tracked": mx.array(100),
            }
        )
        self.assertIn("fc_encoder.layers.0.conv.norm.running_mean", sanitized)
        self.assertNotIn("fc_encoder.layers.0.conv.norm.num_batches_tracked", sanitized)


if __name__ == "__main__":
    unittest.main()
