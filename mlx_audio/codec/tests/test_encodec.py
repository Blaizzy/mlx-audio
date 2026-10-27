import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

from ..models.encodec import Encodec, EncodecConfig
from ..models.encodec.encodec import LSTM, lstm_custom, lstm_fallback

config = EncodecConfig(
    audio_channels=1,
    chunk_length_s=None,
    codebook_dim=128,
    codebook_size=1024,
    compress=2,
    dilation_growth_rate=2,
    hidden_size=128,
    kernel_size=7,
    last_kernel_size=7,
    model_type="encodec",
    norm_type="weight_norm",
    normalize=False,
    num_filters=32,
    num_lstm_layers=2,
    num_residual_layers=1,
    overlap=None,
    pad_mode="reflect",
    residual_kernel_size=3,
    sampling_rate=24000,
    target_bandwidths=[1.5, 3.0, 6.0, 12.0, 24.0],
    trim_right_ratio=1.0,
    upsampling_ratios=[8, 5, 4, 2],
    use_causal_conv=True,
)


def _lstm_step_numpy(x, h_in, cell, time_step):
    i, f, g, o = np.split(x[:, time_step, :] + h_in, 4, axis=-1)
    i = 1 / (1 + np.exp(-i))
    f = 1 / (1 + np.exp(-f))
    o = 1 / (1 + np.exp(-o))
    cell = f * cell + i * np.tanh(g)
    return o * np.tanh(cell), cell


class TestLSTMFallback(unittest.TestCase):
    def test_no_metal_sequence_matches_numpy(self):
        rng = np.random.default_rng(43)
        for batch, steps, inputs, hidden_size in ((1, 1, 3, 1), (2, 4, 5, 17)):
            with self.subTest(shape=(batch, steps, inputs, hidden_size)):
                x = rng.standard_normal((batch, steps, inputs)).astype(np.float32)
                wx = rng.normal(0, 0.2, (4 * hidden_size, inputs)).astype(np.float32)
                wh = rng.normal(0, 0.2, (4 * hidden_size, hidden_size)).astype(
                    np.float32
                )
                b = rng.normal(0, 0.2, (4 * hidden_size,)).astype(np.float32)
                projected = x @ wx.T + b
                hidden = np.zeros((batch, hidden_size), dtype=np.float32)
                cell = np.zeros_like(hidden)
                expected = []
                for t in range(steps):
                    hidden, cell = _lstm_step_numpy(projected, hidden @ wh.T, cell, t)
                    expected.append(hidden)
                devices = [mx.cpu, mx.gpu] if mx.metal.is_available() else [mx.cpu]
                for device in devices:
                    with self.subTest(device=device), mx.stream(device):
                        layer = LSTM(inputs, hidden_size, bias=True)
                        layer.Wx = mx.array(wx)
                        layer.Wh = mx.array(wh)
                        layer.bias = mx.array(b)
                        with (
                            mock.patch.object(
                                mx.metal, "is_available", return_value=False
                            ),
                            mock.patch(
                                "mlx_audio.codec.models.encodec.encodec._lstm_kernel",
                                side_effect=AssertionError(
                                    "Metal kernel called without Metal"
                                ),
                            ),
                        ):
                            actual = layer(mx.array(x))
                            mx.eval(actual)
                        # GPU float32 matmuls may use TF32 products.
                        atol, rtol = (5e-4, 2e-3) if device == mx.gpu else (1e-5, 0)
                        np.testing.assert_allclose(
                            np.array(actual),
                            np.stack(expected, axis=1),
                            atol=atol,
                            rtol=rtol,
                        )

    @unittest.skipUnless(mx.metal.is_available(), "Metal is not available")
    def test_metal_matches_fallback(self):
        rng = np.random.default_rng(44)
        # The existing Metal kernel's batch offsets are only valid for batch=1.
        for hidden_size in (7, 33):
            with self.subTest(hidden_size=hidden_size):
                x = mx.array(rng.standard_normal((1, 4, 4 * hidden_size)), mx.float32)
                h_in = mx.array(rng.standard_normal((1, 4 * hidden_size)), mx.float32)
                cell = mx.array(rng.standard_normal((1, hidden_size)), mx.float32)
                for t in range(4):
                    actual = lstm_custom(x, h_in, cell, t)
                    expected = lstm_fallback(x, h_in, cell, t)
                    for out, ref in zip(actual, expected):
                        np.testing.assert_allclose(
                            np.array(out), np.array(ref), atol=1e-5, rtol=0
                        )


class TestEncodec(unittest.TestCase):
    """Test EnCodec model encoding and decoding."""

    def test_no_metal_random_weights_smoke(self):
        rng = np.random.default_rng(45)
        model = Encodec(
            EncodecConfig(
                num_filters=2,
                hidden_size=4,
                codebook_dim=4,
                codebook_size=16,
                num_lstm_layers=1,
                upsampling_ratios=[2, 2],
                target_bandwidths=[20.0],
                sampling_rate=8000,
            )
        )
        model.apply(lambda p: mx.array(rng.normal(0, 0.1, p.shape).astype(np.float32)))
        audio = mx.array(rng.standard_normal((1, 64, 1)).astype(np.float32))
        with (
            mock.patch.object(mx.metal, "is_available", return_value=False),
            mock.patch(
                "mlx_audio.codec.models.encodec.encodec._lstm_kernel",
                side_effect=AssertionError("Metal kernel called without Metal"),
            ),
        ):
            codes, scales = model.encode(audio)
            audio_out = model.decode(codes, scales)
            mx.eval(codes, audio_out)
        self.assertEqual(codes.shape, (1, 1, 1, 16))
        self.assertEqual(audio_out.shape, audio.shape)
        self.assertTrue(bool(mx.all(mx.isfinite(audio_out))))

    def test_encodec_24khz(self):
        model = Encodec(config)

        audio = mx.zeros((1, 120_000, 1))

        # default bandwidth
        codes, scales = model.encode(audio)
        self.assertEqual(codes.shape, (1, 1, 2, 375))

        audio_out = model.decode(codes, scales)
        self.assertEqual(audio_out.shape, (1, 120_000, 1))

        # 6kbps bandwidth
        codes, scales = model.encode(audio, bandwidth=6)
        self.assertEqual(codes.shape, (1, 1, 8, 375))

        audio_out = model.decode(codes, scales)
        self.assertEqual(audio_out.shape, (1, 120_000, 1))


if __name__ == "__main__":
    unittest.main()
