# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.qwen3_tts.qwen3_tts import mel_spectrogram
from mlx_audio.tts.models.qwen3_tts.speaker_encoder import (
    TimeDelayNetBlock,
    reflect_pad_1d,
)


class TestReflectPad1d(unittest.TestCase):
    """Tests for reflect_pad_1d helper function."""

    def test_no_padding(self):
        """Test that pad=0 returns the input unchanged."""
        x = mx.ones((1, 5, 3))
        result = reflect_pad_1d(x, pad=0)
        np.testing.assert_array_equal(np.array(result), np.array(x))

    def test_pad_1(self):
        """Test reflect padding with pad=1."""
        # Input: [1, 5, 1] with values [0, 1, 2, 3, 4]
        x = mx.array([[[0.0], [1.0], [2.0], [3.0], [4.0]]])
        result = reflect_pad_1d(x, pad=1)

        # Reflect pad=1: left mirrors x[1], right mirrors x[-2]
        # Expected: [1, 0, 1, 2, 3, 4, 3]
        expected = np.array([[[1.0], [0.0], [1.0], [2.0], [3.0], [4.0], [3.0]]])
        np.testing.assert_array_equal(np.array(result), expected)

    def test_pad_2(self):
        """Test reflect padding with pad=2."""
        x = mx.array([[[0.0], [1.0], [2.0], [3.0], [4.0]]])
        result = reflect_pad_1d(x, pad=2)

        # Reflect pad=2: left mirrors x[1:3] reversed, right mirrors x[-3:-1] reversed
        # Left: x[1:3] = [1,2] reversed = [2,1]
        # Right: x[-3:-1] = [2,3] reversed = [3,2]
        # Expected: [2, 1, 0, 1, 2, 3, 4, 3, 2]
        expected = np.array(
            [[[2.0], [1.0], [0.0], [1.0], [2.0], [3.0], [4.0], [3.0], [2.0]]]
        )
        np.testing.assert_array_equal(np.array(result), expected)

    def test_output_shape(self):
        """Test that output shape is [batch, time + 2*pad, channels]."""
        batch, time, channels = 2, 10, 4
        pad = 3
        x = mx.random.normal((batch, time, channels))
        result = reflect_pad_1d(x, pad)
        self.assertEqual(result.shape, (batch, time + 2 * pad, channels))

    def test_multichannel(self):
        """Test that reflect padding works correctly across multiple channels."""
        # Each channel should be padded independently with the same pattern
        x = mx.array(
            [
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                    [5.0, 50.0],
                ]
            ]
        )
        result = reflect_pad_1d(x, pad=1)
        result_np = np.array(result)

        # Channel 0: [2, 1, 2, 3, 4, 5, 4]
        expected_ch0 = [2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0]
        # Channel 1: [20, 10, 20, 30, 40, 50, 40]
        expected_ch1 = [20.0, 10.0, 20.0, 30.0, 40.0, 50.0, 40.0]

        np.testing.assert_array_equal(result_np[0, :, 0], expected_ch0)
        np.testing.assert_array_equal(result_np[0, :, 1], expected_ch1)


class TestTimeDelayNetBlockReflectPadding(unittest.TestCase):
    """Tests for TimeDelayNetBlock reflect padding behavior."""

    def test_output_shape_preserves_time(self):
        """Test that TimeDelayNetBlock with reflect padding preserves time dimension."""
        in_channels, out_channels = 16, 32
        kernel_size, dilation = 3, 1
        block = TimeDelayNetBlock(in_channels, out_channels, kernel_size, dilation)

        batch, time = 1, 20
        x = mx.random.normal((batch, in_channels, time))  # NCL format
        out = block(x)

        # With reflect padding, output time should equal input time
        self.assertEqual(out.shape, (batch, out_channels, time))

    def test_output_shape_with_dilation(self):
        """Test that dilated convolution with reflect padding preserves time."""
        in_channels, out_channels = 16, 32
        kernel_size, dilation = 3, 2
        block = TimeDelayNetBlock(in_channels, out_channels, kernel_size, dilation)

        batch, time = 1, 20
        x = mx.random.normal((batch, in_channels, time))
        out = block(x)

        self.assertEqual(out.shape, (batch, out_channels, time))

    def test_output_shape_kernel5_dilation2(self):
        """Test larger kernel with dilation preserves time."""
        in_channels, out_channels = 16, 32
        kernel_size, dilation = 5, 2
        block = TimeDelayNetBlock(in_channels, out_channels, kernel_size, dilation)

        batch, time = 1, 30
        x = mx.random.normal((batch, in_channels, time))
        out = block(x)

        self.assertEqual(out.shape, (batch, out_channels, time))

    def test_kernel1_no_padding(self):
        """Test that kernel_size=1 results in no padding."""
        block = TimeDelayNetBlock(16, 32, kernel_size=1, dilation=1)
        self.assertEqual(block.pad, 0)

    def test_pad_calculation(self):
        """Test that padding is computed correctly for various kernel/dilation combos."""
        # kernel=3, dilation=1 -> pad = (3-1)*1//2 = 1
        block = TimeDelayNetBlock(16, 32, kernel_size=3, dilation=1)
        self.assertEqual(block.pad, 1)

        # kernel=3, dilation=2 -> pad = (3-1)*2//2 = 2
        block = TimeDelayNetBlock(16, 32, kernel_size=3, dilation=2)
        self.assertEqual(block.pad, 2)

        # kernel=5, dilation=1 -> pad = (5-1)*1//2 = 2
        block = TimeDelayNetBlock(16, 32, kernel_size=5, dilation=1)
        self.assertEqual(block.pad, 2)

        # kernel=5, dilation=3 -> pad = (5-1)*3//2 = 6
        block = TimeDelayNetBlock(16, 32, kernel_size=5, dilation=3)
        self.assertEqual(block.pad, 6)

    def test_output_is_relu_activated(self):
        """Test that output values are non-negative (ReLU applied)."""
        block = TimeDelayNetBlock(16, 32, kernel_size=3, dilation=1)

        x = mx.random.normal((1, 16, 50))
        out = block(x)
        out_np = np.array(out)

        self.assertTrue(np.all(out_np >= 0), "Output should be non-negative after ReLU")


class TestMelSpectrogram(unittest.TestCase):
    """Tests for mel_spectrogram function."""

    def test_output_shape_1d_input(self):
        """Test mel spectrogram output shape with 1D input."""
        # 1 second of audio at 24kHz
        audio = mx.random.normal((24000,))
        mel = mel_spectrogram(audio)

        # Expected: [1, frames, 128]
        self.assertEqual(mel.ndim, 3)
        self.assertEqual(mel.shape[0], 1)  # batch
        self.assertEqual(mel.shape[2], 128)  # n_mels

    def test_output_shape_2d_input(self):
        """Test mel spectrogram output shape with 2D batched input."""
        batch_size = 3
        audio = mx.random.normal((batch_size, 24000))
        mel = mel_spectrogram(audio)

        self.assertEqual(mel.shape[0], batch_size)
        self.assertEqual(mel.shape[2], 128)

    def test_frame_count_with_center_padding(self):
        """Test that center=True reflect padding produces correct number of frames."""
        n_fft = 1024
        hop_size = 256
        num_samples = 24000

        audio = mx.random.normal((num_samples,))
        mel = mel_spectrogram(audio, n_fft=n_fft, hop_size=hop_size)

        # With center=True, padded length = num_samples + n_fft
        # num_frames = 1 + (padded_length - n_fft) // hop_length
        #            = 1 + num_samples // hop_length
        expected_frames = 1 + num_samples // hop_size
        self.assertEqual(mel.shape[1], expected_frames)

    def test_uses_slaney_mel_scale(self):
        """Test that mel spectrogram uses slaney norm and scale by verifying output range."""
        # A sine wave at 1kHz should activate specific mel bins
        t = mx.arange(24000) / 24000.0
        audio = mx.sin(2 * np.pi * 1000 * t)
        mel = mel_spectrogram(audio)

        # With log scale and clip at 1e-5, min should be log(1e-5) ≈ -11.51
        mel_np = np.array(mel)
        self.assertTrue(np.all(mel_np >= np.log(1e-5) - 0.01))

    def test_log_scale_applied(self):
        """Test that output is in log scale (clipped at 1e-5)."""
        audio = mx.random.normal((24000,))
        mel = mel_spectrogram(audio)
        mel_np = np.array(mel)

        # Log(clip(x, 1e-5)) means minimum value is log(1e-5) ≈ -11.51
        min_val = np.log(1e-5)
        self.assertTrue(
            np.all(mel_np >= min_val - 0.01),
            f"Mel values should be >= log(1e-5), got min={mel_np.min()}",
        )

    def test_deterministic_output(self):
        """Test that the same input produces the same output."""
        audio = mx.random.normal((24000,))
        mel1 = mel_spectrogram(audio)
        mel2 = mel_spectrogram(audio)

        np.testing.assert_array_equal(np.array(mel1), np.array(mel2))

    def test_custom_parameters(self):
        """Test mel spectrogram with non-default parameters."""
        audio = mx.random.normal((16000,))
        mel = mel_spectrogram(
            audio,
            n_fft=512,
            num_mels=80,
            sample_rate=16000,
            hop_size=128,
            win_size=512,
            fmin=80.0,
            fmax=7600.0,
        )

        self.assertEqual(mel.shape[2], 80)  # Custom n_mels
        # Frames: 1 + 16000 // 128 = 126
        expected_frames = 1 + 16000 // 128
        self.assertEqual(mel.shape[1], expected_frames)


if __name__ == "__main__":
    unittest.main()
