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
    """Tests for mel_spectrogram verifying correct parameters.

    Uses snapshot values from the known-correct implementation to detect
    if mel_scale, norm, or center/reflect padding is changed.
    """

    def _get_random_audio(self):
        """Get deterministic random audio (seed=42, 12000 samples)."""
        np.random.seed(42)
        return np.random.randn(12000).astype(np.float32)

    def test_output_shape_with_padding(self):
        """Test that manual padding + center=False produces 46 frames for 12000 samples."""
        audio = mx.array(self._get_random_audio())
        mel = mel_spectrogram(audio)

        # manual 384 pad + center=False
        # padded_len = 12000 + 2*384 = 12768
        # frames = 1 + (12768 - 1024) // 256 = 46
        self.assertEqual(
            mel.shape,
            (1, 46, 128),
            "mel_spectrogram must use manual padding without center padding"
            f"Got shape {tuple(mel.shape)}, expected (1, 46, 128).",
        )

    def test_slaney_norm_values(self):
        """Test that slaney norm is applied (not norm=None).

        Without slaney norm, values would be ~3-5 (positive).
        With slaney norm, values are around -1 to 0.
        """
        audio = mx.array(self._get_random_audio())
        mel = mel_spectrogram(audio)
        mel_np = np.array(mel)[0]

        # Without norm="slaney", mel values are much higher (~3-5 range)
        self.assertLess(
            mel_np.mean(),
            1.0,
            "mel_spectrogram must use norm='slaney' in mel_filters(). "
            f"Got mean={mel_np.mean():.2f}, expected ~-0.37. "
            "Values > 1.0 indicate norm=None is being used.",
        )

        # Reference values from official Qwen3-TTS (slaney norm + slaney scale)
        expected_frame0 = np.array(
            [-0.21803714, 0.06630915, -0.31858957, -0.02480409, -0.4512914, -0.5911693]
        )
        actual_frame0 = mel_np[0, [0, 1, 2, 63, 126, 127]]
        np.testing.assert_allclose(
            actual_frame0,
            expected_frame0,
            rtol=1e-4,
            atol=1e-4,
            err_msg="mel_spectrogram must use norm='slaney' in mel_filters(). "
            "These values are specific to slaney-normalized filterbank.",
        )

    def test_slaney_mel_scale(self):
        """Test that slaney mel scale is used (not htk).

        HTK scale distributes mel bins differently, producing different values.
        With slaney scale, frame 0 bin 0 ≈ -0.22. With htk, it's ≈ -0.53.
        """
        audio = mx.array(self._get_random_audio())
        mel = mel_spectrogram(audio)
        mel_np = np.array(mel)[0]

        # With htk scale, low-frequency bins shift significantly
        # slaney: frame[0][0] ≈ -0.22, htk: frame[0][0] ≈ -0.53
        self.assertAlmostEqual(
            float(mel_np[0, 0]),
            -0.21803714,
            places=2,
            msg="mel_spectrogram must use mel_scale='slaney' in mel_filters(). "
            f"Got frame[0][bin 0]={mel_np[0, 0]:.4f}, expected ≈-0.22. "
            "A value of ≈-0.53 indicates mel_scale='htk' is being used.",
        )

        # Frame 23, selected bins - these values are specific to slaney scale
        expected_frame23 = np.array(
            [0.08127937, 0.4368576, 0.43200976, -0.7714137, -0.24601418, 0.04274124]
        )
        actual_frame23 = mel_np[23, [0, 1, 2, 63, 126, 127]]
        np.testing.assert_allclose(
            actual_frame23,
            expected_frame23,
            rtol=1e-4,
            atol=1e-4,
            err_msg="mel_spectrogram must use mel_scale='slaney' in mel_filters(). "
            "HTK scale distributes mel bins differently and produces wrong values.",
        )

    def test_reflect_padding_values(self):
        """Test that reflect padding produces correct boundary frame values.

        Without reflect padding, the first and last frames would have different
        values because the signal edges are handled differently.
        """
        audio = mx.array(self._get_random_audio())
        mel = mel_spectrogram(audio)
        mel_np = np.array(mel)[0]

        # Last frame values - sensitive to padding mode
        expected_last = np.array(
            [-0.16861804, 0.0474052, -0.3970174, -0.01738772, -0.28846806, -0.10941511]
        )
        actual_last = mel_np[-1, [0, 1, 2, 63, 126, 127]]
        np.testing.assert_allclose(
            actual_last,
            expected_last,
            rtol=1e-4,
            atol=1e-4,
            err_msg="mel_spectrogram must use reflect padding. "
            "Boundary frames are sensitive to the padding mode used before STFT.",
        )

    def test_sine_wave_mel_bins(self):
        """Test that a 1kHz sine wave activates the correct mel bins.

        This verifies both the mel scale and filterbank norm are correct,
        since wrong parameters would shift energy to different bins.
        """
        t = np.arange(12000, dtype=np.float32) / 24000.0
        audio = mx.array(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
        mel = mel_spectrogram(audio)
        mel_np = np.array(mel)[0]

        # Frame 0 values for a 1kHz sine - specific to slaney scale + slaney norm
        expected = np.array(
            [
                -1.2959518,
                -1.2937515,
                -1.2902284,
                -1.2074544,
                -0.9268621,
                -2.3822036,
                -5.331841,
                -5.33782,
            ]
        )
        actual = mel_np[0, [0, 1, 2, 10, 20, 63, 126, 127]]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-4,
            err_msg="mel_spectrogram must use mel_scale='slaney' and norm='slaney'. "
            "A 1kHz sine wave should produce these specific bin activations "
            "with slaney-scale mel filterbank.",
        )

    def test_overall_statistics(self):
        """Test overall mean and std match expected values."""
        audio = mx.array(self._get_random_audio())
        mel = mel_spectrogram(audio)
        mel_np = np.array(mel)

        np.testing.assert_allclose(
            mel_np.mean(),
            -0.37329558,
            rtol=1e-3,
            err_msg="mel_spectrogram output mean should be ~-0.37 with correct params. "
            f"Got mean={mel_np.mean():.4f}. A positive mean (~2.5) indicates norm=None.",
        )
        np.testing.assert_allclose(
            mel_np.std(),
            0.37445435,
            rtol=1e-3,
            err_msg="mel_spectrogram output std should be ~0.37 with correct params. "
            f"Got std={mel_np.std():.4f}.",
        )


if __name__ == "__main__":
    unittest.main()
