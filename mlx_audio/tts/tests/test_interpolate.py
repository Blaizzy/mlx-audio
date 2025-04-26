import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.interpolate import interpolate, interpolate1d


class TestInterpolate(unittest.TestCase):
    def test_interpolate_input_validation(self):
        """Test input validation in interpolate function."""
        # Test input with less than 3 dimensions
        with self.assertRaises(ValueError):
            interpolate(mx.array(np.zeros((2, 3))), size=4)

        # Test with both size and scale_factor defined
        with self.assertRaises(ValueError):
            interpolate(mx.array(np.zeros((2, 3, 4))), size=8, scale_factor=2)

        # Test with neither size nor scale_factor defined
        with self.assertRaises(ValueError):
            interpolate(mx.array(np.zeros((2, 3, 4))))

        # Test with unsupported dimensions
        with self.assertRaises(ValueError):
            interpolate(mx.array(np.zeros((2, 3, 4, 5))), size=8)

    def test_interpolate_size_handling(self):
        """Test size handling in interpolate function."""
        # Test with single size value
        x = mx.array(np.zeros((2, 3, 4)))
        result = interpolate(x, size=8)
        self.assertEqual(result.shape, (2, 3, 8))

        # Test with scale_factor
        x = mx.array(np.zeros((2, 3, 4)))
        result = interpolate(x, scale_factor=2)
        self.assertEqual(result.shape, (2, 3, 8))

    def test_interpolate1d_nearest(self):
        """Test 1D nearest neighbor interpolation."""
        # Create a simple test array
        x = mx.array(np.array([[[1.0, 2.0, 3.0, 4.0]]]))  # Shape: (1, 1, 4)

        # Test upsampling
        result = interpolate1d(x, size=8, mode="nearest")
        self.assertEqual(result.shape, (1, 1, 8))

        # Expected values for nearest neighbor interpolation
        expected = mx.array(np.array([[[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]]]))
        np.testing.assert_allclose(result.tolist(), expected.tolist(), rtol=1e-5)

        # Test downsampling
        result = interpolate1d(x, size=2, mode="nearest")
        self.assertEqual(result.shape, (1, 1, 2))

        # Expected values for nearest neighbor downsampling
        expected = mx.array(np.array([[[1.0, 3.0]]]))
        np.testing.assert_allclose(result.tolist(), expected.tolist(), rtol=1e-5)

    def test_interpolate1d_linear(self):
        """Test 1D linear interpolation."""
        # Create a simple test array
        x = mx.array(np.array([[[1.0, 3.0, 5.0, 7.0]]]))  # Shape: (1, 1, 4)

        # Test upsampling with align_corners=True
        result = interpolate1d(x, size=7, mode="linear", align_corners=True)
        self.assertEqual(result.shape, (1, 1, 7))

        # Expected values for linear interpolation with align_corners=True
        expected = mx.array(np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]]))
        np.testing.assert_allclose(result.tolist(), expected.tolist(), rtol=1e-5)

        # Test upsampling with align_corners=False
        result = interpolate1d(x, size=7, mode="linear", align_corners=False)
        # Shape should be correct
        self.assertEqual(result.shape, (1, 1, 7))

        # Test edge case: input width = 1
        x_single = mx.array(np.array([[[5.0]]]))  # Shape: (1, 1, 1)
        result = interpolate1d(x_single, size=4, mode="linear")
        self.assertEqual(result.shape, (1, 1, 4))
        expected = mx.array(np.array([[[5.0, 5.0, 5.0, 5.0]]]))
        np.testing.assert_allclose(result.tolist(), expected.tolist(), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
