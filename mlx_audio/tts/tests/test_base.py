import unittest

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.base import BaseModelArgs, check_array_shape


class TestBaseModel(unittest.TestCase):
    def test_base_model_args_from_dict(self):
        """Test BaseModelArgs.from_dict method."""

        # Define a test subclass
        class TestArgs(BaseModelArgs):
            def __init__(self, param1, param2, param3=None):
                self.param1 = param1
                self.param2 = param2
                self.param3 = param3

        # Test with all parameters
        params = {"param1": 1, "param2": "test", "param3": True}
        args = TestArgs.from_dict(params)
        self.assertEqual(args.param1, 1)
        self.assertEqual(args.param2, "test")
        self.assertEqual(args.param3, True)

        # Test with extra parameters (should be ignored)
        params = {"param1": 1, "param2": "test", "param3": True, "extra": "ignored"}
        args = TestArgs.from_dict(params)
        self.assertEqual(args.param1, 1)
        self.assertEqual(args.param2, "test")
        self.assertEqual(args.param3, True)
        self.assertFalse(hasattr(args, "extra"))

        # Test with missing optional parameter
        params = {"param1": 1, "param2": "test"}
        args = TestArgs.from_dict(params)
        self.assertEqual(args.param1, 1)
        self.assertEqual(args.param2, "test")
        self.assertIsNone(args.param3)

    def test_check_array_shape(self):
        """Test check_array_shape function.

        MLX conv1d format: (out_channels, kernel_size, in_channels)
        PyTorch conv1d format: (out_channels, in_channels, kernel_size)

        Returns True if array is in MLX format (no transpose needed).
        Returns False if array is in PyTorch format (needs transpose).
        """
        # MLX format: kernel_size (small) in middle, in_channels (large) at end
        mlx_format_1 = mx.array(np.zeros((512, 3, 512)))  # (out, kernel=3, in)
        self.assertTrue(check_array_shape(mlx_format_1))

        mlx_format_2 = mx.array(np.zeros((64, 7, 128)))  # (out, kernel=7, in)
        self.assertTrue(check_array_shape(mlx_format_2))

        # PyTorch format: in_channels (large) in middle, kernel_size (small) at end
        pytorch_format_1 = mx.array(np.zeros((512, 512, 3)))  # (out, in, kernel=3)
        self.assertFalse(check_array_shape(pytorch_format_1))

        pytorch_format_2 = mx.array(np.zeros((64, 128, 7)))  # (out, in, kernel=7)
        self.assertFalse(check_array_shape(pytorch_format_2))

        # Ambiguous case: small equal dimensions (both kernel-like)
        # When dim1 <= dim2, assume MLX format
        ambiguous_equal = mx.array(np.zeros((64, 3, 3)))
        self.assertTrue(check_array_shape(ambiguous_equal))

        # Edge case: dim1=1 (likely in_channels) with larger dim2 (likely kernel)
        # Should be detected as PyTorch format
        edge_case_pytorch = mx.array(np.zeros((512, 1, 3)))  # (out, in=1, kernel=3)
        self.assertFalse(check_array_shape(edge_case_pytorch))

        # Edge case: dim2=1 (likely in_channels) with larger dim1 (likely kernel)
        # Should be detected as MLX format
        edge_case_mlx = mx.array(np.zeros((512, 3, 1)))  # (out, kernel=3, in=1)
        self.assertTrue(check_array_shape(edge_case_mlx))

        # Wrong number of dimensions
        wrong_dims_2d = mx.array(np.zeros((64, 3)))
        self.assertFalse(check_array_shape(wrong_dims_2d))

        wrong_dims_4d = mx.array(np.zeros((64, 3, 3, 3)))
        self.assertFalse(check_array_shape(wrong_dims_4d))


if __name__ == "__main__":
    unittest.main()
