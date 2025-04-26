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
        """Test check_array_shape function."""
        # Valid shape: out_channels >= kH == kW
        valid_array = mx.array(np.zeros((64, 3, 3)))
        self.assertTrue(check_array_shape(valid_array))

        # Invalid shape: kH != kW
        invalid_array1 = mx.array(np.zeros((64, 3, 4)))
        self.assertFalse(check_array_shape(invalid_array1))

        # Invalid shape: out_channels < kH
        invalid_array2 = mx.array(np.zeros((2, 3, 3)))
        self.assertFalse(check_array_shape(invalid_array2))

        # Invalid shape: wrong number of dimensions
        invalid_array3 = mx.array(np.zeros((64, 3)))
        self.assertFalse(check_array_shape(invalid_array3))

        # Invalid shape: wrong number of dimensions
        invalid_array4 = mx.array(np.zeros((64, 3, 3, 3)))
        self.assertFalse(check_array_shape(invalid_array4))


if __name__ == "__main__":
    unittest.main()
