import sys  # Import sys to patch argv
import unittest
from unittest.mock import MagicMock, patch

from mlx_audio.tts.convert import configure_parser, main


class TestConvert(unittest.TestCase):
    def setUp(self):
        self.parser = configure_parser()

        # Mock the actual convert function
        self.convert_mock = MagicMock()
        self.patcher = patch("mlx_audio.tts.convert.convert", new=self.convert_mock)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_basic_conversion(self):
        test_args = [
            "--hf-path",
            "dummy_hf",
            "--mlx-path",
            "dummy_mlx",
            "--dtype",
            "float16",
        ]
        # Patch sys.argv for this test run
        with patch.object(sys, "argv", ["convert.py"] + test_args):
            main()

        self.convert_mock.assert_called_once_with(
            hf_path="dummy_hf",
            mlx_path="dummy_mlx",
            quantize=False,
            q_group_size=64,
            q_bits=4,
            quant_predicate=None,
            dtype="float16",
            upload_repo=None,
            dequantize=False,
        )

    def test_quantized_conversion(self):
        test_args = [
            "--hf-path",
            "dummy_hf",
            "--quantize",
            "--q-group-size",
            "128",
            "--q-bits",
            "8",
        ]
        # Patch sys.argv for this test run
        with patch.object(sys, "argv", ["convert.py"] + test_args):
            main()

        self.convert_mock.assert_called_once_with(
            hf_path="dummy_hf",
            mlx_path="mlx_model",  # Default mlx_path
            quantize=True,
            q_group_size=128,
            q_bits=8,
            quant_predicate=None,
            dtype="float16",  # Should be ignored when quantize=True
            upload_repo=None,
            dequantize=False,
        )

    def test_quantized_conversion_invalid_group_size_raises_error(self):
        """Tests if main raises ValueError for invalid group size."""
        test_args = [
            "--hf-path",
            "dummy_hf",
            "--quantize",
            "--q-group-size",
            "100",  # Invalid: not 64 or 128
            "--q-bits",
            "4",
        ]

        # Configure the mock to raise ValueError when called with q_group_size=100
        def side_effect(*args, **kwargs):
            if kwargs.get("q_group_size") == 100:
                raise ValueError(
                    "[quantize] The requested group size 100 is not supported."
                )
            return MagicMock()  # Default return for other calls if needed

        self.convert_mock.side_effect = side_effect

        # Patch sys.argv and assert ValueError is raised
        with patch.object(sys, "argv", ["convert.py"] + test_args):
            with self.assertRaisesRegex(
                ValueError, "requested group size 100 is not supported"
            ):
                main()

        # Verify the mock was called (even though it raised an error)
        self.convert_mock.assert_called_once_with(
            hf_path="dummy_hf",
            mlx_path="mlx_model",
            quantize=True,
            q_group_size=100,
            q_bits=4,
            quant_predicate=None,
            dtype="float16",
            upload_repo=None,
            dequantize=False,
        )

    def test_quantization_recipes(self):
        for recipe in ["mixed_2_6", "mixed_3_6", "mixed_4_6"]:
            with self.subTest(recipe=recipe):
                self.convert_mock.reset_mock()  # Reset mock for each subtest
                test_args = ["--hf-path", "dummy_hf", "--quant-predicate", recipe]
                # Patch sys.argv for this test run
                with patch.object(sys, "argv", ["convert.py"] + test_args):
                    main()

                self.convert_mock.assert_called_once_with(  # Changed to assert_called_once_with
                    hf_path="dummy_hf",
                    mlx_path="mlx_model",  # Default mlx_path
                    quantize=False,  # Default quantize
                    q_group_size=64,  # Default q_group_size
                    q_bits=4,  # Default q_bits
                    quant_predicate=recipe,
                    dtype="float16",  # Default dtype
                    upload_repo=None,  # Default upload_repo
                    dequantize=False,  # Default dequantize
                )
                # No need to reset mock here, it's handled at the start of the loop

    def test_dequantize_flag(self):
        test_args = ["--hf-path", "dummy_hf", "--dequantize"]
        # Patch sys.argv for this test run
        with patch.object(sys, "argv", ["convert.py"] + test_args):
            main()

        self.convert_mock.assert_called_once_with(
            hf_path="dummy_hf",
            mlx_path="mlx_model",  # Default mlx_path
            quantize=False,
            q_group_size=64,
            q_bits=4,
            quant_predicate=None,
            dtype="float16",
            upload_repo=None,
            dequantize=True,
        )

    def test_upload_repo_argument(self):
        test_args = ["--hf-path", "dummy_hf", "--upload-repo", "my/repo"]
        # Patch sys.argv for this test run
        with patch.object(sys, "argv", ["convert.py"] + test_args):
            main()

        self.convert_mock.assert_called_once_with(
            hf_path="dummy_hf",
            mlx_path="mlx_model",  # Default mlx_path
            quantize=False,
            q_group_size=64,
            q_bits=4,
            quant_predicate=None,
            dtype="float16",
            upload_repo="my/repo",
            dequantize=False,
        )


if __name__ == "__main__":
    unittest.main()
