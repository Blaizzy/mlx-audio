import sys  # Import sys to patch argv
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_audio.convert import (
    Domain,
    _discover_detection_hints,
    configure_parser,
    get_model_type,
    main,
)


class TestModelTypeDetection(unittest.TestCase):
    def test_higgs_v3_upstream_model_type_resolves_to_local_implementation(self):
        with patch(
            "mlx_audio.convert.get_model_types", return_value={"higgs_audio_v3"}
        ):
            hints = _discover_detection_hints(Domain.TTS.value)

        self.assertIn(
            "higgs_multimodal_qwen3",
            hints["model_type_aliases"]["higgs_audio_v3"],
        )

        with (
            patch(
                "mlx_audio.convert.get_model_types",
                return_value={"dense", "higgs_audio_v3"},
            ),
            patch("mlx_audio.convert.get_detection_hints", return_value=hints),
        ):
            model_type = get_model_type(
                {"model_type": "higgs_multimodal_qwen3"},
                Path("model"),
                Domain.TTS,
            )

        self.assertEqual(model_type, "higgs_audio_v3")

    def test_config_key_score_ties_use_model_type_name_as_tiebreaker(self):
        hints = {
            "model_type_aliases": {},
            "config_keys": {
                "z_model": {"shared_key"},
                "a_model": {"shared_key"},
            },
            "path_patterns": {},
        }

        with (
            patch(
                "mlx_audio.convert.get_model_types",
                return_value={"z_model", "a_model"},
            ),
            patch("mlx_audio.convert.get_detection_hints", return_value=hints),
        ):
            model_type = get_model_type({"shared_key": True}, Path("model"), Domain.TTS)

        self.assertEqual(model_type, "a_model")

    def test_fallback_model_type_is_deterministic(self):
        hints = {
            "model_type_aliases": {},
            "config_keys": {},
            "path_patterns": {},
        }

        with (
            patch(
                "mlx_audio.convert.get_model_types",
                return_value={"z_model", "a_model"},
            ),
            patch("mlx_audio.convert.get_detection_hints", return_value=hints),
        ):
            model_type = get_model_type({}, Path("model"), Domain.TTS)

        self.assertEqual(model_type, "a_model")


class TestConvert(unittest.TestCase):
    def setUp(self):
        self.parser = configure_parser()

        # Mock the actual convert function
        self.convert_mock = MagicMock()
        self.patcher = patch("mlx_audio.convert.convert", new=self.convert_mock)
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
            q_group_size=None,
            q_bits=None,
            q_mode="affine",
            quant_predicate=None,
            dtype="float16",
            upload_repo=None,
            revision=None,
            dequantize=False,
            model_domain=None,
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
            q_mode="affine",
            quant_predicate=None,
            dtype=None,  # Default dtype is None
            upload_repo=None,
            revision=None,
            dequantize=False,
            model_domain=None,
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
            q_mode="affine",
            quant_predicate=None,
            dtype=None,  # Default dtype is None
            upload_repo=None,
            revision=None,
            dequantize=False,
            model_domain=None,
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
                    q_group_size=None,  # Default q_group_size
                    q_bits=None,  # Default q_bits
                    q_mode="affine",
                    quant_predicate=recipe,
                    dtype=None,  # Default dtype is None
                    upload_repo=None,  # Default upload_repo
                    revision=None,
                    dequantize=False,  # Default dequantize
                    model_domain=None,
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
            q_group_size=None,
            q_bits=None,
            q_mode="affine",
            quant_predicate=None,
            dtype=None,  # Default dtype is None
            upload_repo=None,
            revision=None,
            dequantize=True,
            model_domain=None,
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
            q_group_size=None,
            q_bits=None,
            q_mode="affine",
            quant_predicate=None,
            dtype=None,  # Default dtype is None
            upload_repo="my/repo",
            revision=None,
            dequantize=False,
            model_domain=None,
        )

    def test_q_mode_argument(self):
        test_args = ["--hf-path", "dummy_hf", "--quantize", "--q-mode", "mxfp4"]
        with patch.object(sys, "argv", ["convert.py"] + test_args):
            main()

        self.convert_mock.assert_called_once_with(
            hf_path="dummy_hf",
            mlx_path="mlx_model",
            quantize=True,
            q_group_size=None,
            q_bits=None,
            q_mode="mxfp4",
            quant_predicate=None,
            dtype=None,
            upload_repo=None,
            revision=None,
            dequantize=False,
            model_domain=None,
        )


if __name__ == "__main__":
    unittest.main()
