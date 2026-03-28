import sys  # Import sys to patch argv
import unittest
from unittest.mock import MagicMock, patch

import mlx.nn as nn

from mlx_audio.convert import (
    _is_moss_tts_export_exempt,
    build_quant_predicate,
    configure_parser,
    main,
    resolve_quant_args,
)


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
                    q_group_size=None,
                    q_bits=None,
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

    def test_quantized_conversion_with_q_mode(self):
        test_args = [
            "--hf-path",
            "dummy_hf",
            "--quantize",
            "--q-mode",
            "mxfp8",
        ]
        with patch.object(sys, "argv", ["convert.py"] + test_args):
            main()

        self.convert_mock.assert_called_once_with(
            hf_path="dummy_hf",
            mlx_path="mlx_model",
            quantize=True,
            q_group_size=None,
            q_bits=None,
            q_mode="mxfp8",
            quant_predicate=None,
            dtype=None,
            upload_repo=None,
            revision=None,
            dequantize=False,
            model_domain=None,
        )

    def test_resolve_quant_args_uses_mode_defaults(self):
        self.assertEqual(resolve_quant_args(None, None, "affine"), (64, 4))
        self.assertEqual(resolve_quant_args(None, None, "mxfp8"), (32, 8))
        self.assertEqual(resolve_quant_args(128, 6, "affine"), (128, 6))

    def test_build_quant_predicate_uses_group_size_and_model_predicate(self):
        class DummyModel:
            def model_quant_predicate(self, path, module):
                _ = module
                return not path.startswith("skip")

        pred = build_quant_predicate(DummyModel(), 32)

        allowed = nn.Linear(32, 64, bias=False)
        disallowed = nn.Linear(48, 64, bias=False)

        self.assertTrue(pred("language_model.layers.0.self_attn.q_proj", allowed))
        self.assertFalse(pred("skip.layer", allowed))
        self.assertFalse(pred("language_model.layers.0.self_attn.q_proj", disallowed))

    def test_build_quant_predicate_excludes_moss_tts_embedding_and_heads(self):
        class DummyModel:
            def model_quant_predicate(self, path, module):
                _ = (path, module)
                return True

        pred = build_quant_predicate(DummyModel(), 32)
        allowed = nn.Linear(32, 64, bias=False)

        self.assertFalse(pred("language_model.embed_tokens", allowed))
        self.assertFalse(pred("lm_heads.0", allowed))
        self.assertTrue(pred("language_model.layers.0.self_attn.q_proj", allowed))

    def test_is_moss_tts_export_exempt(self):
        self.assertTrue(_is_moss_tts_export_exempt("language_model.embed_tokens"))
        self.assertTrue(_is_moss_tts_export_exempt("lm_heads.12"))
        self.assertFalse(
            _is_moss_tts_export_exempt("language_model.layers.0.self_attn.q_proj")
        )


if __name__ == "__main__":
    unittest.main()
