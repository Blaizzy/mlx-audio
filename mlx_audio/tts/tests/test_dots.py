import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TestDotsLoadModel(unittest.TestCase):
    REQUIRED_DOTS_FILES = (
        "config.json",
        "llm_config.json",
        "core.safetensors",
        "vocoder.safetensors",
        "speaker.safetensors",
    )

    def _write_dots_checkpoint(self, root: Path, model_type=None):
        config = {}
        if model_type is not None:
            config["model_type"] = model_type

        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "llm_config.json").write_text("{}", encoding="utf-8")
        for name in self.REQUIRED_DOTS_FILES[2:]:
            (root / name).write_bytes(b"")

    def _patch_fake_dots_loader(self, return_value):
        return MagicMock(return_value=return_value)

    def _assert_dispatched_to_dots(self, dots_loader, model_path, lazy, strict):
        self.assertEqual(dots_loader.call_count, 1)
        call = dots_loader.call_args
        called_path = call.kwargs.get("model_path", call.args[0] if call.args else None)
        self.assertEqual(called_path, model_path)
        self.assertEqual(call.kwargs.get("lazy"), lazy)
        self.assertEqual(call.kwargs.get("strict"), strict)

    def _write_variant_root(self, root: Path, *variants: str):
        for variant in variants:
            variant_root = root / variant
            variant_root.mkdir(parents=True, exist_ok=True)
            self._write_dots_checkpoint(variant_root, model_type="dots_tts_mlx")

    def test_load_model_dispatches_local_dots_checkpoint(self):
        from mlx_audio.tts import utils as tts_utils

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            self._write_dots_checkpoint(model_path)

            expected_model = object()
            dots_loader = self._patch_fake_dots_loader(expected_model)

            with (
                patch("mlx_audio.tts.models.dots.load_model", dots_loader),
                patch.object(
                    tts_utils,
                    "base_load_model",
                    side_effect=AssertionError(
                        "dots checkpoints should not fall back to base_load_model"
                    ),
                ),
            ):
                result = tts_utils.load_model(model_path, lazy=True, strict=False)

        self.assertIs(result, expected_model)
        self._assert_dispatched_to_dots(
            dots_loader,
            model_path=model_path,
            lazy=True,
            strict=False,
        )

    def test_load_model_dispatches_model_type_alias_to_dots_loader(self):
        from mlx_audio.tts import utils as tts_utils

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            self._write_dots_checkpoint(model_path, model_type="dots_tts_mlx")

            expected_model = object()
            dots_loader = self._patch_fake_dots_loader(expected_model)

            with (
                patch("mlx_audio.tts.models.dots.load_model", dots_loader),
                patch.object(
                    tts_utils,
                    "base_load_model",
                    side_effect=AssertionError(
                        "dots model_type aliases should bypass base_load_model"
                    ),
                ),
            ):
                result = tts_utils.load_model(model_path, lazy=False, strict=True)

        self.assertIs(result, expected_model)
        self._assert_dispatched_to_dots(
            dots_loader,
            model_path=model_path,
            lazy=False,
            strict=True,
        )

    def test_load_model_defaults_to_int4_subdir_for_multi_variant_root(self):
        from mlx_audio.tts import utils as tts_utils

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_variant_root(repo_root, "int4", "int8")

            expected_model = object()
            dots_loader = self._patch_fake_dots_loader(expected_model)

            with (
                patch("mlx_audio.tts.models.dots.load_model", dots_loader),
                patch.object(tts_utils, "get_model_path", return_value=repo_root),
                patch.object(
                    tts_utils,
                    "base_load_model",
                    side_effect=AssertionError(
                        "dots multi-variant roots should bypass base_load_model"
                    ),
                ),
            ):
                result = tts_utils.load_model(
                    "shraey/dots-tts-mlx", lazy=True, strict=False
                )

        self.assertIs(result, expected_model)
        self._assert_dispatched_to_dots(
            dots_loader,
            model_path="shraey/dots-tts-mlx",
            lazy=True,
            strict=False,
        )
        self.assertEqual(dots_loader.call_args.kwargs.get("subdir"), None)

    def test_load_model_accepts_explicit_subdir_override_for_multi_variant_root(self):
        from mlx_audio.tts import utils as tts_utils

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_variant_root(repo_root, "int4", "mf-int8")

            expected_model = object()
            dots_loader = self._patch_fake_dots_loader(expected_model)

            with (
                patch("mlx_audio.tts.models.dots.load_model", dots_loader),
                patch.object(tts_utils, "get_model_path", return_value=repo_root),
                patch.object(
                    tts_utils,
                    "base_load_model",
                    side_effect=AssertionError(
                        "dots multi-variant roots should bypass base_load_model"
                    ),
                ),
            ):
                result = tts_utils.load_model(
                    "shraey/dots-tts-mlx",
                    lazy=False,
                    strict=True,
                    subdir="mf-int8",
                )

        self.assertIs(result, expected_model)
        self._assert_dispatched_to_dots(
            dots_loader,
            model_path="shraey/dots-tts-mlx",
            lazy=False,
            strict=True,
        )
        self.assertEqual(dots_loader.call_args.kwargs.get("subdir"), "mf-int8")


class TestDotsDirectLoader(unittest.TestCase):
    def _write_dots_checkpoint(self, root: Path, model_type="dots_tts_mlx"):
        config = {"model_type": model_type}
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "llm_config.json").write_text("{}", encoding="utf-8")
        for name in ("core.safetensors", "vocoder.safetensors", "speaker.safetensors"):
            (root / name).write_bytes(b"")

    def test_direct_dots_loader_defaults_to_int4_subdir(self):
        from mlx_audio.tts.models import dots

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_dots_checkpoint(repo_root / "int4")
            self._write_dots_checkpoint(repo_root / "int8")

            with (
                patch.object(dots, "get_model_path", return_value=repo_root),
                patch.object(
                    dots.Model, "load_runtime", autospec=True, return_value=None
                ),
            ):
                model = dots.load_model("shraey/dots-tts-mlx")

        self.assertEqual(model.config.model_path, str(repo_root / "int4"))

    def test_direct_dots_loader_accepts_explicit_subdir_override(self):
        from mlx_audio.tts.models import dots

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_dots_checkpoint(repo_root / "int4")
            self._write_dots_checkpoint(repo_root / "mf-int8")

            with (
                patch.object(dots, "get_model_path", return_value=repo_root),
                patch.object(
                    dots.Model, "load_runtime", autospec=True, return_value=None
                ),
            ):
                model = dots.load_model("shraey/dots-tts-mlx", subdir="mf-int8")

        self.assertEqual(model.config.model_path, str(repo_root / "mf-int8"))

    def test_direct_dots_loader_respects_lazy_flag(self):
        from mlx_audio.tts.models import dots

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_dots_checkpoint(repo_root / "int4")

            with (
                patch.object(dots, "get_model_path", return_value=repo_root),
                patch.object(
                    dots.Model, "load_runtime", autospec=True, return_value=None
                ) as load_runtime,
            ):
                model = dots.load_model("shraey/dots-tts-mlx", lazy=True)

        self.assertEqual(model.config.model_path, str(repo_root / "int4"))
        load_runtime.assert_not_called()


class TestDotsModel(unittest.TestCase):
    def _make_model(self, runtime, sample_rate=24000):
        from mlx_audio.tts.models.dots import Model

        model = Model.__new__(Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(sample_rate=sample_rate)
        model.runtime = runtime
        model._runtime = runtime
        return model

    def _contains_reference_path(self, value, expected_path):
        expected_path = str(expected_path)
        if isinstance(value, (str, Path)):
            return str(value) == expected_path
        if isinstance(value, dict):
            return any(
                self._contains_reference_path(item, expected_path)
                for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(
                self._contains_reference_path(item, expected_path) for item in value
            )
        return False

    def test_generate_preserves_reference_audio_path_and_normalizes_language(self):
        runtime = MagicMock()
        runtime.generate.return_value = [
            {
                "audio": mx.array([0.1, 0.2], dtype=mx.float32),
                "sample_rate": 22050,
                "samples": 2,
                "segment_idx": 0,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            reference_audio_path = Path(tmpdir) / "reference.wav"
            reference_audio_path.write_bytes(b"RIFF")

            model = self._make_model(runtime)
            next(
                model.generate(
                    "hello",
                    ref_audio=str(reference_audio_path),
                    lang_code="en",
                )
            )

        runtime.generate.assert_called_once()
        runtime_kwargs = runtime.generate.call_args.kwargs
        self.assertEqual(runtime_kwargs.get("language"), "EN")
        self.assertTrue(
            self._contains_reference_path(
                {
                    "args": runtime.generate.call_args.args,
                    "kwargs": runtime_kwargs,
                },
                reference_audio_path,
            )
        )

    def test_generate_accepts_predecoded_reference_audio_arrays(self):
        runtime = MagicMock()
        runtime.generate.return_value = [
            {
                "audio": mx.array([0.1, 0.2], dtype=mx.float32),
                "sample_rate": 22050,
                "samples": 2,
                "segment_idx": 0,
            }
        ]

        reference_audio = mx.array([0.0, 0.1, -0.1], dtype=mx.float32)
        model = self._make_model(runtime)
        next(
            model.generate(
                "hello",
                ref_audio=reference_audio,
                ref_text="reference",
                lang_code="en",
            )
        )

        runtime.generate.assert_called_once()
        runtime_kwargs = runtime.generate.call_args.kwargs
        self.assertEqual(runtime_kwargs.get("language"), "EN")
        self.assertIs(runtime_kwargs.get("prompt_audio"), reference_audio)
        self.assertEqual(runtime_kwargs.get("prompt_text"), "reference")

    @patch("builtins.print")
    @patch("mlx_audio.tts.generate.audio_write")
    def test_generate_audio_uses_dots_model_through_common_tts_api(
        self, mock_audio_write, _mock_print
    ):
        from mlx_audio.tts.generate import generate_audio

        runtime = MagicMock()
        runtime.generate.return_value = [
            {
                "audio": mx.array([0.1, 0.2], dtype=mx.float32),
                "sample_rate": 24000,
                "samples": 2,
                "segment_idx": 0,
            }
        ]

        model = self._make_model(runtime, sample_rate=24000)
        generate_audio(
            text="hello dots",
            model=model,
            max_tokens=321,
            cfg_scale=1.7,
            ddpm_steps=8,
            verbose=False,
        )

        runtime.generate.assert_called_once()
        runtime_kwargs = runtime.generate.call_args.kwargs
        self.assertEqual(runtime_kwargs.get("max_generate_length"), 321)
        self.assertEqual(runtime_kwargs.get("guidance_scale"), 1.7)
        self.assertEqual(runtime_kwargs.get("num_steps"), 8)
        for unexpected_key in (
            "voice",
            "speed",
            "temperature",
            "verbose",
            "stream",
            "streaming_interval",
            "instruct",
            "use_zero_spk_emb",
        ):
            self.assertNotIn(unexpected_key, runtime_kwargs)
        mock_audio_write.assert_called_once()

    def test_generate_yields_generation_result_like_fields_from_runtime_dict(self):
        audio = mx.array([0.25, -0.5, 0.75], dtype=mx.float32)
        runtime = MagicMock()
        runtime.generate.return_value = [
            {
                "audio": audio,
                "sample_rate": 44100,
                "samples": 3,
                "segment_idx": 7,
            }
        ]

        model = self._make_model(runtime)
        result = next(model.generate("future dots"))

        self.assertTrue(hasattr(result, "audio"))
        self.assertTrue(hasattr(result, "sample_rate"))
        self.assertTrue(hasattr(result, "samples"))
        self.assertTrue(hasattr(result, "segment_idx"))
        np.testing.assert_allclose(np.array(result.audio), np.array(audio))
        self.assertEqual(result.sample_rate, 44100)
        self.assertEqual(result.samples, 3)
        self.assertEqual(result.segment_idx, 7)

    def test_generate_populates_verbose_stats_defaults_from_runtime_dict(self):
        runtime = MagicMock()
        runtime.generate.return_value = [
            {
                "audio": mx.array([0.25, -0.5, 0.75], dtype=mx.float32),
                "sample_rate": 44100,
                "samples": 3,
                "segment_idx": 7,
                "num_patches": 4,
                "processing_time_seconds": 0.5,
            }
        ]

        model = self._make_model(runtime)
        result = next(model.generate("future dots"))

        self.assertEqual(result.prompt["tokens-per-sec"], 8.0)
        self.assertEqual(result.audio_samples["samples"], 3)
        self.assertEqual(result.audio_samples["samples-per-sec"], 6.0)


if __name__ == "__main__":
    unittest.main()
