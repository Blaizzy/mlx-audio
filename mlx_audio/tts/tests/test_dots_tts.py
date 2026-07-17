import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TestDotsDirectLoader(unittest.TestCase):
    def _write_dots_checkpoint(self, root: Path, model_type="dots_tts_mlx"):
        config = {"model_type": model_type}
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "llm_config.json").write_text("{}", encoding="utf-8")
        for name in ("core.safetensors", "vocoder.safetensors", "speaker.safetensors"):
            (root / name).write_bytes(b"")

    def test_direct_dots_loader_defaults_to_int4_subdir(self):
        from mlx_audio.tts.models import dots_tts as dots

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
                model = dots.load_model("mlx-community/dots-tts-mlx")

        self.assertEqual(model.config.model_path, str(repo_root / "int4"))

    def test_direct_dots_loader_accepts_explicit_subdir_override(self):
        from mlx_audio.tts.models import dots_tts as dots

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
                model = dots.load_model("mlx-community/dots-tts-mlx", subdir="mf-int8")

        self.assertEqual(model.config.model_path, str(repo_root / "mf-int8"))

    def test_direct_dots_loader_respects_lazy_flag(self):
        from mlx_audio.tts.models import dots_tts as dots

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_dots_checkpoint(repo_root / "int4")

            with (
                patch.object(dots, "get_model_path", return_value=repo_root),
                patch.object(
                    dots.Model, "load_runtime", autospec=True, return_value=None
                ) as load_runtime,
            ):
                model = dots.load_model("mlx-community/dots-tts-mlx", lazy=True)

        self.assertEqual(model.config.model_path, str(repo_root / "int4"))
        load_runtime.assert_not_called()


class TestDotsModel(unittest.TestCase):
    def _make_model(self, runtime, sample_rate=24000):
        from mlx_audio.tts.models.dots_tts import Model

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
