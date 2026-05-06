"""Tests for Whisper post_load_hook canonical-openai processor fallback (#645).

mlx-community whisper conversions ship weights only — no
``preprocessor_config.json`` or tokenizer files — so
``WhisperProcessor.from_pretrained(model_path)`` raises and the model
ends up with ``_processor = None``. ``get_tokenizer()`` then crashes on
the first ``generate()`` call.

The fix attempts a fallback: read the architecture dimensions from
``config.json`` and resolve to the canonical ``openai/whisper-*`` repo
that produced this architecture. Identifying by dims is robust to
arbitrary repo naming (``whisper-large-v3-mlx-4bit``,
``whisper-base.en-mlx``) and user-renamed local directories.
"""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

# Canonical architecture signatures: each unique combination of dims maps to
# exactly one openai/whisper-* repo. Sourced from openai/whisper config.json.
_TINY = {
    "n_audio_state": 384,
    "n_mels": 80,
    "n_audio_layer": 4,
    "n_text_layer": 4,
    "n_vocab": 51865,
}
_BASE = {
    "n_audio_state": 512,
    "n_mels": 80,
    "n_audio_layer": 6,
    "n_text_layer": 6,
    "n_vocab": 51865,
}
_BASE_EN = {
    "n_audio_state": 512,
    "n_mels": 80,
    "n_audio_layer": 6,
    "n_text_layer": 6,
    "n_vocab": 51864,
}
_MEDIUM = {
    "n_audio_state": 1024,
    "n_mels": 80,
    "n_audio_layer": 24,
    "n_text_layer": 24,
    "n_vocab": 51865,
}
_LARGE_V3 = {
    "n_audio_state": 1280,
    "n_mels": 128,
    "n_audio_layer": 32,
    "n_text_layer": 32,
    "n_vocab": 51866,
}
_LARGE_V3_TURBO = {
    "n_audio_state": 1280,
    "n_mels": 128,
    "n_audio_layer": 32,
    "n_text_layer": 4,
    "n_vocab": 51866,
}


def _write_whisper_config(repo: Path, dims: dict, hf_format: bool = False) -> None:
    """Write a whisper config.json with the given dims.

    ``hf_format=True`` uses HuggingFace Transformers key names
    (``d_model``, ``num_mel_bins``, ``encoder_layers``, …); the default
    openai/mlx format uses ``n_audio_state``, ``n_mels``, etc.
    """
    if hf_format:
        cfg = {
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
            "d_model": dims["n_audio_state"],
            "num_mel_bins": dims["n_mels"],
            "encoder_layers": dims["n_audio_layer"],
            "decoder_layers": dims["n_text_layer"],
            "vocab_size": dims["n_vocab"],
        }
    else:
        cfg = {"model_type": "whisper", **dims}
    (repo / "config.json").write_text(json.dumps(cfg))


class _FakeModel:
    """post_load_hook only touches ``_processor`` / ``set_alignment_heads``."""

    _processor = "unset"

    def set_alignment_heads(self, *_args, **_kwargs):
        pass


class TestWhisperProcessorFallback(unittest.TestCase):
    def setUp(self):
        from mlx_audio.stt.models.whisper.whisper import Model

        self.Model = Model
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

    def _make_repo(self, name: str, dims=None, hf_format: bool = False) -> Path:
        d = self.tmpdir / name
        d.mkdir()
        if dims is not None:
            _write_whisper_config(d, dims, hf_format=hf_format)
        return d

    def _run_with_fake_processor(self, repo: Path, fake):
        model = _FakeModel()
        with patch("transformers.WhisperProcessor.from_pretrained", side_effect=fake):
            self.Model.post_load_hook(model, repo)
        return model

    # -------- canonical resolution by architecture dims --------

    def test_tiny_dims_resolve_regardless_of_dir_name(self):
        """Architecture identified by dims, not by directory naming."""
        repo = self._make_repo("renamed-experimental-dir", _TINY)
        sentinel = MagicMock(name="canonical_processor")

        def fake(path):
            if str(path) == str(repo):
                raise OSError("missing")
            self.assertEqual(path, "openai/whisper-tiny")
            return sentinel

        model = self._run_with_fake_processor(repo, fake)
        self.assertIs(model._processor, sentinel)

    def test_quantized_variant_resolves_via_dims(self):
        """A repo whose name has -4bit / -8bit suffix shares dims with its
        unquantized peer; the matcher must look at dims, not the suffix."""
        repo = self._make_repo("whisper-base-mlx-4bit", _BASE)
        sentinel = MagicMock()

        def fake(path):
            if str(path) == str(repo):
                raise OSError("missing")
            self.assertEqual(path, "openai/whisper-base")
            return sentinel

        model = self._run_with_fake_processor(repo, fake)
        self.assertIs(model._processor, sentinel)

    def test_large_v3_distinguished_from_v2_by_mels_and_vocab(self):
        repo = self._make_repo("whisper-large-v3-mlx", _LARGE_V3)
        sentinel = MagicMock()

        def fake(path):
            if str(path) == str(repo):
                raise OSError("missing")
            self.assertEqual(path, "openai/whisper-large-v3")
            return sentinel

        model = self._run_with_fake_processor(repo, fake)
        self.assertIs(model._processor, sentinel)

    def test_large_v3_turbo_distinguished_by_decoder_layers(self):
        """large-v3-turbo has n_text_layer=4; vanilla large-v3 has 32."""
        repo = self._make_repo("whisper-large-v3-turbo", _LARGE_V3_TURBO)
        sentinel = MagicMock()

        def fake(path):
            if str(path) == str(repo):
                raise OSError("missing")
            self.assertEqual(path, "openai/whisper-large-v3-turbo")
            return sentinel

        model = self._run_with_fake_processor(repo, fake)
        self.assertIs(model._processor, sentinel)

    def test_english_only_variant_resolves_to_dot_en(self):
        """vocab_size=51864 (no language tokens) → .en variant."""
        repo = self._make_repo("whisper-base.en-mlx", _BASE_EN)
        sentinel = MagicMock()

        def fake(path):
            if str(path) == str(repo):
                raise OSError("missing")
            self.assertEqual(path, "openai/whisper-base.en")
            return sentinel

        model = self._run_with_fake_processor(repo, fake)
        self.assertIs(model._processor, sentinel)

    def test_hf_transformers_format_config_keys_resolve(self):
        """Configs using d_model / num_mel_bins (HF Transformers naming) resolve."""
        repo = self._make_repo("whisper-large-v3-asr-4bit", _LARGE_V3, hf_format=True)
        sentinel = MagicMock()

        def fake(path):
            if str(path) == str(repo):
                raise OSError("missing")
            self.assertEqual(path, "openai/whisper-large-v3")
            return sentinel

        model = self._run_with_fake_processor(repo, fake)
        self.assertIs(model._processor, sentinel)

    # -------- behavior preservation / no-fallback paths --------

    def test_missing_config_json_skips_fallback(self):
        repo = self._make_repo("whisper-foo-mlx", dims=None)
        attempts = []

        def fake(path):
            attempts.append(str(path))
            raise OSError("missing")

        model = self._run_with_fake_processor(repo, fake)
        self.assertIsNone(model._processor)
        self.assertEqual(attempts, [str(repo)])  # no canonical retry

    def test_unknown_dims_skip_fallback(self):
        weird = {**_TINY, "n_audio_state": 999}  # not a real whisper arch
        repo = self._make_repo("whisper-experimental", weird)
        attempts = []

        def fake(path):
            attempts.append(str(path))
            raise OSError("missing")

        model = self._run_with_fake_processor(repo, fake)
        self.assertIsNone(model._processor)
        self.assertEqual(attempts, [str(repo)])

    def test_local_success_skips_fallback(self):
        repo = self._make_repo("whisper-base-mlx", _BASE)
        local_processor = MagicMock(name="local_processor")
        attempts = []

        def fake(path):
            attempts.append(str(path))
            return local_processor

        model = self._run_with_fake_processor(repo, fake)
        self.assertIs(model._processor, local_processor)
        self.assertEqual(attempts, [str(repo)])

    def test_value_error_propagates_no_fallback(self):
        """Non-OSError exceptions (corrupt tokenizer.json, etc.) must propagate
        rather than silently substitute the canonical processor — that would
        be a vocab-mismatch silent corruption for fine-tuned local models."""
        repo = self._make_repo("whisper-base-mlx", _BASE)

        def fake(path):
            raise ValueError("corrupt tokenizer.json")

        model = _FakeModel()
        with patch("transformers.WhisperProcessor.from_pretrained", side_effect=fake):
            with self.assertRaises(ValueError):
                self.Model.post_load_hook(model, repo)

    def test_canonical_lookup_failure_leaves_processor_none(self):
        repo = self._make_repo("whisper-tiny-mlx", _TINY)

        def fake(path):
            raise OSError("offline / 401 / etc")

        model = self._run_with_fake_processor(repo, fake)
        self.assertIsNone(model._processor)


if __name__ == "__main__":
    unittest.main()
