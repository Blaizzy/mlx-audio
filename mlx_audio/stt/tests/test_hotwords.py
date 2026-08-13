"""Uniform ``hotwords`` support across ASR backends (mlx-vlm #1781, part one).

Each supporting model folds a structured ``hotwords`` list into its own native
prompt field inside its ``generate`` forward loop; models without a hook ignore
it (silent drop). These tests stay weight-free: they exercise the shared merge
helper and introspect each model's ``generate`` signature.
"""

import ast
import inspect
from pathlib import Path

from mlx_audio.stt.utils import merge_hotwords


class TestMergeHotwords:
    def test_none_is_noop(self):
        assert merge_hotwords("base", None) == "base"
        assert merge_hotwords(None, None) is None

    def test_empty_or_blank_terms_noop(self):
        assert merge_hotwords("base", []) == "base"
        assert merge_hotwords("base", ["", "   ", None]) == "base"

    def test_terms_only(self):
        assert merge_hotwords(None, ["Nativ", "MLX"]) == "Nativ, MLX"

    def test_appends_to_existing_base(self):
        assert (
            merge_hotwords("Prior text", ["Nativ", "MLX"]) == "Prior text\nNativ, MLX"
        )

    def test_strips_and_drops_blanks(self):
        assert merge_hotwords(None, [" Nativ ", "", "MLX"]) == "Nativ, MLX"


# model file -> the native prompt field hotwords must fold into
_NATIVE_FIELD = {
    "qwen3_asr/qwen3_asr.py": "system_prompt",
    "whisper/whisper.py": "initial_prompt",
    "vibevoice_asr/vibevoice_asr.py": "context",
    "moss_transcribe_diarize/moss_transcribe_diarize.py": "prompt",
}

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class TestGenerateAcceptsHotwords:
    def _generate_args(self, rel_path):
        tree = ast.parse((_MODELS_DIR / rel_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "generate":
                args = [a.arg for a in node.args.args + node.args.kwonlyargs]
                if "hotwords" in args or "audio" in args:
                    return args
        return []

    def test_each_supporting_model_exposes_hotwords_and_native_field(self):
        for rel_path, native in _NATIVE_FIELD.items():
            args = self._generate_args(rel_path)
            assert "hotwords" in args, f"{rel_path}: generate() missing hotwords"
            assert native in args, f"{rel_path}: generate() missing {native}"

    def test_merge_helper_is_used_in_each_model(self):
        for rel_path in _NATIVE_FIELD:
            src = (_MODELS_DIR / rel_path).read_text()
            assert "merge_hotwords" in src, f"{rel_path}: does not call merge_hotwords"
