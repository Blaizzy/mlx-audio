"""Test that lazy imports work correctly for modular installation.

Uses subprocess isolation because pytest imports all test files during collection,
which pollutes sys.modules before tests run (e.g., test_server.py imports miniaudio).
"""

import subprocess
import sys


def test_stt_utils_no_eager_imports():
    """Importing stt.utils should not import miniaudio or scipy."""
    code = """
import sys
import mlx_audio.stt.utils
assert "miniaudio" not in sys.modules, f"miniaudio was eagerly imported"
assert "scipy" not in sys.modules, f"scipy was eagerly imported"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"STT lazy import failed: {result.stderr}"


def test_tts_utils_no_eager_imports():
    """Importing tts.utils should not import transformers or mlx_lm."""
    code = """
import sys
import mlx_audio.tts.utils
assert "transformers" not in sys.modules, f"transformers was eagerly imported"
assert "mlx_lm" not in sys.modules, f"mlx_lm was eagerly imported"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"TTS lazy import failed: {result.stderr}"


def test_codec_no_eager_imports():
    """Importing codec should not import miniaudio."""
    code = """
import sys
import mlx_audio.codec
assert "miniaudio" not in sys.modules, f"miniaudio was eagerly imported"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Codec lazy import failed: {result.stderr}"
