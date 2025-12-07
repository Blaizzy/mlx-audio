"""Tests for mlx_audio.dsp module."""

import subprocess
import sys


def test_dsp_import_isolation():
    """Verify dsp.py doesn't import TTS/STT modules.

    Runs in subprocess to avoid interference with other tests.
    """
    code = """
import sys
from mlx_audio.dsp import stft
assert "mlx_audio.tts" not in sys.modules, "TTS was imported"
assert "mlx_audio.stt" not in sys.modules, "STT was imported"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import isolation failed: {result.stderr}"


def test_dsp_backward_compat():
    """Verify backward compatible imports from utils.py still work."""
    from mlx_audio.utils import hanning, istft, mel_filters, stft

    assert callable(stft)
    assert callable(istft)
    assert callable(mel_filters)
    assert callable(hanning)


def test_dsp_all_exports():
    """Verify __all__ exports work correctly."""
    from mlx_audio import dsp

    expected = [
        "hanning",
        "hamming",
        "blackman",
        "bartlett",
        "STR_TO_WINDOW_FN",
        "stft",
        "istft",
        "mel_filters",
    ]

    for name in expected:
        assert hasattr(dsp, name), f"Missing export: {name}"


def test_utils_lazy_imports():
    """Verify utils.py uses lazy imports for TTS/STT.

    Runs in subprocess to avoid interference with other tests.
    """
    code = """
import sys
from mlx_audio.utils import stft
assert "mlx_audio.tts.utils" not in sys.modules, "TTS utils was imported"
assert "mlx_audio.stt.utils" not in sys.modules, "STT utils was imported"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Lazy import failed: {result.stderr}"
