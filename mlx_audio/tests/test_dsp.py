"""Tests for mlx_audio.dsp module."""

import sys

import pytest


def test_dsp_import_isolation():
    """Verify dsp.py doesn't import TTS/STT modules."""
    # Clear any cached imports
    modules_to_remove = [mod for mod in sys.modules.keys() if "mlx_audio" in mod]
    for mod in modules_to_remove:
        del sys.modules[mod]

    from mlx_audio.dsp import stft

    assert "mlx_audio.tts" not in sys.modules
    assert "mlx_audio.stt" not in sys.modules


def test_dsp_backward_compat():
    """Verify backward compatible imports from utils.py still work."""
    from mlx_audio.utils import hanning, istft, mel_filters, stft

    # Just verify they're callable
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
    """Verify utils.py uses lazy imports for TTS/STT."""
    # Clear any cached imports
    modules_to_remove = [mod for mod in sys.modules.keys() if "mlx_audio" in mod]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Import just the DSP functions from utils
    from mlx_audio.utils import stft

    # TTS/STT should not be loaded yet (lazy imports)
    assert "mlx_audio.tts.utils" not in sys.modules
    assert "mlx_audio.stt.utils" not in sys.modules
