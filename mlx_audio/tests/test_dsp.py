"""Tests for mlx_audio.dsp module."""

import subprocess
import sys

import numpy as np
import pytest


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
        "integrated_loudness",
        "normalize_loudness",
        "normalize_peak",
    ]

    for name in expected:
        assert hasattr(dsp, name), f"Missing export: {name}"


def test_lfilter_fir_and_iir():
    """Verify the local lfilter recurrence for FIR and IIR filters."""
    from mlx_audio.dsp import lfilter

    x = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    fir = lfilter([1.0, -0.5], [1.0], x)
    np.testing.assert_allclose(fir, [1.0, 1.5, 3.0])

    impulse = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    iir = lfilter([1.0], [1.0, -0.5], impulse)
    np.testing.assert_allclose(iir, [1.0, 0.5, 0.25, 0.125])


def test_utils_lazy_imports():
    """Verify utils.py uses lazy imports for TTS/STT/STS.

    Runs in subprocess to avoid interference with other tests.
    """
    code = """
import sys
from mlx_audio.utils import stft
assert "mlx_audio.tts.utils" not in sys.modules, "TTS utils was imported"
assert "mlx_audio.stt.utils" not in sys.modules, "STT utils was imported"
assert "mlx_audio.sts.utils" not in sys.modules, "STS utils was imported"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Lazy import failed: {result.stderr}"


def test_integrated_loudness_matches_reference_values():
    """Verify BS.1770 loudness matches fixed reference outputs."""
    from mlx_audio.dsp import integrated_loudness

    rng = np.random.default_rng(0)
    mono = (rng.standard_normal(24000) * 0.02).astype(np.float64)
    stereo = (rng.standard_normal((24000, 2)) * 0.015).astype(np.float64)

    assert integrated_loudness(mono, 24000) == pytest.approx(
        -31.147497580698033, abs=1e-12
    )
    assert integrated_loudness(stereo, 24000) == pytest.approx(
        -30.587340400145717, abs=1e-12
    )


def test_normalize_loudness_matches_reference_values():
    """Verify loudness normalization matches fixed reference outputs."""
    from mlx_audio.dsp import integrated_loudness, normalize_loudness

    rng = np.random.default_rng(0)
    mono = (rng.standard_normal(24000) * 0.02).astype(np.float64)

    measured = integrated_loudness(mono, 24000, block_size=0.4)
    normalized = normalize_loudness(mono, measured, -18.0)

    assert np.max(np.abs(normalized)) == pytest.approx(0.4083656963780373, abs=1e-12)
    np.testing.assert_allclose(
        normalized[:5],
        np.array(
            [
                0.011424693401069328,
                -0.01200393625946315,
                0.058193108743361296,
                0.009531930078445609,
                -0.04867452152305382,
            ]
        ),
        atol=1e-12,
        rtol=0.0,
    )


def test_normalize_peak_matches_reference_values():
    """Verify peak normalization matches fixed reference outputs."""
    from mlx_audio.dsp import normalize_peak

    data = np.linspace(-0.5, 0.5, 32, dtype=np.float64)
    normalized = normalize_peak(data, -1.0)

    assert np.max(np.abs(normalized)) == pytest.approx(0.8912509381337456, abs=1e-12)
    np.testing.assert_allclose(
        normalized[:5],
        np.array(
            [
                -0.8912509381337456,
                -0.8337508776089878,
                -0.77625081708423,
                -0.7187507565594722,
                -0.6612506960347144,
            ]
        ),
        atol=1e-12,
        rtol=0.0,
    )


def test_integrated_loudness_validation_matches_previous_behavior():
    """Verify the public helper keeps the old validation semantics."""
    from mlx_audio.dsp import integrated_loudness

    with pytest.raises(ValueError, match="Data must be floating point."):
        integrated_loudness(np.arange(10, dtype=np.int16), 24000)

    with pytest.raises(
        ValueError, match="Audio must have length greater than the block size."
    ):
        integrated_loudness(np.zeros(100, dtype=np.float64), 24000)
