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


def test_istft_cache_can_constrain_inverse_frames_to_window_envelope():
    import mlx.core as mx

    from mlx_audio.dsp import ISTFTCache, hanning

    rng = np.random.default_rng(0)
    real = mx.array(rng.normal(size=(2, 9, 8)).astype(np.float32) * 8.0)
    imag = mx.array(rng.normal(size=(2, 9, 8)).astype(np.float32) * 8.0)
    window = hanning(16, periodic=True).astype(mx.float32)
    cache = ISTFTCache()

    unconstrained = cache.istft(
        real,
        imag,
        n_fft=16,
        hop_length=4,
        win_length=16,
        window=window,
        center=False,
    )
    constrained = cache.istft(
        real,
        imag,
        n_fft=16,
        hop_length=4,
        win_length=16,
        window=window,
        center=False,
        constrain_value_range=True,
    )
    mx.eval(unconstrained, constrained)

    assert float(mx.max(mx.abs(unconstrained - constrained))) > 0.1
    assert float(mx.max(mx.abs(constrained))) <= 1.0 + 1e-6


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


def _sine(peak_dbfs, seconds, rate, freq=997.0):
    n = int(seconds * rate)
    return 10.0 ** (peak_dbfs / 20.0) * np.sin(2 * np.pi * freq * np.arange(n) / rate)


def test_k_weighting_matches_bs1770_coefficients():
    """The two K-weighting stages must equal the coefficients ITU-R BS.1770
    tabulates for 48 kHz in Table 1 (spherical head) and Table 2 (RLB)."""
    from mlx_audio.dsp import (
        _K_WEIGHT_HIGHPASS_FREQ,
        _K_WEIGHT_HIGHPASS_Q,
        _K_WEIGHT_SHELF_FREQ,
        _K_WEIGHT_SHELF_GAIN_DB,
        _K_WEIGHT_SHELF_Q,
        _biquad_coefficients,
    )

    shelf_b, shelf_a = _biquad_coefficients(
        _K_WEIGHT_SHELF_GAIN_DB,
        _K_WEIGHT_SHELF_Q,
        _K_WEIGHT_SHELF_FREQ,
        48000,
        "high_shelf",
    )
    np.testing.assert_allclose(
        shelf_b, [1.53512485958697, -2.69169618940638, 1.19839281085285], atol=1e-12
    )
    np.testing.assert_allclose(
        shelf_a, [1.0, -1.69065929318241, 0.73248077421585], atol=1e-12
    )

    pass_b, pass_a = _biquad_coefficients(
        0.0, _K_WEIGHT_HIGHPASS_Q, _K_WEIGHT_HIGHPASS_FREQ, 48000, "high_pass"
    )
    np.testing.assert_allclose(pass_b, [1.0, -2.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        pass_a, [1.0, -1.99004745483398, 0.99007225036621], atol=1e-12
    )

    # BS.1770 Note 1: the -0.691 constant cancels the K-weighting gain at 997 Hz,
    # so that gain has to be 0.691 dB.
    z = np.exp(-2j * np.pi * 997.0 / 48000)

    def gain_db(b, a):
        num = b[0] + b[1] * z + b[2] * z**2
        den = a[0] + a[1] * z + a[2] * z**2
        return 20.0 * np.log10(np.abs(num / den))

    total = gain_db(shelf_b, shelf_a) + gain_db(pass_b, pass_a)
    assert total == pytest.approx(0.691, abs=1e-3)


def test_integrated_loudness_matches_bs1770_997hz_anchor():
    """BS.1770: a 0 dB FS 997 Hz sine on one channel reads -3.01 LKFS, and the
    scale is 1 LKFS per dB."""
    from mlx_audio.dsp import integrated_loudness

    for peak_dbfs in (0.0, -20.0, -40.0):
        measured = integrated_loudness(_sine(peak_dbfs, 2.0, 48000), 48000)
        assert measured == pytest.approx(peak_dbfs - 3.01, abs=0.01)


def test_integrated_loudness_block_hop_11025hz():
    """At 44.1 kHz / 4 the 400 ms block is 4410 samples and the hop is
    round(1102.5) = 1102, so block starts are exact integer multiples of the
    hop (uniform spacing, no drift), and the 997 Hz anchor still holds."""
    from mlx_audio.dsp import integrated_loudness

    rate = 11025
    assert int(round(0.4 * rate)) == 4410
    assert int(round(0.4 * 0.25 * rate)) == 1102

    # -23 dB FS 997 Hz reads -23 - 3.01 LKFS; the K-weighting is redesigned
    # per rate from the analogue prototypes, which costs < 0.05 dB (48 kHz
    # reads -26.010, 44.1 kHz -26.007).
    measured = integrated_loudness(_sine(-23.0, 2.0, rate), rate)
    assert measured == pytest.approx(-23.0 - 3.01, abs=0.05)


def test_integrated_loudness_ignores_incomplete_final_block():
    """BS.1770: "Incomplete gating blocks at the end of the measurement interval
    are not used", so trailing samples that do not complete a block cannot move
    the reading and a steady tone measures the same at any length."""
    from mlx_audio.dsp import integrated_loudness

    rate = 24000
    tone = _sine(-23.0, 0.5, rate)
    reference = integrated_loudness(tone, rate)
    hop = int(0.4 * 0.25 * rate)
    for extra in (1, hop // 2, hop - 1):
        padded = np.concatenate([tone, _sine(-23.0, 0.5, rate)[:extra]])
        assert integrated_loudness(padded, rate) == pytest.approx(reference, abs=1e-12)

    readings = [
        integrated_loudness(_sine(-23.0, duration, rate), rate)
        for duration in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    ]
    assert max(readings) - min(readings) < 0.01


def test_integrated_loudness_gates_quiet_passages():
    """The two-stage -70 LKFS / -10 LU gating keeps a loud programme's reading
    from being dragged down by quiet passages around it."""
    from mlx_audio.dsp import integrated_loudness

    rate = 24000
    loud = _sine(-23.0, 8.0, rate)

    # -60 dBFS sits above the -70 LKFS absolute threshold, so it is the relative
    # threshold that has to exclude it.
    quiet = _sine(-60.0, 2.0, rate)
    assert integrated_loudness(
        np.concatenate([quiet, loud, quiet]), rate
    ) == pytest.approx(integrated_loudness(loud, rate), abs=0.25)

    # Below the absolute threshold, pushing a passage even further down cannot
    # change the result at all.
    readings = [
        integrated_loudness(
            np.concatenate([_sine(peak, 2.0, rate), loud, _sine(peak, 2.0, rate)]),
            rate,
        )
        for peak in (-100.0, -140.0)
    ]
    assert readings[0] == pytest.approx(readings[1], abs=1e-7)


def test_normalize_loudness_reaches_target():
    """Normalizing to a target LUFS has to actually land on that target."""
    from mlx_audio.dsp import integrated_loudness, normalize_loudness

    rng = np.random.default_rng(0)
    mono = (rng.standard_normal(24000) * 0.02).astype(np.float64)

    measured = integrated_loudness(mono, 24000)
    normalized = normalize_loudness(mono, measured, -18.0)

    assert integrated_loudness(normalized, 24000) == pytest.approx(-18.0, abs=1e-9)
    np.testing.assert_allclose(
        normalized, mono * 10.0 ** ((-18.0 - measured) / 20.0), rtol=1e-12
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


def test_resample_rejects_energy_above_target_nyquist():
    """A tone just above the new Nyquist must be band-limited away, not aliased
    back into the signal. The previous Kaiser(5.0) default left a large residual
    in the top bins here (#24)."""
    from mlx_audio.utils import resample_audio

    orig, target = 24000, 16000
    t = np.arange(2 * orig) / orig
    tone = np.sin(2 * np.pi * 8200.0 * t).astype(np.float32)  # 200 Hz above Nyquist
    out = np.asarray(resample_audio(tone, orig, target))
    rms = float(np.sqrt(np.mean(out[400:-400] ** 2)))
    assert rms < 0.01  # sharp filter -> ~0; the old default left ~0.26


def test_resample_preserves_passband():
    """In-band tones pass with ~unit gain, including near the Nyquist edge where
    the old filter drooped (full-scale sine RMS == 1/sqrt(2) ~= 0.7071)."""
    from mlx_audio.utils import resample_audio

    orig, target = 24000, 16000
    t = np.arange(2 * orig) / orig
    for freq in (1000.0, 7000.0):
        tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
        out = np.asarray(resample_audio(tone, orig, target))
        rms = float(np.sqrt(np.mean(out[400:-400] ** 2)))
        assert 0.70 < rms < 0.72


def test_resample_length_and_type():
    """Output length tracks the rate ratio and the return type matches input."""
    import mlx.core as mx

    from mlx_audio.utils import resample_audio

    x = np.zeros(24000, dtype=np.float32)
    out = resample_audio(x, 24000, 16000)
    assert isinstance(out, np.ndarray)
    assert abs(len(out) - 16000) <= 1

    out_mx = resample_audio(mx.array(x), 24000, 16000)
    assert isinstance(out_mx, mx.array)


def test_resample_noop_when_rates_equal():
    from mlx_audio.utils import resample_audio

    x = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    out = resample_audio(x, 16000, 16000)
    np.testing.assert_array_equal(np.asarray(out), x)


@pytest.mark.parametrize("orig", [24000, 44100, 48000])
def test_chunked_resample_matches_whole_buffer(orig):
    from mlx_audio.resample import resample_audio_chunks
    from mlx_audio.utils import resample_audio

    target = 16000
    rng = np.random.default_rng(orig)
    audio = rng.normal(0.0, 0.1, size=(2 * orig + 137, 2)).astype(np.float32)
    chunk_sizes = (137, 997, 4096, 53, 1201)

    def chunks():
        start = 0
        chunk_index = 0
        while start < len(audio):
            end = start + chunk_sizes[chunk_index % len(chunk_sizes)]
            yield audio[start:end]
            start = end
            chunk_index += 1

    expected = resample_audio(audio, orig, target, axis=0)
    actual = resample_audio_chunks(
        chunks(),
        orig,
        target,
        len(audio),
        chunk_duration_seconds=0.025,
    )

    np.testing.assert_array_equal(actual, expected)


def test_integrated_loudness_validation_matches_previous_behavior():
    """Verify the public helper keeps the old validation semantics."""
    from mlx_audio.dsp import integrated_loudness

    with pytest.raises(ValueError, match="Data must be floating point."):
        integrated_loudness(np.arange(10, dtype=np.int16), 24000)

    with pytest.raises(
        ValueError, match="Audio must have length greater than the block size."
    ):
        integrated_loudness(np.zeros(100, dtype=np.float64), 24000)
