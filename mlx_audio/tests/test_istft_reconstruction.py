"""Inverse STFT geometry and waveform alignment regressions."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.dsp import ISTFTCache, hanning, istft, stft


def reference_istft(spectrum, hop, window, center, length, normalized=True):
    n_fft = 2 * (spectrum.shape[0] - 1)
    padding = n_fft - len(window)
    window = np.pad(window, (padding // 2, padding - padding // 2))
    frames = np.fft.irfft(spectrum, n=n_fft, axis=0).T
    output = np.zeros((len(frames) - 1) * hop + n_fft)
    envelope = np.zeros_like(output)
    for index, frame in enumerate(frames):
        start = index * hop
        output[start : start + n_fft] += frame * window
        envelope[start : start + n_fft] += window**2 if normalized else window
    np.divide(output, envelope, out=output, where=envelope > 1e-10)
    start = n_fft // 2 if center else 0
    end = start + length if length is not None else len(output) - start
    output = output[start:end]
    if length is not None and len(output) < length:
        output = np.pad(output, (0, length - len(output)))
    return output


def test_stft_istft_reconstructs_with_short_window():
    window = hanning(16, periodic=True)
    waveform = mx.ones(256)

    spectrum = stft(waveform, n_fft=32, hop_length=4, win_length=16, window=window).T
    reconstructed = istft(
        spectrum,
        hop_length=4,
        win_length=32,
        window=window,
        normalized=True,
        length=waveform.shape[0],
    )

    np.testing.assert_allclose(
        np.array(reconstructed), np.array(waveform), atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("win_length", [16, 17])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("constrain_value_range", [False, True])
def test_stft_istft_cache_reconstructs_with_short_window(
    win_length, center, constrain_value_range
):
    n_fft, hop_length = 32, 4
    window = hanning(win_length, periodic=True)
    rng = np.random.default_rng(958)
    waveforms = mx.stack(
        [mx.ones(256), mx.array(rng.uniform(-0.5, 0.5, 256).astype(np.float32))]
    )
    # Non-centered analysis needs context to cover the signal's endpoints with
    # the short window. Only compare samples covered by the analysis windows.
    padding = 0 if center else n_fft // 2
    inputs = mx.pad(waveforms, [(0, 0), (padding, padding)])
    spectra = mx.stack(
        [
            stft(
                waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
            ).T
            for waveform in inputs
        ]
    )

    cache = ISTFTCache()
    # Reuse the cached geometry for a second, different batch of spectra.
    for scale in (1.0, 0.5):
        reconstructed = cache.istft(
            spectra.real * scale,
            spectra.imag * scale,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            audio_length=inputs.shape[-1],
            constrain_value_range=constrain_value_range,
        )
        reconstructed = reconstructed[:, padding : padding + waveforms.shape[-1]]
        np.testing.assert_allclose(
            np.array(reconstructed), np.array(waveforms * scale), atol=2e-5, rtol=2e-5
        )


@pytest.mark.parametrize("n_fft", [16, 64, 256])
@pytest.mark.parametrize("num_frames", [5, 17])
@pytest.mark.parametrize("center", [False, True])
def test_istft_default_window_uses_frequency_axis(n_fft, num_frames, center):
    rng = np.random.default_rng(4)
    spectrum = (
        rng.normal(size=(n_fft // 2 + 1, num_frames))
        + 1j * rng.normal(size=(n_fft // 2 + 1, num_frames))
    ).astype(np.complex64)
    window = np.hanning(n_fft + 1)[:-1]
    expected = reference_istft(spectrum, n_fft // 4, window, center, None)
    actual = istft(mx.array(spectrum), center=center, normalized=True)
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("length", [31, 64, 105])
@pytest.mark.parametrize("win_length", [16, 24, 32])
@pytest.mark.parametrize("normalized", [False, True])
def test_istft_window_and_requested_length(center, length, win_length, normalized):
    n_fft, hop, num_frames = 32, 4, 17
    rng = np.random.default_rng(0)
    spectrum = (
        rng.normal(size=(17, num_frames)) + 1j * rng.normal(size=(17, num_frames))
    ).astype(np.complex64)
    window = np.hamming(win_length).astype(np.float32)
    actual = istft(
        mx.array(spectrum),
        hop_length=hop,
        win_length=win_length,
        window=mx.array(window),
        center=center,
        length=length,
        normalized=normalized,
    )
    expected = reference_istft(spectrum, hop, window, center, length, normalized)
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("window_name", ["hann", "hamming"])
def test_istft_gradients_are_finite_and_match_finite_differences(center, window_name):
    rng = np.random.default_rng(3)
    real = rng.normal(size=(9, 5)).astype(np.float32)
    imag = rng.normal(size=(9, 5)).astype(np.float32)

    def loss(r, i):
        wave = istft(
            r + 1j * i,
            hop_length=4,
            win_length=16,
            window=window_name,
            center=center,
            normalized=True,
        )
        return mx.sum(wave**2)

    grads = mx.grad(loss, argnums=(0, 1))(mx.array(real), mx.array(imag))
    for source, grad in zip([real, imag], grads):
        assert np.isfinite(np.array(grad)).all()
        # Check interior/DC/Nyquist entries without relying on another autodiff.
        for index in [(0, 0), (4, 2), (8, 4)]:
            step = 0.001
            source[index] += step
            upper = float(loss(mx.array(real), mx.array(imag)))
            source[index] -= 2 * step
            lower = float(loss(mx.array(real), mx.array(imag)))
            source[index] += step
            np.testing.assert_allclose(
                float(grad[index]), (upper - lower) / (2 * step), atol=0.03, rtol=0.01
            )
