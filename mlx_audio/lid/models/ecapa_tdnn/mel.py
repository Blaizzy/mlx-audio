"""SpeechBrain-compatible mel spectrogram computed on GPU via MLX.

Uses periodic Hamming window, zero center-padding, HTK mel scale,
``10 * log10`` normalization, and ``top_db = 80`` clipping.
"""

import math

import mlx.core as mx

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 60


def _hamming_window(length: int) -> mx.array:
    return mx.array(
        [0.54 - 0.46 * math.cos(2.0 * math.pi * n / length) for n in range(length)]
    )


def _htk_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> mx.array:
    def hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low_mel = hz_to_mel(0.0)
    high_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = [
        mel_to_hz(low_mel + i * (high_mel - low_mel) / (n_mels + 1))
        for i in range(n_mels + 2)
    ]
    fft_bins = [sample_rate * k / n_fft for k in range(n_fft // 2 + 1)]

    fb = []
    for m in range(n_mels):
        row = [0.0] * (n_fft // 2 + 1)
        f_left, f_center, f_right = mel_points[m], mel_points[m + 1], mel_points[m + 2]
        band_left = f_center - f_left
        band_right = f_right - f_center
        for k, f in enumerate(fft_bins):
            if f_left <= f <= f_center and band_left > 0:
                row[k] = (f - f_left) / band_left
            elif f_center < f <= f_right and band_right > 0:
                row[k] = (f_right - f) / band_right
        fb.extend(row)

    return mx.array(fb).reshape(n_mels, n_fft // 2 + 1).T


def compute_mel_spectrogram(audio: mx.array) -> mx.array:
    """Compute SpeechBrain-compatible log-mel spectrogram.

    Args:
        audio: 1-D ``mx.array`` of raw 16 kHz audio samples.

    Returns:
        ``[1, num_frames, 60]`` log-mel spectrogram.
    """
    window = _hamming_window(WIN_LENGTH)
    mel_fb = _htk_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)

    pad_len = N_FFT // 2
    padded = mx.concatenate([mx.zeros(pad_len), audio, mx.zeros(pad_len)])

    total_len = padded.shape[0]
    num_frames = max(0, (total_len - N_FFT) // HOP_LENGTH + 1)
    if num_frames == 0:
        return mx.zeros((1, 0, N_MELS))

    frames = mx.as_strided(padded, (num_frames, N_FFT), (HOP_LENGTH, 1))
    fft_result = mx.fft.rfft(frames * window, n=N_FFT, axis=-1)

    power_spec = mx.abs(fft_result) ** 2
    mel_spec = power_spec @ mel_fb

    log_mel = 10.0 * mx.log10(mx.maximum(mel_spec, 1e-10))
    clipped = mx.maximum(log_mel, mx.max(log_mel) - 80.0)

    return clipped.reshape(1, num_frames, N_MELS)
