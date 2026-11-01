"""Log-mel front end adapted from the mlx-vlm LFM3-Audio port."""

import numpy as np

SAMPLE_RATE, N_FFT, HOP, NUM_MEL_BINS, LOG_FLOOR = 16000, 400, 160, 128, -12.0


def num_frames(num_samples: int) -> int:
    """Feature frames of a clip: it is right-padded to 400 + k * 160 samples."""
    return 1 + max(0, -(-(num_samples - N_FFT) // HOP))


def num_encoder_tokens(num_frames: int) -> int:
    return -(num_frames // -8)


_MEL_FILTERS = None


def log_mel(audio) -> np.ndarray:
    """One mono 16 kHz clip (N,) -> float32 features (128, num_frames(N))."""
    global _MEL_FILTERS
    if _MEL_FILTERS is None:
        from mlx_audio.dsp import mel_filters

        _MEL_FILTERS = np.array(
            mel_filters(
                SAMPLE_RATE,
                N_FFT,
                NUM_MEL_BINS,
                norm="slaney",
                mel_scale="slaney",
                precise=True,
            ),
            dtype=np.float64,
        )
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected one mono clip, got shape {x.shape}")
    T = num_frames(len(x))
    x = np.pad(x.astype(np.float64), (0, N_FFT + (T - 1) * HOP - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    with np.errstate(divide="ignore"):
        mel = np.maximum(np.log10(_MEL_FILTERS @ power.T), LOG_FLOOR)
    return ((mel + 4) / 4).astype(np.float32)


def extract_features(clips):
    """Mono 16 kHz clips -> (input_features (B, 128, T) zero-padded to the longest
    clip, feature_attention_mask (B, T)), both numpy, as the reference returns."""
    features = [log_mel(clip) for clip in clips]
    lengths = [f.shape[-1] for f in features]
    out = np.zeros((len(features), NUM_MEL_BINS, max(lengths)), np.float32)
    mask = np.zeros((len(features), max(lengths)), np.int32)
    for i, f in enumerate(features):
        out[i, :, : lengths[i]] = f
        mask[i, : lengths[i]] = 1
    return out, mask
