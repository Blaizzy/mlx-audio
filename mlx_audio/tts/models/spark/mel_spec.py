import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def mlx_stft(
    x,
    n_fft=800,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="reflect",
):
    """MLX implementation of Short-Time Fourier Transform"""
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if isinstance(window, str):
        if window.lower() == "hann":
            w = mx.array(np.hanning(win_length + 1)[:-1])
        else:
            raise ValueError(
                f"Only 'hann' (string) is supported for window, not {window!r}"
            )
    else:
        w = window

    if w.shape[0] < n_fft:
        pad_size = n_fft - w.shape[0]
        w = mx.concatenate([w, mx.zeros((pad_size,))], axis=0)

    def _pad(x, padding, pad_mode="reflect"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    x = mx.array(x)

    if center:
        x = _pad(x, n_fft // 2, pad_mode)

    num_frames = 1 + (x.shape[0] - n_fft) // hop_length
    if num_frames <= 0:
        raise ValueError(
            f"Input is too short (length={x.shape[0]}) for n_fft={n_fft} with "
            f"hop_length={hop_length} and center={center}."
        )

    shape = (num_frames, n_fft)
    strides = (hop_length, 1)
    frames = mx.as_strided(x, shape=shape, strides=strides)
    spec = mx.fft.rfft(frames * w)

    return spec.transpose(1, 0)


class MelSpectrogram(nn.Module):
    """MLX implementation of MelSpectrogram transformation"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = None,
        hop_length: int = None,
        f_min: float = 0.0,
        f_max: float = None,
        n_mels: int = 80,
        power: float = 1.0,
        norm: str = "slaney",
        mel_scale: str = "slaney",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2
        self.n_mels = n_mels
        self.power = power
        self.norm = norm
        self.mel_scale = mel_scale

        # Create mel filter bank
        self.mel_fb = self._create_mel_filterbank()

    def _create_mel_filterbank(self):
        """Create a mel filterbank matrix to convert linear frequency spectrogram to mel spectrogram"""

        # Convert Hz to mel - handles both scalar and array inputs
        def hz_to_mel(hz):
            # Convert scalar to numpy array if needed
            is_scalar = np.isscalar(hz)
            hz_array = np.asarray([hz]) if is_scalar else np.asarray(hz)

            if self.mel_scale == "slaney":
                # Slaney formula for mel scale
                f_min = 0.0
                f_sp = 200.0 / 3
                min_log_hz = 1000.0
                min_log_mel = (min_log_hz - f_min) / f_sp
                logstep = math.log(6.4) / 27.0

                mel = np.zeros_like(hz_array)
                linear_mask = hz_array < min_log_hz
                mel[linear_mask] = (hz_array[linear_mask] - f_min) / f_sp
                mel[~linear_mask] = (
                    min_log_mel + np.log(hz_array[~linear_mask] / min_log_hz) / logstep
                )

                return mel[0] if is_scalar else mel
            else:
                # HTK formula for mel scale
                mel = 2595.0 * np.log10(1.0 + hz_array / 700.0)
                return mel[0] if is_scalar else mel

        # Convert mel to Hz - handles both scalar and array inputs
        def mel_to_hz(mel):
            # Convert scalar to numpy array if needed
            is_scalar = np.isscalar(mel)
            mel_array = np.asarray([mel]) if is_scalar else np.asarray(mel)

            if self.mel_scale == "slaney":
                # Slaney formula for mel scale
                f_min = 0.0
                f_sp = 200.0 / 3
                min_log_hz = 1000.0
                min_log_mel = (min_log_hz - f_min) / f_sp
                logstep = math.log(6.4) / 27.0

                hz = np.zeros_like(mel_array)
                linear_mask = mel_array < min_log_mel
                hz[linear_mask] = f_min + f_sp * mel_array[linear_mask]
                hz[~linear_mask] = min_log_hz * np.exp(
                    logstep * (mel_array[~linear_mask] - min_log_mel)
                )

                return hz[0] if is_scalar else hz
            else:
                # HTK formula for mel scale
                hz = 700.0 * (10.0 ** (mel_array / 2595.0) - 1.0)
                return hz[0] if is_scalar else hz

        # Create mel filterbank
        fft_freqs = np.arange(self.n_fft // 2 + 1) * self.sample_rate / self.n_fft
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_edges = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(
            int
        )

        filters = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(self.n_mels):
            start, center, end = bin_edges[i : i + 3]
            # Create triangular filter
            # Ascending part
            if start != center:
                filters[i, start : center + 1] = np.linspace(0, 1, center - start + 1)
            # Descending part
            if center != end:
                filters[i, center : end + 1] = np.linspace(1, 0, end - center + 1)

        # Normalize filterbank if requested
        if self.norm == "slaney":
            # Normalize each filter by area
            enorm = 2.0 / (hz_points[2 : self.n_mels + 2] - hz_points[: self.n_mels])
            filters *= enorm[:, np.newaxis]

        return mx.array(filters, dtype=mx.float32)

    def __call__(self, audio):
        """
        Transform audio waveform to mel spectrogram.

        Args:
            audio (mx.array): Audio waveform with shape [batch_size, samples] or [samples]

        Returns:
            mx.array: Mel spectrogram with shape [batch_size, n_mels, time]
        """
        # Handle 1D input (no batch dimension)
        if audio.ndim == 1:
            audio_1d = audio
            batch_mode = False
        else:
            # Handle batched input - we'll process each item separately
            batch_mode = True
            batch_size = audio.shape[0]
            results = []

            for i in range(batch_size):
                # Process each batch item and collect results
                spec = self.__call__(audio[i])
                results.append(spec)

            # Stack along batch dimension
            return mx.stack(results)

        # For 1D input, process directly
        if not batch_mode:
            # Calculate STFT using mlx_stft
            stft = mlx_stft(
                audio_1d,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window="hann",
                center=True,
            )

            # Calculate magnitude
            magnitudes = mx.abs(stft)

            # Apply power
            if self.power != 1.0:
                magnitudes = mx.power(magnitudes, self.power)

            # Convert to mel scale - filter bank has shape [n_mels, n_freqs]
            mel_spec = mx.matmul(self.mel_fb, magnitudes)

            # # Add batch dimension for consistency
            # mel_spec = mel_spec.reshape(1, self.n_mels, mel_spec.shape[1])

            return mel_spec
