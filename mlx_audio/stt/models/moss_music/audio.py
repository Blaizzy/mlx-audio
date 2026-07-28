from typing import Tuple, Union

import mlx.core as mx
import numpy as np

from mlx_audio.utils import hanning, mel_filters, stft

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160


class MossMusicFeatureExtractor:
    def __init__(
        self,
        num_mel_bins: int = 128,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ):
        self.num_mel_bins = int(num_mel_bins)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.window = hanning(self.n_fft, periodic=True).astype(mx.float32)
        self.filters = mel_filters(
            self.sample_rate,
            self.n_fft,
            self.num_mel_bins,
            norm="slaney",
            mel_scale="slaney",
            precise=True,
        ).astype(mx.float32)
        mx.eval(self.window, self.filters)

    def _normalize_waveform(self, audio: Union[mx.array, np.ndarray]) -> mx.array:
        if isinstance(audio, mx.array):
            wav = audio.astype(mx.float32)
        else:
            wav = mx.array(np.asarray(audio), dtype=mx.float32)

        if wav.ndim == 2:
            if wav.shape[0] <= 8 and wav.shape[1] > wav.shape[0]:
                wav = mx.mean(wav, axis=0)
            else:
                wav = mx.mean(wav, axis=1)
        if wav.ndim != 1:
            raise ValueError(f"Expected mono waveform, got shape {wav.shape}.")
        return wav

    def __call__(self, audio: Union[mx.array, np.ndarray]) -> Tuple[mx.array, int]:
        wav = self._normalize_waveform(audio)
        spec = stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
        )
        power = mx.square(mx.abs(spec)).astype(mx.float32)
        mel = power @ self.filters.T

        if mel.shape[0] > 1:
            mel = mel[:-1]

        log_spec = mx.log10(mx.maximum(mel, 1e-10))
        log_spec = mx.maximum(log_spec, mx.max(log_spec) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.T.astype(mx.float32), int(log_spec.shape[0])
