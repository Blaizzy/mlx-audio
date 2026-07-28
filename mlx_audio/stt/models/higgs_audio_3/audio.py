import mlx.core as mx
import mlx.nn as nn

from mlx_audio.utils import hanning, mel_filters, stft

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160


class AudioFeatureExtractor(nn.Module):
    def __init__(
        self,
        num_mel_bins: int = 128,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, audio: mx.array) -> mx.array:
        audio = audio.reshape(-1).astype(mx.float32)
        window = hanning(self.n_fft)
        freqs = stft(audio, window=window, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitudes = freqs[:-1, :].abs().square()
        filters = mel_filters(
            self.sample_rate,
            self.n_fft,
            self.num_mel_bins,
            norm="slaney",
            mel_scale=None,
        )
        mel_spec = magnitudes @ filters.T
        log_spec = mx.maximum(mel_spec, 1e-10).log10()
        log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.T[None]
