import mlx.core as mx

from mlx_audio.utils import mel_filters, stft


def log_mel_spectrogram(
    audio: mx.array,
    sample_rate: int = 24_000,
    n_mels: int = 100,
    n_fft: int = 1024,
    hop_length: int = 256,
    padding: int = 0,
):
    if not isinstance(audio, mx.array):
        audio = mx.array(audio)

    if padding > 0:
        audio = mx.pad(audio, (0, padding))

    freqs = stft(
        audio,
        window="hann",
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
    )
    magnitudes = freqs.abs()
    filters = mel_filters(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        norm=None,
        mel_scale="htk",
    )
    mel_spec = magnitudes @ filters.T
    log_spec = mx.maximum(mel_spec, 1e-5).log()
    return mx.expand_dims(log_spec, axis=0)
