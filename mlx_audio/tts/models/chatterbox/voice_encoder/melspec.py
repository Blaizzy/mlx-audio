# Ported from https://github.com/resemble-ai/chatterbox

import math

import mlx.core as mx

from mlx_audio.utils import mel_filters, stft

from .config import VoiceEncConfig


def melspectrogram(wav: mx.array, hp: VoiceEncConfig, pad: bool = True) -> mx.array:
    """
    Compute mel spectrogram from waveform.

    Args:
        wav: Waveform (T,) or (B, T)
        hp: Voice encoder config
        pad: Whether to pad the STFT

    Returns:
        Mel spectrogram (M, T') or (B, M, T')
    """
    was_1d = len(wav.shape) == 1
    if was_1d:
        wav = mx.expand_dims(wav, 0)  # (1, T)

    # STFT - process each batch item separately since stft expects 1D input
    specs = []
    for i in range(wav.shape[0]):
        spec = stft(
            wav[i],  # 1D input (T,)
            window="hann",
            n_fft=hp.n_fft,
            hop_length=hp.hop_size,
            win_length=hp.win_size,
        )
        specs.append(spec)

    # Stack: each spec is (T', F) -> stack to (B, T', F)
    spec = mx.stack(specs, axis=0)

    # Get magnitudes
    spec_magnitudes = mx.abs(spec)  # (B, T', F)

    # Apply power
    if hp.mel_power != 1.0:
        spec_magnitudes = spec_magnitudes**hp.mel_power

    # Create mel filterbank
    # Use librosa defaults: norm='slaney', mel_scale='slaney' (htk=False)
    filters = mel_filters(
        sample_rate=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        f_min=hp.fmin,
        f_max=hp.fmax,
        norm="slaney",
        mel_scale="slaney",
    )

    # Apply mel filterbank: (B, T', F) @ (F, M) -> (B, T', M)
    mel = spec_magnitudes @ filters.T  # (B, T', M)
    mel = mx.transpose(mel, [0, 2, 1])  # (B, M, T')

    # Convert to dB if needed
    if hp.mel_type == "db":
        mel = 20 * mx.log10(mx.maximum(mel, hp.stft_magnitude_min))

    # Normalize if needed
    if hp.normalized_mels:
        min_level_db = 20 * math.log10(hp.stft_magnitude_min)
        headroom_db = 15
        mel = (mel - min_level_db) / (-min_level_db + headroom_db)

    return mel.squeeze(0) if was_1d else mel  # (M, T') or (B, M, T')
