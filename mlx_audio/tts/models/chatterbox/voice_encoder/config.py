from dataclasses import dataclass


@dataclass
class VoiceEncConfig:
    """Voice encoder configuration."""

    num_mels: int = 40
    sample_rate: int = 16000
    speaker_embed_size: int = 256
    ve_hidden_size: int = 256
    n_fft: int = 400
    hop_size: int = 160
    win_size: int = 400
    fmax: int = 8000
    fmin: int = 0
    preemphasis: float = 0.0
    mel_power: float = 2.0
    mel_type: str = "amp"
    normalized_mels: bool = False
    ve_partial_frames: int = 160
    ve_final_relu: bool = True
    stft_magnitude_min: float = 1e-4
