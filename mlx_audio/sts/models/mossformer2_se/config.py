"""
Configuration for MossFormer2 SE 48K speech enhancement model.
"""

from dataclasses import dataclass


@dataclass
class MossFormer2SEConfig:
    """Configuration for MossFormer2 SE 48K model.

    Attributes:
        sample_rate: Audio sample rate (48000 Hz)
        win_len: STFT window length
        win_inc: STFT hop length
        fft_len: FFT size
        num_mels: Number of mel filterbank channels
        win_type: Window type for STFT ('hamming' or 'hann')
        preemphasis: Pre-emphasis coefficient
        one_time_decode_length: Max audio length (seconds) for full processing
        decode_window: Chunk size (seconds) for segmented processing
        chunk_seconds: Chunk duration for chunked mode
        chunk_overlap: Overlap ratio for chunked mode
        auto_chunk_threshold: Auto-enable chunked mode above this duration (seconds)
    """

    # Audio parameters
    sample_rate: int = 48000

    # STFT parameters
    win_len: int = 1920
    win_inc: int = 384
    fft_len: int = 1920
    win_type: str = "hamming"

    # Feature extraction
    num_mels: int = 60
    preemphasis: float = 0.97

    # Processing mode
    one_time_decode_length: int = 20
    decode_window: int = 4

    # Chunked processing
    chunk_seconds: float = 4.0
    chunk_overlap: float = 0.25
    auto_chunk_threshold: float = 60.0  # Auto-enable chunked mode above 60s

    # Model architecture
    in_channels: int = 180  # 3 * num_mels (fbank + delta + delta-delta)
    out_channels: int = 512
    out_channels_final: int = 961
    num_blocks: int = 24

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MossFormer2SEConfig":
        """Create config from dictionary."""
        return cls(
            **{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "win_len": self.win_len,
            "win_inc": self.win_inc,
            "fft_len": self.fft_len,
            "num_mels": self.num_mels,
            "win_type": self.win_type,
            "preemphasis": self.preemphasis,
            "one_time_decode_length": self.one_time_decode_length,
            "decode_window": self.decode_window,
            "chunk_seconds": self.chunk_seconds,
            "chunk_overlap": self.chunk_overlap,
            "auto_chunk_threshold": self.auto_chunk_threshold,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "out_channels_final": self.out_channels_final,
            "num_blocks": self.num_blocks,
        }

    # Alias for compatibility
    @property
    def sampling_rate(self) -> int:
        return self.sample_rate
