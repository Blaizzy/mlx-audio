"""
Configuration for DeepFilterNet speech enhancement model.

Based on the DeepFilterNet architecture by Hendrik Schröter et al.
https://github.com/Rikorose/DeepFilterNet
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DeepFilterNetConfig:
    """Configuration for DeepFilterNet model.

    DeepFilterNet is a low complexity speech enhancement framework for full-band audio (48kHz)
    using deep filtering.

    Attributes:
        sample_rate: Audio sample rate (48000 Hz)
        fft_size: FFT size in samples
        hop_size: STFT hop size in samples
        nb_erb: Number of ERB (Equivalent Rectangular Bandwidth) bands
        nb_df: Number of deep filtering bins
        df_order: Deep filtering order
        df_lookahead: Deep filtering look-ahead
        conv_ch: Number of convolution channels
        conv_depthwise: Use depthwise convolutions
        emb_hidden_dim: Embedding hidden dimension
        emb_num_layers: Number of embedding GRU layers
        df_hidden_dim: Deep filtering hidden dimension
        df_num_layers: Number of deep filtering GRU layers
        mask_pf: Enable post-filter
        pf_beta: Post-filter beta parameter
    """

    # Audio parameters
    model_version: str = "DeepFilterNet3"
    sample_rate: int = 48000

    # STFT parameters
    fft_size: int = 960
    hop_size: int = 480

    # ERB and Deep Filtering parameters
    nb_erb: int = 32
    erb_widths: Optional[List[int]] = None
    nb_df: int = 96
    df_order: int = 5
    df_lookahead: int = 0
    conv_lookahead: int = 0

    # Model architecture
    conv_ch: int = 16
    conv_k_enc: int = 1
    conv_k_dec: int = 1
    conv_width_factor: int = 1
    conv_dec_mode: str = "transposed"
    conv_depthwise: bool = True
    convt_depthwise: bool = True
    conv_kernel: List[int] = field(default_factory=lambda: [1, 3])
    convt_kernel: List[int] = field(default_factory=lambda: [1, 3])
    conv_kernel_inp: List[int] = field(default_factory=lambda: [3, 3])

    # Embedding dimensions
    emb_hidden_dim: int = 256
    emb_num_layers: int = 2

    # Deep filtering dimensions
    df_hidden_dim: int = 256
    df_num_layers: int = 3
    df_pathway_kernel_size_t: int = (
        5  # Kernel size for DF pathway convolution in time dimension
    )

    # Skip connections
    emb_gru_skip: str = "none"
    emb_gru_skip_enc: str = "none"
    df_gru_skip: str = "none"

    # Linear layer groups
    gru_groups: int = 8
    linear_groups: int = 8  # Must match PyTorch model (was 8 during training)
    enc_linear_groups: int = 16
    group_shuffle: bool = False

    # Post-filter
    mask_pf: bool = False
    pf_beta: float = 0.02

    # Other
    enc_concat: bool = False
    dfop_method: str = "real_unfold"
    lsnr_max: int = 35
    lsnr_min: int = -15
    lsnr_dropout: bool = False

    # Processing
    chunk_seconds: float = 4.0
    chunk_overlap: float = 0.25
    auto_chunk_threshold: float = 60.0

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins."""
        return self.fft_size // 2 + 1

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DeepFilterNetConfig":
        """Create config from dictionary."""
        known_fields = {
            k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__
        }
        return cls(**known_fields)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "model_version": self.model_version,
            "fft_size": self.fft_size,
            "hop_size": self.hop_size,
            "nb_erb": self.nb_erb,
            "erb_widths": self.erb_widths,
            "nb_df": self.nb_df,
            "df_order": self.df_order,
            "df_lookahead": self.df_lookahead,
            "conv_lookahead": self.conv_lookahead,
            "conv_ch": self.conv_ch,
            "conv_k_enc": self.conv_k_enc,
            "conv_k_dec": self.conv_k_dec,
            "conv_width_factor": self.conv_width_factor,
            "conv_dec_mode": self.conv_dec_mode,
            "conv_depthwise": self.conv_depthwise,
            "convt_depthwise": self.convt_depthwise,
            "conv_kernel": self.conv_kernel,
            "convt_kernel": self.convt_kernel,
            "conv_kernel_inp": self.conv_kernel_inp,
            "emb_hidden_dim": self.emb_hidden_dim,
            "emb_num_layers": self.emb_num_layers,
            "df_hidden_dim": self.df_hidden_dim,
            "df_num_layers": self.df_num_layers,
            "emb_gru_skip": self.emb_gru_skip,
            "emb_gru_skip_enc": self.emb_gru_skip_enc,
            "df_gru_skip": self.df_gru_skip,
            "gru_groups": self.gru_groups,
            "linear_groups": self.linear_groups,
            "enc_linear_groups": self.enc_linear_groups,
            "group_shuffle": self.group_shuffle,
            "mask_pf": self.mask_pf,
            "pf_beta": self.pf_beta,
            "enc_concat": self.enc_concat,
            "dfop_method": self.dfop_method,
            "lsnr_max": self.lsnr_max,
            "lsnr_min": self.lsnr_min,
            "lsnr_dropout": self.lsnr_dropout,
            "chunk_seconds": self.chunk_seconds,
            "chunk_overlap": self.chunk_overlap,
            "auto_chunk_threshold": self.auto_chunk_threshold,
        }

    # Alias for compatibility
    @property
    def sr(self) -> int:
        return self.sample_rate


@dataclass
class DeepFilterNet2Config(DeepFilterNetConfig):
    """Configuration for DeepFilterNet2 model.

    DeepFilterNet2 is optimized for embedded devices while maintaining quality.
    """

    model_version: str = "DeepFilterNet2"
    conv_ch: int = 16
    emb_hidden_dim: int = 256
    df_hidden_dim: int = 256
    emb_num_layers: int = 2
    df_num_layers: int = 3


@dataclass
class DeepFilterNet3Config(DeepFilterNetConfig):
    """Configuration for DeepFilterNet3 model.

    DeepFilterNet3 adds perceptually motivated improvements.
    """

    model_version: str = "DeepFilterNet3"
    conv_ch: int = 16
    emb_hidden_dim: int = 256
    df_hidden_dim: int = 256
    emb_num_layers: int = 2
    df_num_layers: int = 3
