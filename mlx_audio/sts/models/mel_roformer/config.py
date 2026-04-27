# Copyright (c) 2026 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Mel-Band-RoFormer configuration.

The architecture itself is licensed under MIT (from lucidrains/BS-RoFormer and
ZFTurbo/Music-Source-Separation-Training). No default constructor is provided —
callers must name their checkpoint family explicitly to avoid accidentally
pulling GPL-3 or undeclared-license weights via a tutorial copy-paste.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MelRoFormerConfig:
    """Configuration for Mel-Band-RoFormer vocal source separation.

    Users must choose an explicit preset matching their checkpoint family,
    or construct directly via `MelRoFormerConfig(...)` with all required
    hyperparameters for a custom checkpoint.

    Architecture: STFT → CaC interleave → BandSplit → N× DualAxisTransformer →
                  MaskEstimate → complex multiply → iSTFT
    """

    # Model architecture
    dim: int = 384
    depth: int = 6  # Number of dual-axis transformer depth levels
    heads: int = 8
    dim_head: int = 64
    num_bands: int = 60
    num_stems: int = 1  # 1 = vocals only
    ff_mult: int = 4
    mlp_expansion_factor: int = 4
    mask_estimator_depth: int = 2

    # STFT parameters
    n_fft: int = 2048
    hop_length: int = 441
    win_length: int = 2048
    sample_rate: int = 44100

    # Processing
    chunk_size: int = 352800  # 8 seconds at 44.1kHz
    num_overlap: int = 2  # 50% overlap for chunked processing

    # Optional metadata (set by presets or from companion config.json)
    checkpoint_family: Optional[str] = None

    @property
    def dim_inner(self) -> int:
        """Inner dimension of attention (heads × dim_head)."""
        return self.heads * self.dim_head  # 512

    @property
    def ff_dim(self) -> int:
        """Feed-forward hidden dimension."""
        return self.dim * self.ff_mult  # 1536

    @property
    def mlp_hidden(self) -> int:
        """MLP hidden dimension in mask estimator."""
        return self.dim * self.mlp_expansion_factor  # 1536

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins from STFT (n_fft/2 + 1)."""
        return self.n_fft // 2 + 1  # 1025

    # ---------- Checkpoint-family presets ----------
    #
    # Each preset is named after its checkpoint family and hard-codes the
    # hyperparameters published in that family's training config.
    #
    # The architecture implementation is MIT-licensed regardless of which
    # preset is selected. Individual checkpoints have their own licenses —
    # see the README license posture table and the `checkpoint_family`
    # docstring for each preset below.

    @classmethod
    def kim_vocal_2(cls) -> "MelRoFormerConfig":
        """Matches KimberleyJSN/melbandroformer (depth=6, 60 bands, 44.1kHz).

        Weight license: GPL-3.0 (author-tagged June 2025). Running inference
        locally is unrestricted; redistribution or derivative-work shipping
        triggers GPL obligations on the wider product.

        Architecture config from:
            configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml
            in ZFTurbo/Music-Source-Separation-Training (MIT-licensed configs).
        """
        return cls(depth=6, checkpoint_family="kim_vocal_2")

    @classmethod
    def viperx_vocals(cls) -> "MelRoFormerConfig":
        """Matches viperx vocals checkpoints (depth=12, 60 bands, 44.1kHz).

        Weight license: undeclared. `TRvlvr/model_repo` ships no LICENSE file,
        so default copyright applies — no explicit redistribution right.

        Architecture config from:
            configs/viperx/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml
            in ZFTurbo/Music-Source-Separation-Training (MIT-licensed configs).
        """
        return cls(depth=12, checkpoint_family="viperx_vocals")

    @classmethod
    def zfturbo_bs_roformer(cls) -> "MelRoFormerConfig":
        """Matches ZFTurbo MSS-Training release-asset Mel-Band-RoFormer checkpoints.

        Weight license: MIT (inherited from the ZFTurbo/MSS-Training repo that
        ships them as GitHub release assets). Free to use and redistribute.

        This is the cleanest-license path for commercial / distributed products.
        Configure depth/dim if your specific ZFTurbo checkpoint differs from
        the default 12-depth architecture.
        """
        return cls(depth=12, checkpoint_family="zfturbo_bs_roformer")

    @classmethod
    def zfturbo_vocals_v1(cls) -> "MelRoFormerConfig":
        """Matches ZFTurbo v1.0.0 release asset ``model_vocals_mel_band_roformer_sdr_8.42.ckpt``.

        Architecture config derived from:
            configs/config_vocals_mel_band_roformer.yaml
            in ZFTurbo/Music-Source-Separation-Training (MIT-licensed configs).

        The shipped checkpoint was trained with ``mask_estimator_depth=1``
        (the yaml-default 2 does not match — confirmed by state-dict shapes).

        Weight license: MIT (inherited from the ZFTurbo MSS-Training release).

        Smaller than the viperx/zfturbo_bs_roformer presets — ~135 MB on disk —
        and uses ``hop_length=512`` rather than ``441``.
        """
        return cls(
            dim=192,
            depth=8,
            hop_length=512,
            mask_estimator_depth=1,
            checkpoint_family="zfturbo_vocals_v1",
        )

    @classmethod
    def custom(
        cls,
        *,
        depth: int,
        num_bands: int = 60,
        dim: int = 384,
        heads: int = 8,
        dim_head: int = 64,
        n_fft: int = 2048,
        hop_length: int = 441,
        sample_rate: int = 44100,
        **kwargs,
    ) -> "MelRoFormerConfig":
        """Escape hatch for community-trained variants with non-standard hyperparameters.

        Use this when your checkpoint's training config doesn't match any of
        the named presets. Pass the exact hyperparameters from that training
        config's `model` section.
        """
        return cls(
            depth=depth,
            num_bands=num_bands,
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate,
            checkpoint_family="custom",
            **kwargs,
        )
