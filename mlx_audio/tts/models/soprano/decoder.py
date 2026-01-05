# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.vocos import VocosBackbone
from mlx_audio.utils import hanning, istft

from ..interpolate import interpolate


class ISTFTHead(nn.Module):

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "center"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, L, C) where C is the hidden dimension

        Returns:
            Reconstructed audio of shape (B, T)
        """
        # Project to STFT coefficients and transpose: (B, L, n_fft+2) -> (B, n_fft+2, L)
        x = self.out(x).swapaxes(1, 2)

        # Split into magnitude and phase
        mag, p = x.split(2, axis=1)
        mag = mx.exp(mag)
        mag = mx.clip(mag, None, 1e2)

        # Construct complex STFT
        S = mag * (mx.cos(p) + 1j * mx.sin(p))

        # Use mlx_audio's tested ISTFT
        audio = istft(
            S.squeeze(0),
            window=hanning(self.n_fft),
            hop_length=self.hop_length,
            win_length=self.n_fft,
        )
        return audio[None, :]  # Add batch dimension back


class SopranoDecoder(nn.Module):

    def __init__(
        self,
        num_input_channels: int = 512,
        decoder_num_layers: int = 8,
        decoder_dim: int = 512,
        decoder_intermediate_dim: Optional[int] = None,
        hop_length: int = 512,
        n_fft: int = 2048,
        upscale: int = 4,
        dw_kernel: int = 3,
    ):
        super().__init__()
        self.decoder_initial_channels = num_input_channels
        self.num_layers = decoder_num_layers
        self.dim = decoder_dim
        self.intermediate_dim = (
            decoder_intermediate_dim if decoder_intermediate_dim else decoder_dim * 3
        )
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.upscale = upscale
        self.dw_kernel = dw_kernel

        self.decoder = VocosBackbone(
            input_channels=self.decoder_initial_channels,
            dim=self.dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            input_kernel_size=dw_kernel,
            dw_kernel_size=dw_kernel,
        )
        self.head = ISTFTHead(
            dim=self.dim,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Hidden states from LLM, shape (B, L, C) - MLX convention

        Returns:
            Reconstructed audio, shape (B, audio_length)
        """
        # x is (B, L, C) in MLX convention
        # For interpolate, we need (B, C, L)
        x = mx.transpose(x, (0, 2, 1))  # (B, C, L)

        L = x.shape[2]
        # Upscale hidden states
        target_size = self.upscale * (L - 1) + 1
        x = interpolate(x, size=target_size, mode="linear", align_corners=True)

        # Convert back to MLX convention (B, L, C)
        x = mx.transpose(x, (0, 2, 1))

        # Decode through backbone
        x = self.decoder(x)

        # Convert to audio via ISTFT
        reconstructed = self.head(x)
        return reconstructed
