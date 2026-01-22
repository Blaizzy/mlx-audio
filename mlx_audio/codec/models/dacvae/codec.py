# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download


@dataclass
class DACVAEConfig:
    """Configuration for the DACVAE audio codec."""

    encoder_dim: int = 64
    encoder_rates: List[int] = field(default_factory=lambda: [2, 8, 10, 12])
    latent_dim: int = 1024
    decoder_dim: int = 1536
    decoder_rates: List[int] = field(default_factory=lambda: [12, 10, 8, 2])
    n_codebooks: int = 16
    codebook_size: int = 1024
    codebook_dim: int = 128
    quantizer_dropout: bool = False
    sample_rate: int = 48_000
    mean: float = 0.0
    std: float = 1.0

    @property
    def hop_length(self) -> int:
        return int(np.prod(self.encoder_rates))


def normalize_weight(x: mx.array, except_dim: int = 0) -> mx.array:
    """Compute weight normalization factor."""
    if x.ndim != 3:
        raise ValueError("Input tensor must have 3 dimensions")
    axes = tuple(i for i in range(x.ndim) if i != except_dim)
    return mx.sqrt(mx.sum(x * x, axis=axes, keepdims=True))


# =============================================================================
# Basic Layers
# =============================================================================


def snake(x: mx.array, alpha: mx.array) -> mx.array:
    """Snake activation function."""
    recip = 1.0 / (alpha + 1e-9)
    return x + recip * mx.power(mx.sin(alpha * x), 2)


class Snake1d(nn.Module):
    """Snake activation for 1D signals."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, 1, channels))

    def __call__(self, x: mx.array) -> mx.array:
        return snake(x, self.alpha)


class WNConv1d(nn.Module):
    """Weight-normalized 1D convolution with optional causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = False,
        pad_mode: str = "none",
        norm: str = "weight_norm",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.causal = causal
        self.pad_mode = pad_mode
        self.use_weight_norm = norm == "weight_norm"

        # Calculate padding for pad_mode="none"
        if pad_mode == "none":
            self.padding = (kernel_size - stride) * dilation // 2
        else:
            self.padding = 0

        if self.use_weight_norm:
            scale = math.sqrt(1 / (in_channels * kernel_size))
            weight_init = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(out_channels, kernel_size, in_channels),
            )
            self.weight_g = normalize_weight(weight_init)
            self.weight_v = weight_init / (self.weight_g + 1e-12)
        else:
            scale = math.sqrt(1 / (in_channels * kernel_size))
            self.weight = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(out_channels, kernel_size, in_channels),
            )

        self.bias = mx.zeros((out_channels,)) if bias else None

    def _get_weight(self):
        if self.use_weight_norm:
            return self.weight_g * self.weight_v / normalize_weight(self.weight_v)
        return self.weight

    def _auto_pad(self, x: mx.array) -> mx.array:
        """Apply automatic padding for causal/non-causal convolutions."""
        if self.pad_mode == "none":
            return x

        length = x.shape[1]
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        padding_total = effective_kernel_size - self.stride
        n_frames = (length - effective_kernel_size + padding_total) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (
            self.kernel_size - padding_total
        )
        extra_padding = max(0, ideal_length - length)

        if self.causal:
            # Causal: all padding on left
            pad_left = padding_total
            pad_right = extra_padding
        else:
            # Non-causal: symmetric padding
            pad_right = extra_padding // 2
            pad_left = padding_total - pad_right + extra_padding - pad_right

        if pad_left > 0 or pad_right > 0:
            x = mx.pad(x, [(0, 0), (pad_left, pad_right), (0, 0)])

        return x

    def __call__(self, x: mx.array) -> mx.array:
        x = self._auto_pad(x)
        weight = self._get_weight()
        y = mx.conv1d(x, weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            y = y + self.bias
        return y


class WNConvTranspose1d(nn.Module):
    """Weight-normalized transposed 1D convolution with optional causal unpadding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = False,
        pad_mode: str = "none",
        norm: str = "weight_norm",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.causal = causal
        self.pad_mode = pad_mode
        self.use_weight_norm = norm == "weight_norm"

        # Calculate padding for pad_mode="none"
        if pad_mode == "none":
            self.padding = (stride + 1) // 2
        else:
            self.padding = 0

        if self.use_weight_norm:
            scale = math.sqrt(1 / (in_channels * kernel_size))
            weight_init = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(out_channels, kernel_size, in_channels),
            )
            self.weight_g = normalize_weight(weight_init, except_dim=2)
            self.weight_v = weight_init / (self.weight_g + 1e-12)
        else:
            scale = math.sqrt(1 / (in_channels * kernel_size))
            self.weight = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(out_channels, kernel_size, in_channels),
            )

        self.bias = mx.zeros((out_channels,)) if bias else None

    def _get_weight(self):
        if self.use_weight_norm:
            return (
                self.weight_g
                * self.weight_v
                / normalize_weight(self.weight_v, except_dim=2)
            )
        return self.weight

    def _unpad(self, x: mx.array) -> mx.array:
        """Remove padding from output for causal/non-causal convolutions."""
        if self.pad_mode == "none":
            return x

        length = x.shape[1]
        padding_total = self.kernel_size - self.stride

        if self.causal:
            # Causal: remove padding from end
            end = length - padding_total
            x = x[:, :end, :]
        else:
            # Non-causal: remove from both sides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            end = length - padding_right
            x = x[:, padding_left:end, :]

        return x

    def __call__(self, x: mx.array) -> mx.array:
        weight = self._get_weight()
        y = mx.conv_transpose1d(x, weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            y = y + self.bias
        return self._unpad(y)


# =============================================================================
# Residual Units
# =============================================================================


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions supporting Snake or ELU activation."""

    def __init__(
        self,
        dim: int = 16,
        kernel: int = 7,
        dilation: int = 1,
        act: str = "Snake",
        compress: int = 1,
        causal: bool = False,
        pad_mode: str = "none",
        norm: str = "weight_norm",
        true_skip: bool = False,
    ):
        super().__init__()
        self.true_skip = true_skip
        self.act_type = act

        hidden = dim // compress

        # First activation + conv
        if act == "Snake":
            self.act1 = Snake1d(dim)
        else:
            self.act1 = nn.ELU()

        self.conv1 = WNConv1d(
            dim,
            hidden,
            kernel_size=kernel,
            dilation=dilation,
            causal=causal,
            pad_mode=pad_mode,
            norm=norm,
        )

        # Second activation + conv
        if act == "Snake":
            self.act2 = Snake1d(hidden)
        else:
            self.act2 = nn.ELU()

        self.conv2 = WNConv1d(
            hidden,
            dim,
            kernel_size=1,
            causal=causal,
            pad_mode=pad_mode,
            norm=norm,
        )

    def __call__(self, x: mx.array) -> mx.array:
        y = self.act1(x)
        y = self.conv1(y)
        y = self.act2(y)
        y = self.conv2(y)

        if self.true_skip:
            return x

        # Handle padding differences
        pad = (x.shape[1] - y.shape[1]) // 2
        if pad > 0:
            x = x[:, pad:-pad, :]
        return x + y


# =============================================================================
# Encoder
# =============================================================================


class EncoderBlock(nn.Module):
    """Encoder block with residual units and downsampling."""

    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.res1 = ResidualUnit(dim // 2, dilation=1)
        self.res2 = ResidualUnit(dim // 2, dilation=3)
        self.res3 = ResidualUnit(dim // 2, dilation=9)
        self.snake = Snake1d(dim // 2)
        self.conv = WNConv1d(
            dim // 2,
            dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.snake(x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """DACVAE Encoder."""

    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        self.conv_in = WNConv1d(1, d_model, kernel_size=7, padding=3)

        self.blocks = []
        current_dim = d_model
        for stride in strides:
            current_dim *= 2
            self.blocks.append(EncoderBlock(current_dim, stride=stride))

        self.snake_out = Snake1d(current_dim)
        self.conv_out = WNConv1d(current_dim, d_latent, kernel_size=3, padding=1)
        self.enc_dim = current_dim

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.snake_out(x)
        x = self.conv_out(x)
        return x


# =============================================================================
# LSTM Block for Watermarking
# =============================================================================


class StackedLSTM(nn.Module):
    """Multi-layer LSTM that matches PyTorch's nn.LSTM(num_layers=N)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create stacked LSTM layers
        self.layers = [
            nn.LSTM(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ]

    def __call__(self, x: mx.array, hidden=None):
        """
        Forward pass through all LSTM layers.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional tuple of (h_0, c_0), each of shape (num_layers, batch, hidden_size)

        Returns:
            output: Output tensor of shape (batch, seq_len, hidden_size)
            (h_n, c_n): Final hidden states
        """
        if hidden is None:
            h_list = [None] * self.num_layers
            c_list = [None] * self.num_layers
        else:
            h_0, c_0 = hidden
            h_list = [h_0[i] for i in range(self.num_layers)]
            c_list = [c_0[i] for i in range(self.num_layers)]

        output = x
        new_h = []
        new_c = []

        for i, layer in enumerate(self.layers):
            all_h, all_c = layer(output, hidden=h_list[i], cell=c_list[i])
            output = all_h
            # Keep final timestep for hidden state
            new_h.append(all_h[:, -1, :] if all_h.ndim == 3 else all_h)
            new_c.append(all_c[:, -1, :] if all_c.ndim == 3 else all_c)

        h_n = mx.stack(new_h, axis=0) if new_h else None
        c_n = mx.stack(new_c, axis=0) if new_c else None

        return output, (h_n, c_n)


class LSTMBlock(nn.Module):
    """LSTM block with optional skip connection."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, skip: bool = True
    ):
        super().__init__()
        self.skip = skip
        self.lstm = StackedLSTM(input_size, hidden_size, num_layers)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C) - already in the right format for MLX LSTM
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        return y


# =============================================================================
# Decoder Block (Complex with watermarking support)
# =============================================================================


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, residual units, and watermarking paths."""

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        stride_wm: int = 1,
        downsampling_factor: int = 3,
    ):
        super().__init__()
        self.stride = stride
        self.stride_wm = stride_wm
        self.downsampling_factor = downsampling_factor

        # Block 0: Snake activation
        self.block_0 = Snake1d(input_dim)

        # Block 1: Main upsample ConvTranspose (Snake path)
        self.block_1 = WNConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            causal=False,
            pad_mode="none",
            norm="weight_norm",
        )

        # Block 2: ELU activation (for watermark path)
        self.block_2 = nn.ELU()

        # Block 3: Watermark upsample ConvTranspose (ELU path)
        wm_in = input_dim // downsampling_factor
        wm_out = output_dim // downsampling_factor
        self.block_3 = WNConvTranspose1d(
            wm_in,
            wm_out,
            kernel_size=2 * stride_wm,
            stride=stride_wm,
            causal=True,
            pad_mode="auto",
            norm="none",
        )

        # Block 4: ResidualUnit (Snake, dilation=1)
        self.block_4 = ResidualUnit(
            output_dim,
            dilation=1,
            act="Snake",
            compress=1,
            causal=False,
            pad_mode="none",
            norm="weight_norm",
            true_skip=False,
        )

        # Block 5: ResidualUnit (Snake, dilation=3)
        self.block_5 = ResidualUnit(
            output_dim,
            dilation=3,
            act="Snake",
            compress=1,
            causal=False,
            pad_mode="none",
            norm="weight_norm",
            true_skip=False,
        )

        # Block 6: ResidualUnit (ELU, causal, kernel=3)
        self.block_6 = ResidualUnit(
            output_dim // downsampling_factor,
            kernel=3,
            dilation=1,
            act="ELU",
            compress=2,
            causal=True,
            pad_mode="auto",
            norm="none",
            true_skip=True,
        )

        # Block 7: ResidualUnit (ELU, causal, kernel=3)
        self.block_7 = ResidualUnit(
            output_dim // downsampling_factor,
            kernel=3,
            dilation=1,
            act="ELU",
            compress=2,
            causal=True,
            pad_mode="auto",
            norm="none",
            true_skip=True,
        )

        # Block 8: ResidualUnit (Snake, dilation=9)
        self.block_8 = ResidualUnit(
            output_dim,
            dilation=9,
            act="Snake",
            compress=1,
            causal=False,
            pad_mode="none",
            norm="weight_norm",
            true_skip=False,
        )

        # Block 9: Identity (placeholder for optional last ResidualUnit)
        # In the default case, this is nn.Identity()

        # Block 10: ELU activation
        self.block_10 = nn.ELU()

        # Block 11: Downsample Conv for watermark path
        self.block_11 = WNConv1d(
            wm_out,
            wm_in,
            kernel_size=2 * stride_wm,
            stride=stride_wm,
            causal=True,
            pad_mode="auto",
            norm="none",
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Main forward pass (only uses main path blocks)."""
        # Main decoder path: blocks 0, 1, 4, 5, 8
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_8(x)
        return x

    def upsample_group(self, x: mx.array) -> mx.array:
        """Watermark upsample path: blocks 2, 3, 6, 7."""
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_6(x)
        x = self.block_7(x)
        return x

    def downsample_group(self, x: mx.array) -> mx.array:
        """Watermark downsample path: blocks 10, 11."""
        x = self.block_10(x)
        x = self.block_11(x)
        return x


# =============================================================================
# Watermarking Components
# =============================================================================


class MsgProcessor(nn.Module):
    """Apply the secret message to the encoder output."""

    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_processor = nn.Embedding(2 * nbits, hidden_size)

    def __call__(self, hidden: mx.array, msg: mx.array) -> mx.array:
        """
        Args:
            hidden: (B, C, T) - encoder output
            msg: (B, nbits) - binary message
        Returns:
            hidden + msg_embedding
        """
        # Create indices: 0, 2, 4, ..., 2*nbits
        batch_size = msg.shape[0]
        indices = mx.arange(0, 2 * self.nbits, 2)  # (nbits,)
        indices = mx.broadcast_to(indices, (batch_size, self.nbits))  # (B, nbits)
        indices = (indices + msg.astype(mx.int32)).astype(mx.int32)  # (B, nbits)

        msg_aux = self.msg_processor(indices)  # (B, nbits, hidden_size)
        msg_aux = mx.sum(msg_aux, axis=1)  # (B, hidden_size)
        msg_aux = mx.expand_dims(msg_aux, axis=2)  # (B, hidden_size, 1)
        msg_aux = mx.broadcast_to(msg_aux, hidden.shape)  # (B, C, T)

        return hidden + msg_aux


class WatermarkEncoderBlock(nn.Module):
    """Watermark encoder block with Tanh and LSTM.

    Note: pre_0 (Snake) and pre_1 (Conv) are shared with decoder's snake_out and conv_out.
    They are passed in via set_shared_layers() after construction.
    """

    def __init__(
        self,
        out_dim: int = 128,
        wm_channels: int = 32,
        hidden: int = 512,
        lstm_layers: int = 2,
    ):
        super().__init__()

        # Shared with decoder - set after construction
        self._snake_out = None  # Will be decoder.snake_out
        self._conv_out = None  # Will be decoder.conv_out

        # Pre-processing after shared layers: Tanh + Conv
        self.pre_3 = WNConv1d(
            1, wm_channels, kernel_size=7, causal=True, pad_mode="auto", norm="none"
        )

        # Post-processing: LSTM + ELU + Conv
        self.post_0 = LSTMBlock(hidden, hidden, lstm_layers, skip=True)
        self.post_1 = nn.ELU()
        self.post_2 = WNConv1d(
            hidden, out_dim, kernel_size=7, causal=True, pad_mode="auto", norm="none"
        )

    def set_shared_layers(self, snake_out: Snake1d, conv_out: WNConv1d):
        """Set shared layers from decoder."""
        self._snake_out = snake_out
        self._conv_out = conv_out

    def __call__(self, x: mx.array) -> mx.array:
        """Forward through pre-processing (shared + own layers)."""
        x = self._snake_out(x)
        x = self._conv_out(x)
        x = mx.tanh(x)
        x = self.pre_3(x)
        return x

    def forward_no_wm_conv(self, x: mx.array) -> mx.array:
        """Forward through shared layers + tanh only (for blending)."""
        x = self._snake_out(x)
        x = self._conv_out(x)
        x = mx.tanh(x)
        return x

    def post_process(self, x: mx.array) -> mx.array:
        """Forward through post-processing."""
        x = self.post_0(x)
        x = self.post_1(x)
        x = self.post_2(x)
        return x


class WatermarkDecoderBlock(nn.Module):
    """Watermark decoder block with LSTM."""

    def __init__(
        self,
        in_dim: int = 128,
        out_dim: int = 1,
        channels: int = 32,
        hidden: int = 512,
        lstm_layers: int = 2,
    ):
        super().__init__()

        # Pre-processing: Conv + LSTM
        self.pre_0 = WNConv1d(
            in_dim, hidden, kernel_size=7, causal=True, pad_mode="auto", norm="none"
        )
        self.pre_1 = LSTMBlock(hidden, hidden, lstm_layers, skip=True)

        # Post-processing: ELU + Conv
        self.post_0 = nn.ELU()
        self.post_1 = WNConv1d(
            channels, out_dim, kernel_size=7, causal=True, pad_mode="auto", norm="none"
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward through pre-processing."""
        x = self.pre_0(x)
        x = self.pre_1(x)
        return x

    def post_process(self, x: mx.array) -> mx.array:
        """Forward through post-processing."""
        x = self.post_0(x)
        x = self.post_1(x)
        return x


class Watermarker(nn.Module):
    """Watermarking module combining encoder and decoder."""

    def __init__(
        self,
        d_out: int = 1,
        d_latent: int = 128,
        channels: int = 32,
        hidden: int = 512,
        nbits: int = 16,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.nbits = nbits

        self.encoder_block = WatermarkEncoderBlock(
            d_latent, channels, hidden, lstm_layers
        )
        self.msg_processor = MsgProcessor(nbits, d_latent)
        self.decoder_block = WatermarkDecoderBlock(
            d_latent, d_out, channels, hidden, lstm_layers
        )

    def set_shared_layers(self, snake_out: Snake1d, conv_out: WNConv1d):
        """Set shared layers from decoder."""
        self.encoder_block.set_shared_layers(snake_out, conv_out)

    def random_message(self, batch_size: int) -> mx.array:
        """Generate random binary message."""
        return mx.random.randint(0, 2, (batch_size, self.nbits))


# =============================================================================
# Full Decoder with Watermarking
# =============================================================================


class Decoder(nn.Module):
    """DACVAE Decoder with watermarking support."""

    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: List[int],
        wm_rates: Optional[List[int]] = None,
        wm_channels: int = 32,
        nbits: int = 16,
        d_out: int = 1,
        d_wm_out: int = 128,
    ):
        super().__init__()

        if wm_rates is None:
            wm_rates = [8, 5, 4, 2]

        # First conv layer
        self.conv_in = WNConv1d(input_channel, channels, kernel_size=7, padding=3)

        # Decoder blocks
        self.blocks = []
        for i, (stride, wm_stride) in enumerate(zip(rates, wm_rates)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            self.blocks.append(DecoderBlock(input_dim, output_dim, stride, wm_stride))

        # Final output layers (shared with watermark encoder)
        final_dim = channels // 2 ** len(rates)
        self.snake_out = Snake1d(final_dim)
        self.conv_out = WNConv1d(final_dim, d_out, kernel_size=7, padding=3)

        # Watermarking (uses snake_out/conv_out as shared layers)
        self.wm_model = Watermarker(
            d_out=d_out,
            d_latent=d_wm_out,
            channels=wm_channels,
            hidden=512,
            nbits=nbits,
            lstm_layers=2,
        )
        self.wm_model.set_shared_layers(self.snake_out, self.conv_out)
        self.alpha = wm_channels / d_wm_out

    def __call__(self, x: mx.array, message: Optional[mx.array] = None) -> mx.array:
        """
        Decode latent features to audio (without final output layers).

        Args:
            x: (B, T, C) latent features
            message: Optional (B, nbits) watermark message

        Returns:
            Decoded features before snake_out/conv_out
        """
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        return x

    def decode_with_watermark(
        self, x: mx.array, message: Optional[mx.array] = None
    ) -> mx.array:
        """
        Decode with optional watermarking.

        Args:
            x: (B, T, C) features from decoder blocks
            message: Optional watermark message

        Returns:
            Final audio output with tanh activation
        """
        if message is not None and self.alpha > 0.0:
            return self._watermark(x, message)
        else:
            # Standard path: snake -> conv -> tanh
            x = self.snake_out(x)
            x = mx.tanh(self.conv_out(x))
            return x

    def _watermark(self, x: mx.array, message: mx.array) -> mx.array:
        """Apply watermarking to the decoder output."""
        # Watermark encoder: snake_out -> conv_out -> tanh -> wm_conv
        h = self.wm_model.encoder_block(x)

        # Upsample through decoder blocks (watermark path)
        for block in reversed(self.blocks):
            h = block.upsample_group(h)

        # Post-process: LSTM -> ELU -> conv
        h = self.wm_model.encoder_block.post_process(h)

        # Apply message embedding
        # Transpose h to (B, C, T) for msg_processor
        h_t = mx.transpose(h, (0, 2, 1))
        h_t = self.wm_model.msg_processor(h_t, message)
        h = mx.transpose(h_t, (0, 2, 1))

        # Watermark decoder: conv -> LSTM
        h = self.wm_model.decoder_block(h)

        # Downsample through decoder blocks (watermark path)
        for block in self.blocks:
            h = block.downsample_group(h)

        # Post-process: ELU -> conv
        h = self.wm_model.decoder_block.post_process(h)

        # Blend: snake_out(x) -> conv_out -> tanh + alpha * watermark
        x_base = self.wm_model.encoder_block.forward_no_wm_conv(x)
        result = x_base + self.alpha * h

        return result


# =============================================================================
# Quantizer Projections
# =============================================================================


class QuantizerInProj(nn.Module):
    """Quantizer input projection (VAE-style with mean/logvar).

    Uses weight normalization like NormConv1d with kernel_size=1.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Projects to 2*out_dim for mean and logvar
        out_channels = out_dim * 2

        scale = math.sqrt(1 / in_dim)
        weight_init = mx.random.uniform(
            low=-scale, high=scale, shape=(out_channels, 1, in_dim)
        )
        self.weight_g = normalize_weight(weight_init)
        self.weight_v = weight_init / (self.weight_g + 1e-12)
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight_g * self.weight_v / normalize_weight(self.weight_v)
        y = mx.conv1d(x, weight, stride=1, padding=0)
        return y + self.bias


class QuantizerOutProj(nn.Module):
    """Quantizer output projection.

    Uses weight normalization like NormConv1d with kernel_size=1.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        scale = math.sqrt(1 / in_dim)
        weight_init = mx.random.uniform(
            low=-scale, high=scale, shape=(out_dim, 1, in_dim)
        )
        self.weight_g = normalize_weight(weight_init)
        self.weight_v = weight_init / (self.weight_g + 1e-12)
        self.bias = mx.zeros((out_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight_g * self.weight_v / normalize_weight(self.weight_v)
        y = mx.conv1d(x, weight, stride=1, padding=0)
        return y + self.bias


# =============================================================================
# Full DACVAE Model
# =============================================================================


class DACVAE(nn.Module):
    """
    DACVAE audio codec for SAM-Audio.

    This is a VAE-style audio codec that encodes audio to a latent space
    and decodes it back. Unlike the standard DAC, this uses continuous
    latent representations instead of discrete codes.
    """

    def __init__(self, config: DACVAEConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self.hop_length = config.hop_length

        # Encoder
        self.encoder = Encoder(
            d_model=config.encoder_dim,
            strides=config.encoder_rates,
            d_latent=config.latent_dim,
        )

        # Quantizer projections (VAE-style)
        self.quantizer_in_proj = QuantizerInProj(config.latent_dim, config.codebook_dim)
        self.quantizer_out_proj = QuantizerOutProj(
            config.codebook_dim, config.latent_dim
        )

        # Decoder with watermarking
        self.decoder = Decoder(
            input_channel=config.latent_dim,
            channels=config.decoder_dim,
            rates=config.decoder_rates,
        )

    def _pad(self, wavs: mx.array) -> mx.array:
        """Pad waveform to be divisible by hop_length."""
        length = wavs.shape[1]
        if length % self.hop_length:
            pad_amount = self.hop_length - (length % self.hop_length)
            if pad_amount > 0:
                wavs = mx.pad(wavs, [(0, 0), (0, pad_amount), (0, 0)])
        return wavs

    def encode(self, waveform: mx.array) -> mx.array:
        """
        Encode waveform to latent representation.

        Args:
            waveform: Audio tensor of shape (batch, length, 1)

        Returns:
            Latent features of shape (batch, channels, frames)
        """
        waveform = self._pad(waveform)
        z = self.encoder(waveform)

        # VAE-style: project and take mean
        proj = self.quantizer_in_proj(z)
        mean, _ = mx.split(proj, 2, axis=-1)

        # Transpose to (batch, channels, frames)
        return mx.transpose(mean, (0, 2, 1))

    def decode(
        self,
        encoded_frames: mx.array,
        chunk_size: Optional[int] = None,
    ) -> mx.array:
        """
        Decode latent features back to waveform.

        For SAM-Audio, this accepts features in codebook_dim space (128)
        and projects them to latent_dim before decoding.

        Args:
            encoded_frames: Tensor of shape (batch, codebook_dim, frames)
                           Features in VAE codebook space.
            chunk_size: If provided, decode in chunks of this many frames
                       to reduce peak memory usage. Recommended: 50-100 for
                       long audio. None for single-pass decoding.

        Returns:
            Waveform of shape (batch, length, 1)
        """
        # Use chunked decoding for memory efficiency if requested
        if chunk_size is not None:
            return self._decode_chunked(encoded_frames, chunk_size)

        # Transpose to (batch, frames, codebook_dim)
        encoded_frames = mx.transpose(encoded_frames, (0, 2, 1))

        # Project from codebook_dim to latent_dim
        emb = self.quantizer_out_proj(encoded_frames)

        # Decode
        out = self.decoder(emb)

        # Apply final output
        out = self.decoder.snake_out(out)
        out = mx.tanh(self.decoder.conv_out(out))

        return out

    def _decode_chunked(
        self,
        encoded_frames: mx.array,
        chunk_size: int,
        overlap: int = 4,
    ) -> mx.array:
        """
        Decode in chunks to reduce peak memory usage.

        Args:
            encoded_frames: (batch, codebook_dim, frames)
            chunk_size: Number of frames per chunk
            overlap: Number of overlapping frames for smooth transitions

        Returns:
            Waveform of shape (batch, length, 1)
        """
        _, _, total_frames = encoded_frames.shape

        # Transpose to (batch, frames, codebook_dim)
        encoded_frames = mx.transpose(encoded_frames, (0, 2, 1))

        # Calculate output samples per frame (hop_length)
        samples_per_frame = self.hop_length
        overlap_samples = overlap * samples_per_frame

        chunks = []
        start = 0

        while start < total_frames:
            end = min(start + chunk_size, total_frames)

            # Extract chunk
            chunk = encoded_frames[:, start:end, :]

            # Project from codebook_dim to latent_dim
            emb = self.quantizer_out_proj(chunk)

            # Decode
            out = self.decoder(emb)
            out = self.decoder.snake_out(out)
            out = mx.tanh(self.decoder.conv_out(out))
            mx.eval(out)

            chunks.append(out)

            # Move to next chunk (with overlap for blending)
            if end >= total_frames:
                break
            start = end - overlap

            # Clear cache between chunks
            mx.clear_cache()

        # Concatenate chunks with crossfade blending
        if len(chunks) == 1:
            return chunks[0]

        # Simple concatenation with crossfade
        result_parts = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: keep all but overlap at end
                if len(chunks) > 1:
                    fade_out_start = chunk.shape[1] - overlap_samples
                    # Apply fade out to overlap region
                    fade = mx.linspace(1.0, 0.0, overlap_samples)
                    fade = fade.reshape(1, -1, 1)
                    chunk_main = chunk[:, :fade_out_start, :]
                    chunk_fade = chunk[:, fade_out_start:, :] * fade
                    result_parts.append(chunk_main)
                    result_parts.append(chunk_fade)
                else:
                    result_parts.append(chunk)
            elif i == len(chunks) - 1:
                # Last chunk: fade in the overlap, keep rest
                fade_in = mx.linspace(0.0, 1.0, overlap_samples)
                fade_in = fade_in.reshape(1, -1, 1)
                chunk_fade = chunk[:, :overlap_samples, :] * fade_in
                chunk_rest = chunk[:, overlap_samples:, :]
                # Add the fade-in part to the previous fade-out
                result_parts[-1] = result_parts[-1] + chunk_fade
                result_parts.append(chunk_rest)
            else:
                # Middle chunks: fade in at start, fade out at end
                fade_in = mx.linspace(0.0, 1.0, overlap_samples)
                fade_in = fade_in.reshape(1, -1, 1)
                fade_out = mx.linspace(1.0, 0.0, overlap_samples)
                fade_out = fade_out.reshape(1, -1, 1)

                chunk_fade_in = chunk[:, :overlap_samples, :] * fade_in
                chunk_middle = chunk[:, overlap_samples:-overlap_samples, :]
                chunk_fade_out = chunk[:, -overlap_samples:, :] * fade_out

                # Add fade-in to previous fade-out
                result_parts[-1] = result_parts[-1] + chunk_fade_in
                result_parts.append(chunk_middle)
                result_parts.append(chunk_fade_out)

        return mx.concatenate(result_parts, axis=1)

    def decode_streaming(
        self,
        encoded_frames: mx.array,
        chunk_size: int = 50,
        overlap: int = 4,
    ) -> Generator[Tuple[mx.array, bool], None, None]:
        """
        Streaming decode that yields chunks one at a time for minimal memory usage.

        This generator yields decoded audio chunks progressively, allowing you to
        write them to disk or process them without holding all decoded audio in memory.

        Args:
            encoded_frames: (batch, codebook_dim, frames) - encoded features
            chunk_size: Number of frames per chunk (default: 50)
            overlap: Number of overlapping frames for smooth transitions (default: 4)

        Yields:
            Tuple of (audio_chunk, is_last):
                - audio_chunk: Decoded audio of shape (batch, samples, 1)
                - is_last: True if this is the final chunk

        Example:
            ```python
            # Stream decode and write to file progressively
            all_audio = []
            for chunk, is_last in codec.decode_streaming(features, chunk_size=50):
                all_audio.append(chunk)
                if is_last:
                    break

            # Or write directly to disk with audio_io
            from mlx_audio.audio_io import write as audio_write
            import numpy as np
            all_chunks = []
            for chunk, is_last in codec.decode_streaming(features):
                all_chunks.append(chunk[0, :, 0])
            audio_write('output.wav', np.concatenate(all_chunks), 48000)
            ```
        """
        _, _, total_frames = encoded_frames.shape

        # Handle edge case: no frames
        if total_frames == 0:
            return

        # Transpose to (batch, frames, codebook_dim)
        encoded_frames = mx.transpose(encoded_frames, (0, 2, 1))
        mx.eval(encoded_frames)

        # Calculate overlap in samples
        samples_per_frame = self.hop_length
        overlap_samples = overlap * samples_per_frame

        # State for crossfade blending
        prev_fade_out: Optional[mx.array] = None
        start = 0
        chunk_idx = 0

        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            is_last = end >= total_frames

            # Extract and decode chunk
            chunk = encoded_frames[:, start:end, :]
            emb = self.quantizer_out_proj(chunk)
            out = self.decoder(emb)
            out = self.decoder.snake_out(out)
            out = mx.tanh(self.decoder.conv_out(out))
            mx.eval(out)

            # Get actual output length
            out_samples = out.shape[1]

            # Handle crossfade blending
            if chunk_idx == 0:
                # First chunk
                if not is_last and out_samples > overlap_samples:
                    fade_out_start = out_samples - overlap_samples
                    # Store fade-out region for blending with next chunk
                    fade_out = mx.linspace(1.0, 0.0, overlap_samples).reshape(1, -1, 1)
                    prev_fade_out = out[:, fade_out_start:, :] * fade_out
                    mx.eval(prev_fade_out)  # Ensure it's materialized before yielding
                    # Yield main part (without overlap)
                    result = out[:, :fade_out_start, :]
                    mx.eval(result)
                    yield result, False
                else:
                    # Single chunk or too small for overlap - yield everything
                    yield out, True
                    return
            elif is_last:
                # Last chunk: blend start with previous overlap, yield all
                if prev_fade_out is not None and out_samples >= overlap_samples:
                    fade_in = mx.linspace(0.0, 1.0, overlap_samples).reshape(1, -1, 1)
                    blended = prev_fade_out + out[:, :overlap_samples, :] * fade_in
                    mx.eval(blended)
                    # Yield blended overlap + rest of chunk
                    final_chunk = mx.concatenate(
                        [blended, out[:, overlap_samples:, :]], axis=1
                    )
                    mx.eval(final_chunk)
                    yield final_chunk, True
                else:
                    # No previous overlap or chunk too small
                    yield out, True
                return
            else:
                # Middle chunk: blend start, prepare end for next blend
                if prev_fade_out is not None and out_samples > 2 * overlap_samples:
                    fade_in = mx.linspace(0.0, 1.0, overlap_samples).reshape(1, -1, 1)
                    fade_out = mx.linspace(1.0, 0.0, overlap_samples).reshape(1, -1, 1)

                    # Blend with previous overlap
                    blended = prev_fade_out + out[:, :overlap_samples, :] * fade_in
                    mx.eval(blended)

                    # Prepare overlap for next chunk
                    fade_out_start = out_samples - overlap_samples
                    prev_fade_out = out[:, fade_out_start:, :] * fade_out
                    mx.eval(prev_fade_out)

                    # Yield blended + middle section
                    middle_chunk = mx.concatenate(
                        [blended, out[:, overlap_samples:fade_out_start, :]], axis=1
                    )
                    mx.eval(middle_chunk)
                    yield middle_chunk, False
                else:
                    # Chunk too small for proper overlap handling
                    yield out, False

            # Move to next chunk
            start = end - overlap
            chunk_idx += 1

            # Clear cache between chunks
            mx.clear_cache()

    def decode_stream(
        self,
        encoded_frames: mx.array,
        callback: Callable[[mx.array, int, bool], None],
        chunk_size: int = 50,
        overlap: int = 4,
    ) -> int:
        """
        Stream decode with callback for each chunk - ideal for writing directly to disk.

        This method decodes audio in chunks and calls the provided callback for each
        chunk, allowing progressive writes to disk without holding all audio in memory.

        Args:
            encoded_frames: (batch, codebook_dim, frames) - encoded features
            callback: Function called for each chunk with signature:
                      callback(audio_chunk, chunk_index, is_last) -> None
                      - audio_chunk: (batch, samples, 1) decoded audio
                      - chunk_index: 0-based index of the chunk
                      - is_last: True if this is the final chunk
            chunk_size: Number of frames per chunk (default: 50)
            overlap: Number of overlapping frames for smooth transitions (default: 4)

        Returns:
            Total number of samples decoded

        Example:
            ```python
            from mlx_audio.audio_io import write as audio_write
            import numpy as np

            # Collect chunks and write to file
            all_chunks = []
            def collect_chunk(chunk, idx, is_last):
                # Collect chunk (batch=0, squeeze channel dim)
                audio_np = np.array(chunk[0, :, 0])
                all_chunks.append(audio_np)

            total_samples = codec.decode_stream(features, collect_chunk)
            audio_write('output.wav', np.concatenate(all_chunks), 48000)
            print(f"Wrote {total_samples} samples")
            ```
        """
        total_samples = 0
        chunk_idx = 0

        for audio_chunk, is_last in self.decode_streaming(
            encoded_frames, chunk_size=chunk_size, overlap=overlap
        ):
            callback(audio_chunk, chunk_idx, is_last)
            total_samples += audio_chunk.shape[1]
            chunk_idx += 1

        return total_samples

    def decode_streaming(
        self,
        encoded_frames: mx.array,
        chunk_size: int = 50,
        overlap: int = 4,
    ) -> Generator[Tuple[mx.array, bool], None, None]:
        """
        Streaming decode that yields chunks one at a time for minimal memory usage.

        This generator yields decoded audio chunks progressively, allowing you to
        write them to disk or process them without holding all decoded audio in memory.

        Args:
            encoded_frames: (batch, codebook_dim, frames) - encoded features
            chunk_size: Number of frames per chunk (default: 50)
            overlap: Number of overlapping frames for smooth transitions (default: 4)

        Yields:
            Tuple of (audio_chunk, is_last):
                - audio_chunk: Decoded audio of shape (batch, samples, 1)
                - is_last: True if this is the final chunk

        Example:
            ```python
            # Stream decode and write to file progressively
            all_audio = []
            for chunk, is_last in codec.decode_streaming(features, chunk_size=50):
                all_audio.append(chunk)
                if is_last:
                    break

            # Or write directly to disk with audio_io
            from mlx_audio.audio_io import write as audio_write
            import numpy as np
            all_chunks = []
            for chunk, is_last in codec.decode_streaming(features):
                all_chunks.append(chunk[0, :, 0])
            audio_write('output.wav', np.concatenate(all_chunks), 48000)
            ```
        """
        _, _, total_frames = encoded_frames.shape

        # Handle edge case: no frames
        if total_frames == 0:
            return

        # Transpose to (batch, frames, codebook_dim)
        encoded_frames = mx.transpose(encoded_frames, (0, 2, 1))
        mx.eval(encoded_frames)

        # Calculate overlap in samples
        samples_per_frame = self.hop_length
        overlap_samples = overlap * samples_per_frame

        # State for crossfade blending
        prev_fade_out: Optional[mx.array] = None
        start = 0
        chunk_idx = 0

        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            is_last = end >= total_frames

            # Extract and decode chunk
            chunk = encoded_frames[:, start:end, :]
            emb = self.quantizer_out_proj(chunk)
            out = self.decoder(emb)
            out = self.decoder.snake_out(out)
            out = mx.tanh(self.decoder.conv_out(out))
            mx.eval(out)

            # Get actual output length
            out_samples = out.shape[1]

            # Handle crossfade blending
            if chunk_idx == 0:
                # First chunk
                if not is_last and out_samples > overlap_samples:
                    fade_out_start = out_samples - overlap_samples
                    # Store fade-out region for blending with next chunk
                    fade_out = mx.linspace(1.0, 0.0, overlap_samples).reshape(1, -1, 1)
                    prev_fade_out = out[:, fade_out_start:, :] * fade_out
                    mx.eval(prev_fade_out)  # Ensure it's materialized before yielding
                    # Yield main part (without overlap)
                    result = out[:, :fade_out_start, :]
                    mx.eval(result)
                    yield result, False
                else:
                    # Single chunk or too small for overlap - yield everything
                    yield out, True
                    return
            elif is_last:
                # Last chunk: blend start with previous overlap, yield all
                if prev_fade_out is not None and out_samples >= overlap_samples:
                    fade_in = mx.linspace(0.0, 1.0, overlap_samples).reshape(1, -1, 1)
                    blended = prev_fade_out + out[:, :overlap_samples, :] * fade_in
                    mx.eval(blended)
                    # Yield blended overlap + rest of chunk
                    final_chunk = mx.concatenate(
                        [blended, out[:, overlap_samples:, :]], axis=1
                    )
                    mx.eval(final_chunk)
                    yield final_chunk, True
                else:
                    # No previous overlap or chunk too small
                    yield out, True
                return
            else:
                # Middle chunk: blend start, prepare end for next blend
                if prev_fade_out is not None and out_samples > 2 * overlap_samples:
                    fade_in = mx.linspace(0.0, 1.0, overlap_samples).reshape(1, -1, 1)
                    fade_out = mx.linspace(1.0, 0.0, overlap_samples).reshape(1, -1, 1)

                    # Blend with previous overlap
                    blended = prev_fade_out + out[:, :overlap_samples, :] * fade_in
                    mx.eval(blended)

                    # Prepare overlap for next chunk
                    fade_out_start = out_samples - overlap_samples
                    prev_fade_out = out[:, fade_out_start:, :] * fade_out
                    mx.eval(prev_fade_out)

                    # Yield blended + middle section
                    middle_chunk = mx.concatenate(
                        [blended, out[:, overlap_samples:fade_out_start, :]], axis=1
                    )
                    mx.eval(middle_chunk)
                    yield middle_chunk, False
                else:
                    # Chunk too small for proper overlap handling
                    yield out, False

            # Move to next chunk
            start = end - overlap
            chunk_idx += 1

            # Clear cache between chunks
            mx.clear_cache()

    def decode_stream(
        self,
        encoded_frames: mx.array,
        callback: Callable[[mx.array, int, bool], None],
        chunk_size: int = 50,
        overlap: int = 4,
    ) -> int:
        """
        Stream decode with callback for each chunk - ideal for writing directly to disk.

        This method decodes audio in chunks and calls the provided callback for each
        chunk, allowing progressive writes to disk without holding all audio in memory.

        Args:
            encoded_frames: (batch, codebook_dim, frames) - encoded features
            callback: Function called for each chunk with signature:
                      callback(audio_chunk, chunk_index, is_last) -> None
                      - audio_chunk: (batch, samples, 1) decoded audio
                      - chunk_index: 0-based index of the chunk
                      - is_last: True if this is the final chunk
            chunk_size: Number of frames per chunk (default: 50)
            overlap: Number of overlapping frames for smooth transitions (default: 4)

        Returns:
            Total number of samples decoded

        Example:
            ```python
            from mlx_audio.audio_io import write as audio_write
            import numpy as np

            # Collect chunks and write to file
            all_chunks = []
            def collect_chunk(chunk, idx, is_last):
                # Collect chunk (batch=0, squeeze channel dim)
                audio_np = np.array(chunk[0, :, 0])
                all_chunks.append(audio_np)

            total_samples = codec.decode_stream(features, collect_chunk)
            audio_write('output.wav', np.concatenate(all_chunks), 48000)
            print(f"Wrote {total_samples} samples")
            ```
        """
        total_samples = 0
        chunk_idx = 0

        for audio_chunk, is_last in self.decode_streaming(
            encoded_frames, chunk_size=chunk_size, overlap=overlap
        ):
            callback(audio_chunk, chunk_idx, is_last)
            total_samples += audio_chunk.shape[1]
            chunk_idx += 1

        return total_samples

    def __call__(self, waveform: mx.array) -> mx.array:
        """
        Encode waveform to codebook space (for SAM-Audio).

        This returns VAE mean features in codebook_dim (128) which is what
        SAM-Audio uses for flow matching.

        Args:
            waveform: Audio tensor of shape (batch, 1, length)

        Returns:
            Latent features of shape (batch, codebook_dim, frames)
        """
        # Transpose from (batch, 1, length) to (batch, length, 1)
        waveform = mx.transpose(waveform, (0, 2, 1))
        waveform = self._pad(waveform)

        # Encode to latent space
        z = self.encoder(waveform)  # (B, T, latent_dim)

        # Project to codebook space and take VAE mean
        proj = self.quantizer_in_proj(z)  # (B, T, 2*codebook_dim)
        mean, _ = mx.split(proj, 2, axis=-1)  # (B, T, codebook_dim)

        # Transpose to (batch, codebook_dim, frames)
        return mx.transpose(mean, (0, 2, 1))

    def wav_idx_to_feature_idx(
        self,
        wav_idx: Union[mx.array, int],
        sample_rate: Optional[int] = None,
    ) -> Union[mx.array, int]:
        """Convert waveform sample index to feature frame index."""
        if sample_rate is None:
            sample_rate = self.sample_rate

        orig_freq = sample_rate
        new_freq = self.sample_rate
        target_length = int(np.ceil(new_freq * wav_idx / orig_freq))
        res = int(np.ceil(target_length / self.hop_length))

        if isinstance(wav_idx, mx.array):
            return mx.array(res)
        return res

    def feature_idx_to_wav_idx(
        self,
        feature_idx: Union[mx.array, int],
        sample_rate: Optional[int] = None,
    ) -> Union[mx.array, int]:
        """Convert feature frame index to waveform sample index."""
        if sample_rate is None:
            sample_rate = self.sample_rate

        orig_freq = sample_rate
        new_freq = self.sample_rate
        wav_chunklen = feature_idx * self.hop_length * (orig_freq / new_freq)

        if isinstance(feature_idx, mx.array):
            return feature_idx * self.hop_length
        return int(wav_chunklen)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
    ) -> "DACVAE":
        path = fetch_from_hub(repo_id)
        if path is None:
            raise ValueError(f"Could not find model {path}")

        model_path = Path(path) / "model.safetensors"
        config_path = Path(path) / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        dac_vae = DACVAE(DACVAEConfig(**config))
        dac_vae.load_weights(model_path.as_posix())
        mx.eval(dac_vae.parameters())

        return dac_vae


# fetch model from hub


def fetch_from_hub(hf_repo: str) -> Path:
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )
    return model_path
