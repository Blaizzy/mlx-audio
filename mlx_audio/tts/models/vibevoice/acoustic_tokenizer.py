"""Acoustic tokenizer (VAE decoder) for VibeVoice - converts latents to audio."""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import AcousticTokenizerConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
        else:
            self.weight = None

    def _norm(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        if self.weight is not None:
            output = output * self.weight
        return output


class ConvRMSNorm(nn.Module):
    """RMSNorm for convolutional features (B, C, T) format."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
        else:
            self.weight = None

    def _norm(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) -> transpose to (B, T, C) for normalization
        x = mx.transpose(x, (0, 2, 1))
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        if self.weight is not None:
            output = output * self.weight
        # Transpose back to (B, C, T)
        return mx.transpose(output, (0, 2, 1))


class CausalConv1d(nn.Module):
    """Causal 1D convolution with padding on the left.

    Input/output format: (B, C, T) - batch, channels, time (PyTorch convention)
    MLX Conv1d expects: (B, T, C) - batch, time, channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation

        # Use MLX Conv1d with groups parameter
        # For grouped conv, MLX weight shape is (C_out, K, C_in/groups)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) - input in PyTorch format
        # Transpose to MLX format: (B, C, T) -> (B, T, C)
        x = mx.transpose(x, (0, 2, 1))

        # Add causal padding on the time dimension (now axis 1)
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])

        # Apply conv - MLX expects (B, T, C)
        x = self.conv(x)

        # Transpose back to PyTorch format: (B, T, C) -> (B, C, T)
        x = mx.transpose(x, (0, 2, 1))

        return x


class CausalConvTranspose1d(nn.Module):
    """Causal transposed 1D convolution for upsampling.

    Input/output format: (B, C, T) - batch, channels, time (PyTorch convention)
    MLX ConvTranspose1d expects: (B, T, C) - batch, time, channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.trim_right_ratio = trim_right_ratio

        # Calculate padding
        self.padding_total = kernel_size - stride

        # Use MLX ConvTranspose1d
        self.convtr = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) - input in PyTorch format
        # Transpose to MLX format: (B, C, T) -> (B, T, C)
        x = mx.transpose(x, (0, 2, 1))

        # Apply transposed conv
        x = self.convtr(x)

        # Transpose back to PyTorch format: (B, T, C) -> (B, C, T)
        x = mx.transpose(x, (0, 2, 1))

        # Trim padding for causal (on time dimension, now axis 2)
        padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
        padding_left = self.padding_total - padding_right

        if padding_left > 0:
            x = x[:, :, padding_left:]
        if padding_right > 0:
            x = x[:, :, :-padding_right]

        return x


class DepthwiseConv(nn.Module):
    """Depthwise separable convolution wrapped in a conv module."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        causal: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.causal = causal

        # Wrapped in another conv module (to match HF structure: mixer.conv.conv.conv)
        self.conv = CausalConv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Mixer(nn.Module):
    """Mixer module wrapping depthwise conv."""

    def __init__(
        self, dim: int, kernel_size: int = 7, causal: bool = True, bias: bool = True
    ):
        super().__init__()
        self.conv = DepthwiseConv(dim, kernel_size, causal, bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class FeedForward(nn.Module):
    """Feed-forward network with SiLU activation.

    Note: Uses linear1/linear2 naming to match HuggingFace weights.
    """

    def __init__(self, dim: int, mult: float = 4.0, bias: bool = True):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.linear1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.silu(x)
        x = self.linear2(x)
        return x


class Block1D(nn.Module):
    """1D convolutional block with depthwise conv and FFN."""

    def __init__(
        self,
        dim: int,
        layernorm: str = "RMSNorm",
        eps: float = 1e-6,
        causal: bool = True,
        bias: bool = True,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim

        # Normalization
        self.norm = ConvRMSNorm(dim, eps=eps)
        self.ffn_norm = ConvRMSNorm(dim, eps=eps)

        # Mixer (depthwise conv)
        self.mixer = Mixer(dim, kernel_size=7, causal=causal, bias=bias)

        # FFN
        self.ffn = FeedForward(dim, mult=4.0, bias=bias)

        # Layer scale - stored as parameters (gamma, ffn_gamma)
        if layer_scale_init_value > 0:
            self.gamma = mx.ones((dim,)) * layer_scale_init_value
            self.ffn_gamma = mx.ones((dim,)) * layer_scale_init_value
        else:
            self.gamma = None
            self.ffn_gamma = None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T)

        # Mixer path
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        if self.gamma is not None:
            x = x * mx.expand_dims(self.gamma, axis=(0, 2))
        x = residual + x

        # FFN path
        residual = x
        x = self.ffn_norm(x)
        # Transpose for FFN: (B, C, T) -> (B, T, C)
        x = mx.transpose(x, (0, 2, 1))
        x = self.ffn(x)
        # Transpose back: (B, T, C) -> (B, C, T)
        x = mx.transpose(x, (0, 2, 1))
        if self.ffn_gamma is not None:
            x = x * mx.expand_dims(self.ffn_gamma, axis=(0, 2))
        x = residual + x

        return x


class StemConv(nn.Module):
    """Stem convolution layer wrapped in Sequential structure to match HF."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class UpsampleLayer(nn.Module):
    """Upsample layer with transposed convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self.convtr = CausalConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.convtr(x)


class HeadConv(nn.Module):
    """Output head convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class TokenizerDecoder(nn.Module):
    """Decoder that converts latent representations back to audio.

    Architecture matches HuggingFace VibeVoice structure:
    - upsample_layers[0] is stem conv
    - upsample_layers[1-6] are transposed convolutions
    - stages[0-6] are transformer blocks
    - head is output convolution
    """

    def __init__(self, config: AcousticTokenizerConfig):
        super().__init__()

        self.dimension = config.vae_dim
        self.channels = config.channels
        self.n_filters = (
            config.decoder_n_filters
            if config.decoder_n_filters
            else config.encoder_n_filters
        )

        # Use decoder ratios or fallback to encoder ratios
        self.ratios = (
            config.decoder_ratios if config.decoder_ratios else config.encoder_ratios
        )

        # Parse depths - should be reversed encoder depths for decoder
        if config.decoder_depths:
            if isinstance(config.decoder_depths, str):
                self.depths = [int(d) for d in config.decoder_depths.split("-")]
            else:
                self.depths = config.decoder_depths
        else:
            if isinstance(config.encoder_depths, str):
                encoder_depths = [int(d) for d in config.encoder_depths.split("-")]
            else:
                encoder_depths = config.encoder_depths
            self.depths = list(reversed(encoder_depths))

        self.causal = config.causal
        self.n_stages = len(self.depths)

        # Upsample layers - wrapped in list structure to match HF naming
        # HF: upsample_layers.X.0.conv or upsample_layers.X.0.convtr
        self.upsample_layers = []

        # First upsample layer is stem conv (upsample_layers.0.0.conv)
        stem_out_ch = self.n_filters * (2 ** (self.n_stages - 1))
        self.upsample_layers.append(
            [
                StemConv(
                    in_channels=self.dimension,
                    out_channels=stem_out_ch,
                    kernel_size=7,
                    bias=config.conv_bias,
                )
            ]
        )

        # Remaining upsample layers are transposed convolutions
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** (self.n_stages - 1 - i))
            out_ch = (
                self.n_filters * (2 ** (self.n_stages - 2 - i))
                if i < len(self.ratios) - 1
                else self.n_filters
            )

            self.upsample_layers.append(
                [
                    UpsampleLayer(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=self.ratios[i] * 2,
                        stride=self.ratios[i],
                        bias=config.conv_bias,
                    )
                ]
            )

        # Transformer stages
        self.stages = []
        for i in range(self.n_stages):
            in_ch = self.n_filters * (2 ** (self.n_stages - 1 - i))
            stage_blocks = []
            for j in range(self.depths[i]):
                stage_blocks.append(
                    Block1D(
                        dim=in_ch,
                        layernorm=config.layernorm,
                        eps=config.layernorm_eps,
                        causal=config.causal,
                        bias=config.conv_bias,
                        layer_scale_init_value=config.layer_scale_init_value,
                    )
                )
            self.stages.append(stage_blocks)

        # Output head
        self.head = HeadConv(
            in_channels=self.n_filters,
            out_channels=config.channels,
            kernel_size=7,
            bias=config.conv_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Latent tensor of shape (B, T, D) or (B, D, T)

        Returns:
            Audio tensor of shape (B, 1, T')
        """
        # Ensure x is in (B, D, T) format
        if x.shape[1] != self.dimension:
            x = mx.transpose(x, (0, 2, 1))

        # Apply stem (first upsample layer)
        x = self.upsample_layers[0][0](x)

        # Process through stages and upsampling
        for i in range(self.n_stages):
            # Apply stage blocks
            for block in self.stages[i]:
                x = block(x)

            # Apply upsampling (skip first upsample which was stem)
            if i + 1 < len(self.upsample_layers):
                x = self.upsample_layers[i + 1][0](x)

        # Output head
        x = self.head(x)

        return x


class AcousticTokenizer(nn.Module):
    """VibeVoice acoustic tokenizer (decoder only for inference)."""

    def __init__(self, config: AcousticTokenizerConfig):
        super().__init__()
        self.config = config
        self.fix_std = config.fix_std
        self.std_dist_type = config.std_dist_type

        self.decoder = TokenizerDecoder(config)

    def decode(self, latents: mx.array) -> mx.array:
        """Convert latent representations to audio.

        Args:
            latents: Latent tensor of shape (B, T, D) where D = vae_dim

        Returns:
            Audio tensor of shape (B, 1, T')
        """
        return self.decoder(latents)

    def __call__(self, latents: mx.array) -> mx.array:
        """Alias for decode."""
        return self.decode(latents)
