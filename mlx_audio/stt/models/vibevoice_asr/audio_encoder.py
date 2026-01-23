# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import AcousticTokenizerConfig, SemanticTokenizerConfig


class ConvRMSNorm(nn.Module):
    """RMSNorm for convolutional layers - operates on channel dimension."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: [B, C, T] -> transpose to [B, T, C] for norm
        x = x.transpose(0, 2, 1)
        # RMSNorm
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        if self.elementwise_affine:
            x = x * self.weight
        # Transpose back to [B, C, T]
        return x.transpose(0, 2, 1)


class SConv1d(nn.Module):
    """
    Causal Conv1d with proper padding for streaming support.

    For causal convolutions, all padding is applied to the left side.
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
        causal: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.causal = causal
        self.pad_mode = pad_mode

        # Calculate total padding needed
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)

        # Create the convolution layer (no padding in conv, we'll handle it manually)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )

    def _get_extra_padding(self, length: int) -> int:
        """Calculate extra padding for stride alignment."""
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (
            self.kernel_size - self.padding_total
        )
        return int(ideal_length - length)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with causal padding.

        Args:
            x: Input tensor of shape [B, T, C] (MLX convention)

        Returns:
            Output tensor of shape [B, T', C']
        """
        B, T, C = x.shape

        # Calculate extra padding for stride alignment
        extra_padding = self._get_extra_padding(T)

        if self.causal:
            # Left padding for causal
            padding_left = self.padding_total
            padding_right = extra_padding
        else:
            # Symmetric padding for non-causal
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
            padding_right += extra_padding

        # Apply padding
        if padding_left > 0 or padding_right > 0:
            # MLX pad format: list of (before, after) tuples for each dimension
            # x is [B, T, C], we pad the T dimension
            x = mx.pad(x, [(0, 0), (padding_left, padding_right), (0, 0)])

        # Apply convolution
        return self.conv(x)


class FFN(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, embed_dim: int, ffn_dim: int, bias: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.gelu(x)
        x = self.linear2(x)
        return x


class DepthwiseConv(nn.Module):
    """Depthwise separable convolution wrapper."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        causal: bool = True,
        pad_mode: str = "constant",
        bias: bool = True,
    ):
        super().__init__()
        self.conv = SConv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            groups=dim,  # Depthwise
            bias=bias,
            causal=causal,
            pad_mode=pad_mode,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Block1D(nn.Module):
    """
    Transformer-style block with depthwise conv mixer and FFN.

    Structure:
    - Pre-norm -> Mixer (depthwise conv) -> Residual
    - Pre-norm -> FFN -> Residual
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        mixer_layer: str = "depthwise_conv",
        layernorm: str = "RMSNorm",
        eps: float = 1e-6,
        causal: bool = True,
        pad_mode: str = "constant",
        bias: bool = True,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        # Normalization layers
        if layernorm == "RMSNorm":
            self.norm = ConvRMSNorm(dim, eps=eps)
            self.ffn_norm = ConvRMSNorm(dim, eps=eps)
        else:
            self.norm = nn.LayerNorm(dim, eps=eps)
            self.ffn_norm = nn.LayerNorm(dim, eps=eps)

        # Mixer (depthwise conv)
        if mixer_layer == "depthwise_conv":
            self.mixer = DepthwiseConv(
                dim=dim,
                kernel_size=kernel_size,
                causal=causal,
                pad_mode=pad_mode,
                bias=bias,
            )
        else:
            # Regular conv
            self.mixer = SConv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kernel_size,
                stride=1,
                groups=1,
                bias=bias,
                causal=causal,
                pad_mode=pad_mode,
            )

        # FFN
        ffn_dim = dim * 4
        self.ffn = FFN(dim, ffn_dim, bias=bias)

        # Layer scale
        if layer_scale_init_value > 0:
            self.gamma = mx.ones((dim,)) * layer_scale_init_value
            self.ffn_gamma = mx.ones((dim,)) * layer_scale_init_value
        else:
            self.gamma = None
            self.ffn_gamma = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, C]

        Returns:
            Output tensor [B, T, C]
        """
        # Mixer path (with conv, needs channel-last to channel-first conversion)
        residual = x
        # Transpose to [B, C, T] for ConvRMSNorm
        x_transposed = x.transpose(0, 2, 1)
        x_normed = self.norm(x_transposed)
        # Transpose back for conv (which expects [B, T, C])
        x_normed = x_normed.transpose(0, 2, 1)
        x_mixed = self.mixer(x_normed)
        if self.gamma is not None:
            x_mixed = x_mixed * self.gamma
        x = residual + x_mixed

        # FFN path
        residual = x
        x_transposed = x.transpose(0, 2, 1)
        x_normed = self.ffn_norm(x_transposed)
        x_normed = x_normed.transpose(0, 2, 1)
        x_ffn = self.ffn(x_normed)
        if self.ffn_gamma is not None:
            x_ffn = x_ffn * self.ffn_gamma
        x = residual + x_ffn

        return x


class TokenizerEncoder(nn.Module):
    """
    Encoder for VibeVoice tokenizer.

    Converts raw audio waveform to latent representations through:
    1. Downsample layers (index 0 is stem conv, rest are strided downsamples)
    2. Block1D transformer stages at each scale
    3. Head projection to latent dimension

    The flow is: for each i: downsample_layers[i] -> stages[i], then head.
    """

    def __init__(
        self,
        channels: int = 1,
        vae_dim: int = 64,
        n_filters: int = 32,
        ratios: list = None,
        depths: list = None,
        causal: bool = True,
        pad_mode: str = "constant",
        conv_bias: bool = True,
        layernorm: str = "RMSNorm",
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        mixer_layer: str = "depthwise_conv",
        layer_scale_init_value: float = 1e-6,
        disable_last_norm: bool = True,
    ):
        super().__init__()

        if ratios is None:
            ratios = [8, 5, 5, 4, 2, 2]
        if depths is None:
            depths = [3, 3, 3, 3, 3, 3, 8]

        self.channels = channels
        self.vae_dim = vae_dim
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))  # Reverse for encoding
        self.depths = depths
        self.causal = causal
        self.n_stages = len(depths)  # 7

        self.hop_length = int(np.prod(ratios))

        # Downsample layers: index 0 is the stem, rest are strided convs
        # This matches PyTorch structure: downsample_layers[0] = stem, [1..6] = downsamples
        self.downsample_layers = []

        # Index 0: Stem convolution (stride=1, kernel=7, channels -> n_filters)
        self.downsample_layers.append(
            SConv1d(
                in_channels=channels,
                out_channels=n_filters,
                kernel_size=7,
                stride=1,
                bias=conv_bias,
                causal=causal,
                pad_mode=pad_mode,
            )
        )

        # Indices 1..len(ratios): Strided downsample convolutions
        for i in range(len(self.ratios)):
            in_ch = n_filters * (2**i)
            out_ch = n_filters * (2 ** (i + 1))
            self.downsample_layers.append(
                SConv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=self.ratios[i] * 2,
                    stride=self.ratios[i],
                    bias=conv_bias,
                    causal=causal,
                    pad_mode=pad_mode,
                )
            )

        # Transformer blocks for each stage
        self.stages = []
        for i in range(self.n_stages):
            # Channel dim after downsample_layers[i]
            if i == 0:
                in_ch = n_filters
            else:
                in_ch = n_filters * (2**i)
            stage_blocks = []
            for _ in range(depths[i]):
                stage_blocks.append(
                    Block1D(
                        dim=in_ch,
                        kernel_size=7,
                        mixer_layer=mixer_layer,
                        layernorm=layernorm,
                        eps=layernorm_eps,
                        causal=causal,
                        pad_mode=pad_mode,
                        bias=conv_bias,
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            self.stages.append(stage_blocks)

        # Final norm (if enabled)
        final_channels = n_filters * (2 ** len(self.ratios))
        if not disable_last_norm:
            self.norm = ConvRMSNorm(final_channels, eps=layernorm_eps)
        else:
            self.norm = None

        # Head projection to latent dimension
        self.head = SConv1d(
            in_channels=final_channels,
            out_channels=vae_dim,
            kernel_size=7,
            stride=1,
            bias=conv_bias,
            causal=causal,
            pad_mode=pad_mode,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Encode audio to latent representation.

        Args:
            x: Audio input [B, 1, T] (batch, channels, time) or [B, T] for mono

        Returns:
            Latent representation [B, T', vae_dim]
        """
        # Ensure correct shape [B, T, C]
        if x.ndim == 2:
            x = x[:, :, None]  # Add channel dim: [B, T] -> [B, T, 1]
        elif x.ndim == 3 and x.shape[1] == 1:
            # Input is [B, 1, T], transpose to [B, T, 1]
            x = x.transpose(0, 2, 1)

        # Process through downsample + stage pairs
        for i in range(self.n_stages):
            x = self.downsample_layers[i](x)
            for block in self.stages[i]:
                x = block(x)

        # Final norm
        if self.norm is not None:
            x_transposed = x.transpose(0, 2, 1)
            x_normed = self.norm(x_transposed)
            x = x_normed.transpose(0, 2, 1)

        # Head projection
        x = self.head(x)

        return x  # [B, T', vae_dim]


class AcousticTokenizerEncoder(nn.Module):
    """Acoustic tokenizer encoder wrapper with config-based initialization."""

    def __init__(self, config: AcousticTokenizerConfig):
        super().__init__()
        self.config = config
        self.fix_std = config.fix_std
        self.std_dist_type = config.std_dist_type

        depths = config.parsed_encoder_depths

        self.encoder = TokenizerEncoder(
            channels=config.channels,
            vae_dim=config.vae_dim,
            n_filters=config.encoder_n_filters,
            ratios=config.encoder_ratios,
            depths=depths,
            causal=config.causal,
            pad_mode=config.pad_mode,
            conv_bias=config.conv_bias,
            layernorm=config.layernorm,
            layernorm_eps=config.layernorm_eps,
            layernorm_elementwise_affine=config.layernorm_elementwise_affine,
            mixer_layer=config.mixer_layer,
            layer_scale_init_value=config.layer_scale_init_value,
            disable_last_norm=config.disable_last_norm,
        )

    def encode(self, audio: mx.array) -> mx.array:
        """
        Encode audio to mean latent representation.

        Args:
            audio: Audio waveform [B, 1, T] or [B, T]

        Returns:
            Mean latent [B, T', vae_dim]
        """
        return self.encoder(audio)

    def sample(self, mean: mx.array) -> mx.array:
        """
        Sample from latent distribution.

        Args:
            mean: Mean latent [B, T, vae_dim]

        Returns:
            Sampled latent [B, T, vae_dim]
        """
        if self.std_dist_type == "gaussian":
            # Gaussian sampling with fixed std
            batch_size = mean.shape[0]
            value = self.fix_std / 0.8
            std = mx.random.normal((batch_size, 1, 1)) * value
            noise = mx.random.normal(mean.shape)
            return mean + std * noise
        elif self.std_dist_type == "fix":
            # Fixed std sampling
            noise = mx.random.normal(mean.shape)
            return mean + self.fix_std * noise
        else:
            # No sampling
            return mean

    def __call__(self, audio: mx.array) -> mx.array:
        """Encode and sample."""
        mean = self.encode(audio)
        return self.sample(mean)


class SemanticTokenizerEncoder(nn.Module):
    """Semantic tokenizer encoder wrapper with config-based initialization."""

    def __init__(self, config: SemanticTokenizerConfig):
        super().__init__()
        self.config = config

        depths = config.parsed_encoder_depths

        self.encoder = TokenizerEncoder(
            channels=config.channels,
            vae_dim=config.vae_dim,
            n_filters=config.encoder_n_filters,
            ratios=config.encoder_ratios,
            depths=depths,
            causal=config.causal,
            pad_mode=config.pad_mode,
            conv_bias=config.conv_bias,
            layernorm=config.layernorm,
            layernorm_eps=config.layernorm_eps,
            layernorm_elementwise_affine=config.layernorm_elementwise_affine,
            mixer_layer=config.mixer_layer,
            layer_scale_init_value=config.layer_scale_init_value,
            disable_last_norm=config.disable_last_norm,
        )

    def encode(self, audio: mx.array) -> mx.array:
        """
        Encode audio to semantic latent representation.

        Args:
            audio: Audio waveform [B, 1, T] or [B, T]

        Returns:
            Semantic latent [B, T', vae_dim]
        """
        return self.encoder(audio)

    def __call__(self, audio: mx.array) -> mx.array:
        """Encode audio (semantic tokenizer doesn't sample)."""
        return self.encode(audio)
