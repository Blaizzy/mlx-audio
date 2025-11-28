# Ported from https://github.com/shivammehta25/Matcha-TTS
# Original: matcha/models/components/decoder.py

import math

import mlx.core as mx
import mlx.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def __call__(self, x: mx.array, scale: float = 1000) -> mx.array:
        if x.ndim < 1:
            x = mx.expand_dims(x, 0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
        emb = scale * mx.expand_dims(x, 1) * mx.expand_dims(emb, 0)
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """MLP for timestep embedding."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
        self.act_fn = act_fn

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = nn.silu(sample) if self.act_fn == "silu" else nn.gelu(sample)
        sample = self.linear_2(sample)
        return sample


class Block1D(nn.Module):
    """1D convolutional block with group norm."""

    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        # x is (B, C, T) but MLX Conv1d expects (B, T, C)
        x_in = mx.swapaxes(x * mask, 1, 2)  # (B, C, T) -> (B, T, C)
        output = self.conv(x_in)
        output = mx.swapaxes(output, 1, 2)  # (B, T, C) -> (B, C, T)
        output = self.norm(output)
        output = nn.mish(output)
        return output * mask


class ResnetBlock1D(nn.Module):
    """1D ResNet block with time embedding."""

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        # MLP: Mish activation is applied before the linear layer to match original
        self.mlp_linear = nn.Linear(time_emb_dim, dim_out)
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def __call__(self, x: mx.array, mask: mx.array, time_emb: mx.array) -> mx.array:
        h = self.block1(x, mask)
        # Original: h += self.mlp(time_emb) where mlp = Sequential(Mish(), Linear())
        # So Mish is applied first, then Linear
        h = h + mx.expand_dims(self.mlp_linear(nn.mish(time_emb)), -1)
        h = self.block2(h, mask)
        # res_conv: (B, C, T) -> transpose -> conv -> transpose back
        x_res = mx.swapaxes(x * mask, 1, 2)  # (B, C, T) -> (B, T, C)
        res_out = self.res_conv(x_res)
        res_out = mx.swapaxes(res_out, 1, 2)  # (B, T, C) -> (B, C, T)
        output = h + res_out
        return output


class Downsample1D(nn.Module):
    """1D downsampling with stride-2 convolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x is (B, C, T) but MLX Conv1d expects (B, T, C)
        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
        x = self.conv(x)
        x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)
        return x


class Upsample1D(nn.Module):
    """1D upsampling with transposed convolution."""

    def __init__(self, channels: int, use_conv_transpose: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv_transpose = use_conv_transpose

        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(
                channels, channels, kernel_size=4, stride=2, padding=1
            )
        else:
            self.conv = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_conv_transpose:
            # x is (B, C, T) but MLX ConvTranspose1d expects (B, T, C)
            x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)
            x = self.conv(x)
            x = mx.swapaxes(x, 1, 2)  # (B, T, C) -> (B, C, T)
            return x
        else:
            # Nearest neighbor upsampling
            B, C, T = x.shape
            x = mx.repeat(x, 2, axis=2)
            return x
