"""SAME (Stereo Audio Masked autoEncoder) decoder for Stable Audio 3.

Decodes 256-dim latent sequences to stereo 44.1kHz audio via:
  SoftNormBottleneck → TransformerResamplingBlock (16x upsample)
  → Conv1d mapping → patch unpacking (256x upsample) → stereo audio

Total downsampling ratio: 4096 (256 patch_size × 16 stride).
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .dit import RotaryEmbedding, apply_rotary_emb


@dataclass
class SAMEConfig:
    latent_dim: int = 256
    patch_size: int = 256
    audio_channels: int = 2
    encoder_channels: int = 128
    encoder_c_mults: list = field(default_factory=lambda: [6])
    encoder_strides: list = field(default_factory=lambda: [16])
    encoder_depths: list = field(default_factory=lambda: [6])
    dim_heads: int = 64
    downsampling_ratio: int = 4096
    ff_mult: int = 3
    chunk_size: int = 32
    chunk_midpoint_shift: bool = True
    conv_mapping: bool = True
    differential: bool = True
    decoder_out_channels: int = 512


class SAMEBottleneck(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scaling_factor = mx.ones((1, dim, 1))
        self.bias = mx.zeros((1, dim, 1))
        # running_std is learned during training; decode multiplies by it
        self.running_std = mx.ones((1,))

    def decode(self, x: mx.array) -> mx.array:
        return x * self.running_std


class DynamicTanh(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = mx.array([4.0])
        self.gamma = mx.ones((dim,))
        self.beta = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return self.gamma * mx.tanh(self.alpha * x) + self.beta


class _NoOp(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class SAMEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x, gate = mx.split(x, 2, axis=-1)
        return x * nn.silu(gate)


class SAMEFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 3):
        super().__init__()
        inner_dim = dim * mult
        self.ff = [SAMEGLU(dim, inner_dim), _NoOp(), nn.Linear(inner_dim, dim), _NoOp()]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ff[0](x)
        x = self.ff[2](x)
        return x


class SAMEDiffAttention(nn.Module):
    """Differential attention: attn(q,k,v) - attn(q_diff,k_diff,v)."""

    def __init__(self, dim: int, dim_heads: int = 64):
        super().__init__()
        self.dim_heads = dim_heads
        self.num_heads = dim // dim_heads
        self.to_qkv = nn.Linear(dim, dim * 5, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.q_norm = DynamicTanh(dim_heads)
        self.k_norm = DynamicTanh(dim_heads)

    def __call__(self, x: mx.array, freqs: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        h, dh = self.num_heads, self.dim_heads

        qkv = self.to_qkv(x)
        q, k, v, q_diff, k_diff = mx.split(qkv, 5, axis=-1)

        def reshape(t):
            return t.reshape(B, L, h, dh).transpose(0, 2, 1, 3)

        q, k, v = reshape(q), reshape(k), reshape(v)
        q_diff, k_diff = reshape(q_diff), reshape(k_diff)

        q, k = self.q_norm(q), self.k_norm(k)
        q_diff, k_diff = self.q_norm(q_diff), self.k_norm(k_diff)

        if freqs is not None:
            q = apply_rotary_emb(q, freqs)
            k = apply_rotary_emb(k, freqs)
            q_diff = apply_rotary_emb(q_diff, freqs)
            k_diff = apply_rotary_emb(k_diff, freqs)

        scale = dh**-0.5

        attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
        out = attn @ v

        attn_diff = mx.softmax((q_diff @ k_diff.transpose(0, 1, 3, 2)) * scale, axis=-1)
        out_diff = attn_diff @ v

        out = (out - out_diff).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.to_out(out)


class SAMETransformerBlock(nn.Module):
    def __init__(self, dim: int, dim_heads: int = 64, ff_mult: int = 3):
        super().__init__()
        self.pre_norm = DynamicTanh(dim)
        self.self_attn = SAMEDiffAttention(dim, dim_heads)
        self.ff_norm = DynamicTanh(dim)
        self.ff = SAMEFeedForward(dim, ff_mult)
        # Only first half of dim_heads gets RoPE (partial rotation)
        self.rope = RotaryEmbedding(dim_heads // 2)

    def __call__(self, x: mx.array) -> mx.array:
        freqs = self.rope(x.shape[1])
        x = x + self.self_attn(self.pre_norm(x), freqs)
        x = x + self.ff(self.ff_norm(x))
        return x


class SAMEResamplingBlock(nn.Module):
    """Upsamples via token expansion: each input token spawns stride new tokens."""

    def __init__(self, config: SAMEConfig):
        super().__init__()
        inner_dim = config.encoder_channels * config.encoder_c_mults[0]
        out_channels = config.decoder_out_channels
        stride = config.encoder_strides[0]
        depth = config.encoder_depths[0]
        self.stride = stride
        self.chunk_size = config.chunk_size
        self.chunk_midpoint_shift = config.chunk_midpoint_shift

        kernel_size = 3 if config.conv_mapping else 1
        self.mapping = nn.Conv1d(
            inner_dim, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.new_tokens = mx.zeros((1, 1, inner_dim))
        self.transformers = [
            SAMETransformerBlock(inner_dim, config.dim_heads, config.ff_mult)
            for _ in range(depth)
        ]

    def __call__(self, x: mx.array, stride: Optional[int] = None) -> mx.array:
        if stride is None:
            stride = self.stride
        B = x.shape[0]

        x = x.transpose(0, 2, 1)

        pad_modulo = self.chunk_size // stride
        L = x.shape[1]
        if L % pad_modulo != 0:
            pad_len = pad_modulo - (L % pad_modulo)
            x = mx.pad(x, [(0, 0), (0, pad_len), (0, 0)])

        n = x.shape[1]
        x = x.reshape(B * n, 1, -1)
        new_tokens = mx.broadcast_to(self.new_tokens, (B * n, stride, x.shape[-1]))
        x = mx.concatenate([x, new_tokens], axis=1)
        sub_chunk_size = stride + 1
        x = x.reshape(B, n * sub_chunk_size, -1)

        effective_chunk_size = self.chunk_size + pad_modulo

        if self.chunk_midpoint_shift:
            split = len(self.transformers) // 2
            shift = effective_chunk_size // 2
            nc = x.shape[1] // effective_chunk_size

            x = x.reshape(B * nc, effective_chunk_size, -1)
            for layer in self.transformers[:split]:
                x = layer(x)
            x = x.reshape(B, nc * effective_chunk_size, -1)

            x = mx.concatenate([x[:, :shift, :], x, x[:, -shift:, :]], axis=1)
            nc_shifted = x.shape[1] // effective_chunk_size
            x = x.reshape(B * nc_shifted, effective_chunk_size, -1)
            for layer in self.transformers[split:]:
                x = layer(x)
            x = x.reshape(B, nc_shifted * effective_chunk_size, -1)
            x = x[:, shift:-shift, :]
        else:
            nc = x.shape[1] // effective_chunk_size
            x = x.reshape(B * nc, effective_chunk_size, -1)
            for layer in self.transformers:
                x = layer(x)
            x = x.reshape(B, nc * effective_chunk_size, -1)

        x = x.reshape(B * n, sub_chunk_size, -1)
        x = x[:, -stride:, :]
        x = x.reshape(B, -1, x.shape[-1])

        x = self.mapping(x)
        x = x.transpose(0, 2, 1)
        return x


class _Transpose(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x.transpose(0, 2, 1)


class SAMEDecoder(nn.Module):
    def __init__(self, config: SAMEConfig):
        super().__init__()
        inner_dim = config.encoder_channels * config.encoder_c_mults[0]
        self.layers = [
            _Transpose(),
            nn.Linear(config.latent_dim, inner_dim),
            _Transpose(),
            SAMEResamplingBlock(config),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class SAMEEncoderResamplingBlock(nn.Module):
    """Encoder resampling block — only needed for weight loading."""

    def __init__(self, config: SAMEConfig):
        super().__init__()
        inner_dim = config.encoder_channels * config.encoder_c_mults[0]
        in_channels = config.decoder_out_channels
        self.mapping = nn.Conv1d(in_channels, inner_dim, 1)
        self.new_tokens = mx.zeros((1, 1, inner_dim))
        self.transformers = [
            SAMETransformerBlock(inner_dim, config.dim_heads, config.ff_mult)
            for _ in range(config.encoder_depths[0])
        ]


class SAMEEncoderStub(nn.Module):
    """Encoder stub — only needed for weight loading, not used during inference."""

    def __init__(self, config: SAMEConfig):
        super().__init__()
        inner_dim = config.encoder_channels * config.encoder_c_mults[0]
        self.layers = [
            SAMEEncoderResamplingBlock(config),
            _NoOp(),
            nn.Linear(inner_dim, config.latent_dim),
        ]


class SAMEAutoencoder(nn.Module):
    def __init__(self, config: SAMEConfig):
        super().__init__()
        self.bottleneck = SAMEBottleneck(config.latent_dim)
        self.encoder = SAMEEncoderStub(config)
        self.decoder = SAMEDecoder(config)

    def decode(self, latents: mx.array) -> mx.array:
        x = self.bottleneck.decode(latents)
        return self.decoder(x)


class Pretransform(nn.Module):
    """Wraps SAME autoencoder; handles patch unpacking on decode."""

    def __init__(self, config: SAMEConfig):
        super().__init__()
        self.model = SAMEAutoencoder(config)
        self.patch_size = config.patch_size
        self.audio_channels = config.audio_channels
        self.downsampling_ratio = config.downsampling_ratio

    def decode(self, latents: mx.array) -> mx.array:
        x = self.model.decode(latents)
        B, C, T = x.shape
        x = x.reshape(B, self.audio_channels, self.patch_size, T)
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(B, self.audio_channels, T * self.patch_size)
        return x
