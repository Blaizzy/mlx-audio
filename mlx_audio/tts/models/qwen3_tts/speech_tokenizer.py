# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


import math
import re
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import KVCache

from mlx_audio.codec.models.mimi.mimi import _reset_kv_cache
from mlx_audio.codec.models.mimi.modules import (
    ConvDownsample1d,
    ProjectedTransformer,
    SeanetConfig,
    SeanetEncoder,
)
from mlx_audio.codec.models.mimi.modules import (
    SplitResidualVectorQuantizer as MimiSplitRVQ,
)
from mlx_audio.codec.models.mimi.modules import TransformerConfig

from .config import (
    Qwen3TTSTokenizerConfig,
    Qwen3TTSTokenizerDecoderConfig,
    Qwen3TTSTokenizerEncoderConfig,
)


class DepthwiseConvWeight(nn.Module):
    """Container for depthwise conv weights to match PyTorch key structure."""

    def __init__(self, out_channels: int, kernel_size: int, in_per_group: int):
        super().__init__()
        # Weight shape: [out_channels, kernel, in_per_group] = [channels, kernel, 1] for depthwise
        self.weight = mx.zeros((out_channels, kernel_size, in_per_group))
        self.bias = mx.zeros((out_channels,))


class CausalConv1d(nn.Module):
    """Causal 1D convolution with proper padding.

    Supports grouped convolutions for depthwise convs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size_val = kernel_size
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

        if groups == 1:
            # Regular convolution
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation,
            )
        else:
            # Grouped/depthwise convolution - implement manually
            # Use a wrapper Module to hold weight/bias with matching key structure
            # Weight shape after MLX transpose: [out_channels, kernel, in_channels/groups]
            # For depthwise: [channels, kernel, 1]
            in_per_group = in_channels // groups
            self.conv = DepthwiseConvWeight(out_channels, kernel_size, in_per_group)

    def _get_extra_padding(self, length: int) -> int:
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (
            self.kernel_size - self.padding
        )
        return int(ideal_length - length)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time] (NCL format)
        extra_padding = self._get_extra_padding(x.shape[-1])
        # Pad on the left (causal) - padding is on time dimension (last)
        x = mx.pad(x, [(0, 0), (0, 0), (self.padding, extra_padding)])

        if self.groups == 1:
            # MLX Conv1d expects [batch, time, channels] (NLC format)
            x = mx.transpose(x, (0, 2, 1))  # NCL -> NLC
            x = self.conv(x)
            x = mx.transpose(x, (0, 2, 1))  # NLC -> NCL
            return x
        else:
            # Depthwise convolution: apply each filter to its corresponding channel
            # x: [batch, channels, time], weight: [channels, kernel, 1]
            batch, channels, time = x.shape
            kernel_size = self.conv.weight.shape[1]

            # Use einsum for efficient depthwise conv
            # Extract sliding windows: [batch, channels, time - kernel + 1, kernel]
            output_time = time - kernel_size + 1
            # Create windows using indexing
            windows = mx.stack(
                [x[..., i : i + output_time] for i in range(kernel_size)], axis=-1
            )  # [batch, channels, output_time, kernel]

            # Apply weights: weight is [channels, kernel, 1] -> squeeze to [channels, kernel]
            w = self.conv.weight.squeeze(-1)  # [channels, kernel]

            # Multiply and sum: [batch, channels, output_time, kernel] * [channels, kernel] -> [batch, channels, output_time]
            out = mx.sum(windows * w[None, :, None, :], axis=-1)

            # Add bias
            out = out + self.conv.bias[None, :, None]

            return out


class CausalTransposeConv1d(nn.Module):
    """Causal transposed 1D convolution for upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0
        )
        # Trim from the right for causal behavior (matches Encodec/DAC/Mimi)
        self.trim_right = kernel_size - stride

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time] (NCL format)
        # MLX ConvTranspose1d expects [batch, time, channels] (NLC format)
        x = mx.transpose(x, (0, 2, 1))  # NCL -> NLC
        x = self.conv(x)
        x = mx.transpose(x, (0, 2, 1))  # NLC -> NCL
        # Trim from right for causal behavior
        if self.trim_right > 0:
            x = x[..., : -self.trim_right]
        return x


class SnakeBeta(nn.Module):
    """Snake activation with learnable alpha and beta parameters.

    SnakeBeta(x) = x + (1/beta) * sin^2(x * alpha)
    """

    def __init__(self, channels: int, alpha: float = 1.0):
        super().__init__()
        self.channels = channels
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))
        self.eps = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time]
        alpha = mx.exp(self.alpha)[None, :, None]
        beta = mx.exp(self.beta)[None, :, None]
        return x + (1.0 / (beta + self.eps)) * mx.power(mx.sin(x * alpha), 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for feature processing."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = mx.ones((dim,)) * 1e-6

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = mx.transpose(x, (0, 2, 1))  # [B, T, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = mx.transpose(x, (0, 2, 1))  # [B, C, T]
        return residual + x


class DecoderRMSNorm(nn.Module):
    """RMS normalization for decoder."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        variance = mx.mean(x_float**2, axis=-1, keepdims=True)
        x_normed = x_float * mx.rsqrt(variance + self.eps)
        return (self.weight * x_normed).astype(x.dtype)


class LayerScale(nn.Module):
    """Layer scale for residual connections."""

    def __init__(self, channels: int, initial_scale: float = 0.01):
        super().__init__()
        self.scale = mx.ones((channels,)) * initial_scale

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x


class DecoderRotaryEmbedding(nn.Module):
    """Rotary position embedding for decoder transformer."""

    def __init__(
        self, dim: int, max_position_embeddings: int = 8000, base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self._inv_freq = inv_freq

    def __call__(
        self, x: mx.array, position_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        inv_freq = self._inv_freq[None, :, None].astype(mx.float32)
        pos = position_ids[:, None, :].astype(mx.float32)
        freqs = (inv_freq * pos).transpose(0, 2, 1)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).astype(x.dtype)
        sin = mx.sin(emb).astype(x.dtype)
        return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DecoderAttention(nn.Module):
    """Multi-head attention for decoder transformer."""

    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5
        self.sliding_window = config.sliding_window

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def __call__(
        self,
        x: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape

        q = (
            self.q_proj(x)
            .reshape(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache using mlx_lm's KVCache
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Use fast scaled dot product attention with GQA support
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(output)


class DecoderMLP(nn.Module):
    """MLP for decoder transformer."""

    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderTransformerLayer(nn.Module):
    """Transformer layer for decoder."""

    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DecoderAttention(config, layer_idx)
        self.mlp = DecoderMLP(config)
        self.input_layernorm = DecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DecoderRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn_layer_scale = LayerScale(
            config.hidden_size, config.layer_scale_initial_scale
        )
        self.mlp_layer_scale = LayerScale(
            config.hidden_size, config.layer_scale_initial_scale
        )

    def __call__(
        self,
        x: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_embeddings, mask, cache)
        x = residual + self.self_attn_layer_scale(x)

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + self.mlp_layer_scale(x)

        return x


class DecoderTransformer(nn.Module):
    """Transformer model for decoder."""

    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig):
        super().__init__()
        self.config = config
        self.layers = [
            DecoderTransformerLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = DecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = DecoderRotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)

    def make_cache(self) -> List[KVCache]:
        """Create KV cache for all layers."""
        return [KVCache() for _ in self.layers]

    def __call__(
        self,
        inputs_embeds: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
    ) -> mx.array:
        batch, seq_len, _ = inputs_embeds.shape

        x = self.input_proj(inputs_embeds)

        # Position ids - use KVCache.offset for position tracking
        offset = cache[0].offset if cache is not None else 0
        position_ids = mx.arange(offset, offset + seq_len)[None, :]
        position_ids = mx.broadcast_to(position_ids, (batch, seq_len))

        position_embeddings = self.rotary_emb(x, position_ids)

        if mask is None and seq_len > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(x.dtype)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, position_embeddings, mask, layer_cache)

        x = self.norm(x)
        x = self.output_proj(x)

        return x


class EuclideanCodebook(nn.Module):
    """Euclidean codebook for vector quantization.

    Uses nn.Embedding for the codebook to ensure proper weight loading.
    The embedding is computed from cluster_usage and embedding_sum during sanitization.
    """

    def __init__(self, dim: int, codebook_size: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.eps = eps
        # Use Embedding layer for proper weight loading
        self.embed = nn.Embedding(codebook_size, dim)

    def decode(self, codes: mx.array) -> mx.array:
        # codes: [batch, time]
        return self.embed(codes)


class VectorQuantization(nn.Module):
    """Vector quantization layer."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        codebook_dim = codebook_dim or dim
        requires_projection = codebook_dim != dim

        if requires_projection:
            self.project_out = nn.Linear(codebook_dim, dim)
        else:
            self.project_out = None

        self.codebook = EuclideanCodebook(codebook_dim, codebook_size, eps)
        self.codebook_size = codebook_size

    def decode(self, codes: mx.array) -> mx.array:
        # codes: [batch, time]
        quantized = self.codebook.decode(codes)  # [batch, time, codebook_dim]
        if self.project_out is not None:
            quantized = self.project_out(quantized)
        quantized = mx.transpose(quantized, (0, 2, 1))  # [batch, dim, time]
        return quantized


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization with multiple codebooks."""

    def __init__(
        self,
        num_quantizers: int,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = [
            VectorQuantization(dim, codebook_size, codebook_dim)
            for _ in range(num_quantizers)
        ]

    def decode(self, codes: mx.array) -> mx.array:
        # codes: [num_quantizers, batch, time]
        quantized = mx.zeros(
            (codes.shape[1], self.layers[0].codebook.dim, codes.shape[2])
        )
        for idx, layer_codes in enumerate(codes):
            quantized = quantized + self.layers[idx].decode(layer_codes)
        return quantized


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with input/output projections."""

    def __init__(
        self,
        dimension: int = 128,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        n_q: int = 8,
        bins: int = 1024,
        force_projection: bool = False,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins

        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = None
        else:
            self.input_proj = nn.Conv1d(
                self.input_dimension, self.dimension, 1, bias=False
            )

        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = None
        else:
            self.output_proj = nn.Conv1d(
                self.dimension, self.output_dimension, 1, bias=False
            )

        self.vq = ResidualVectorQuantization(
            num_quantizers=n_q,
            dim=dimension,
            codebook_size=bins,
        )

    def decode(self, codes: mx.array) -> mx.array:
        # codes: [batch, num_quantizers, time]
        codes = mx.transpose(codes, (1, 0, 2))  # [num_quantizers, batch, time]
        quantized = self.vq.decode(codes)  # [batch, dim, time]
        if self.output_proj is not None:
            # Conv1d expects [batch, time, channels], transpose for MLX
            quantized = mx.transpose(quantized, (0, 2, 1))  # [batch, time, dim]
            quantized = self.output_proj(quantized)  # [batch, time, output_dim]
            quantized = mx.transpose(quantized, (0, 2, 1))  # [batch, output_dim, time]
        return quantized


class SplitResidualVectorQuantizer(nn.Module):
    """Split RVQ with separate quantizers for semantic and acoustic."""

    def __init__(
        self,
        n_q: int = 8,
        n_q_semantic: int = 1,
        dimension: int = 128,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        bins: int = 1024,
    ):
        super().__init__()
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic

        self.rvq_first = ResidualVectorQuantizer(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q_semantic,
            bins=bins,
            force_projection=True,
        )
        self.rvq_rest = ResidualVectorQuantizer(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q - n_q_semantic,
            bins=bins,
            force_projection=True,
        )

    def decode(self, codes: mx.array) -> mx.array:
        # codes: [batch, num_quantizers, time]
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized = quantized + self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized


class DecoderResidualUnit(nn.Module):
    """Residual unit for decoder.

    PyTorch weight keys:
    - act1.alpha, act1.beta (SnakeBeta)
    - conv1.conv.weight, conv1.conv.bias (CausalConv1d)
    - act2.alpha, act2.beta (SnakeBeta)
    - conv2.conv.weight, conv2.conv.bias (CausalConv1d)
    """

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + residual


class DecoderBlockUpsample(nn.Module):
    """Upsample layer wrapper for decoder block.

    PyTorch weight keys: conv.weight, conv.bias

    Note: Implements causal transpose conv inline to match PyTorch key structure.
    """

    def __init__(self, in_dim: int, out_dim: int, upsample_rate: int):
        super().__init__()
        kernel_size = 2 * upsample_rate
        self.conv = nn.ConvTranspose1d(
            in_dim, out_dim, kernel_size, stride=upsample_rate, padding=0
        )
        # Trim from the right for causal behavior (matches Encodec/DAC/Mimi)
        self.trim_right = kernel_size - upsample_rate

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time] (NCL format)
        # MLX ConvTranspose1d expects [batch, time, channels] (NLC format)
        x = mx.transpose(x, (0, 2, 1))  # NCL -> NLC
        x = self.conv(x)
        x = mx.transpose(x, (0, 2, 1))  # NLC -> NCL
        # Trim from right for causal behavior
        if self.trim_right > 0:
            x = x[..., : -self.trim_right]
        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling.

    PyTorch structure - self.block is a ModuleList:
    - block[0]: SnakeBeta (alpha, beta)
    - block[1]: ConvWrapper (conv.weight, conv.bias)
    - block[2:]: DecoderResidualUnits (act1, conv1, act2, conv2)
    """

    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig, layer_idx: int):
        super().__init__()
        in_dim = config.decoder_dim // (2**layer_idx)
        out_dim = config.decoder_dim // (2 ** (layer_idx + 1))
        upsample_rate = config.upsample_rates[layer_idx]

        # PyTorch uses self.block as a ModuleList
        self.block = [
            SnakeBeta(in_dim),  # block[0]: snake
            DecoderBlockUpsample(in_dim, out_dim, upsample_rate),  # block[1]: upsample
            DecoderResidualUnit(out_dim, dilation=1),  # block[2]: residual unit
            DecoderResidualUnit(out_dim, dilation=3),  # block[3]: residual unit
            DecoderResidualUnit(out_dim, dilation=9),  # block[4]: residual unit
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class DecoderInitialConv(nn.Module):
    """Wrapper for initial decoder conv to match PyTorch weight structure.

    PyTorch key: decoder.decoder.0.conv.{weight,bias}
    Note: Uses nn.Conv1d directly, not CausalConv1d wrapper.
    """

    def __init__(self, latent_dim: int, decoder_dim: int, kernel_size: int = 7):
        super().__init__()
        # Note: Causal padding handled inline
        self.conv = nn.Conv1d(latent_dim, decoder_dim, kernel_size, padding=0)
        self.kernel_size = kernel_size

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time] (NCL format)
        # Causal padding (left pad on time dimension)
        x = mx.pad(x, [(0, 0), (0, 0), (self.kernel_size - 1, 0)])
        # MLX Conv1d expects [batch, time, channels] (NLC format)
        x = mx.transpose(x, (0, 2, 1))  # NCL -> NLC
        x = self.conv(x)
        x = mx.transpose(x, (0, 2, 1))  # NLC -> NCL
        return x


class DecoderOutputSnake(nn.Module):
    """Output SnakeBeta layer - decoder.decoder[5].

    PyTorch key: decoder.decoder.5.{alpha, beta}
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))
        self.eps = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha).reshape(1, -1, 1)
        beta = mx.exp(self.beta).reshape(1, -1, 1)
        return x + (1.0 / (beta + self.eps)) * mx.power(mx.sin(x * alpha), 2)


class DecoderOutputConv(nn.Module):
    """Output conv layer - decoder.decoder[6].

    PyTorch key: decoder.decoder.6.conv.{weight, bias}
    """

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(channels, 1, kernel_size, padding=0)
        self.kernel_size = kernel_size

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, channels, time] (NCL format)
        # Causal padding (left pad on time dimension)
        x = mx.pad(x, [(0, 0), (0, 0), (self.kernel_size - 1, 0)])
        # MLX Conv1d expects [batch, time, channels] (NLC format)
        x = mx.transpose(x, (0, 2, 1))  # NCL -> NLC
        x = self.conv(x)
        x = mx.transpose(x, (0, 2, 1))  # NLC -> NCL
        return x


class Qwen3TTSSpeechTokenizerDecoder(nn.Module):
    """Full decoder for speech tokenizer."""

    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig):
        super().__init__()
        self.config = config
        self.total_upsample = int(
            np.prod(config.upsample_rates + config.upsampling_ratios)
        )

        # Transformer
        self.pre_transformer = DecoderTransformer(config)

        # Quantizer
        self.quantizer = SplitResidualVectorQuantizer(
            dimension=config.codebook_dim // 2,
            n_q=config.num_quantizers,
            n_q_semantic=config.num_semantic_quantizers,
            bins=config.codebook_size,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
        )

        # Pre-conv
        self.pre_conv = CausalConv1d(
            config.codebook_dim, config.latent_dim, kernel_size=3
        )

        # Upsampling blocks (matches PyTorch's self.upsample ModuleList)
        # Each upsample is a list: [CausalTransposeConv, ConvNeXtBlock]
        # PyTorch key: decoder.upsample.{idx}.{0|1}.{attr}
        self.upsample = [
            [
                CausalTransposeConv1d(
                    config.latent_dim, config.latent_dim, factor, factor
                ),
                ConvNeXtBlock(config.latent_dim),
            ]
            for factor in config.upsampling_ratios
        ]

        # Main decoder (matches PyTorch's self.decoder ModuleList)
        # [0]: Initial conv, [1:5]: 4 DecoderBlocks, [5]: Output snake, [6]: Output conv
        # PyTorch key: decoder.decoder.{idx}.{attr}
        output_dim = config.decoder_dim // (2 ** len(config.upsample_rates))
        self.decoder = [
            DecoderInitialConv(config.latent_dim, config.decoder_dim, 7),  # [0]
            *[
                DecoderBlock(config, i) for i in range(len(config.upsample_rates))
            ],  # [1-4]
            DecoderOutputSnake(output_dim),  # [5]
            DecoderOutputConv(output_dim, 7),  # [6]
        ]

    def __call__(self, codes: mx.array) -> mx.array:
        """
        Args:
            codes: [batch, num_quantizers, time]

        Returns:
            audio: [batch, 1, samples]
        """
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} layers of codes, got {codes.shape[1]}"
            )

        # Dequantize
        hidden = self.quantizer.decode(codes)  # [batch, codebook_dim, time]

        # Pre-conv and transpose for transformer
        hidden = self.pre_conv(hidden)  # [batch, latent_dim, time]
        hidden = mx.transpose(hidden, (0, 2, 1))  # [batch, time, latent_dim]

        # Transformer (no caching needed for decoder-only inference)
        hidden = self.pre_transformer(hidden)

        # Back to conv format
        hidden = mx.transpose(hidden, (0, 2, 1))  # [batch, latent_dim, time]

        # Upsampling
        for upsample_layers in self.upsample:
            for layer in upsample_layers:
                hidden = layer(hidden)

        # Decoder
        wav = hidden
        for decoder_layer in self.decoder:
            wav = decoder_layer(wav)

        return mx.clip(wav, -1.0, 1.0)

    def chunked_decode(
        self,
        codes: mx.array,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> mx.array:
        """Decode in chunks to handle long sequences with reduced memory usage.

        Args:
            codes: Input codes [batch, num_quantizers, time]
            chunk_size: Number of tokens per chunk (reduced from 300 to 150 for lower memory)
            left_context_size: Context from previous chunk for continuity

        Returns:
            Decoded audio waveform
        """
        wavs = []
        start_index = 0

        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = (
                left_context_size
                if start_index - left_context_size > 0
                else start_index
            )
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wav_chunk = wav_chunk[..., context_size * self.total_upsample :]

            # Evaluate and store chunk immediately to free intermediate tensors
            mx.eval(wav_chunk)
            wavs.append(wav_chunk)

            # Clear cache after each chunk to reduce peak memory
            mx.clear_cache()

            start_index = end_index

        return mx.concatenate(wavs, axis=-1)


class Qwen3TTSSpeechTokenizerEncoder(nn.Module):
    """Encoder for the speech tokenizer using Mimi components.

    Architecture: SeanetEncoder -> ProjectedTransformer -> ConvDownsample1d -> SplitRVQ
    """

    def __init__(self, config: Qwen3TTSTokenizerEncoderConfig):
        super().__init__()
        self.config = config
        self.valid_num_quantizers = 16  # Only first 16 quantizers are used for ICL

        # Build SeanetConfig from encoder config
        seanet_cfg = SeanetConfig(
            dimension=config.hidden_size,
            channels=config.audio_channels,
            causal=config.use_causal_conv,
            nfilters=config.num_filters,
            nresidual_layers=config.num_residual_layers,
            ratios=config.upsampling_ratios,
            ksize=config.kernel_size,
            residual_ksize=config.residual_kernel_size,
            last_ksize=config.last_kernel_size,
            dilation_base=config.dilation_growth_rate,
            pad_mode="constant",
            true_skip=not config.use_conv_shortcut,
            compress=config.compress,
        )
        self.encoder = SeanetEncoder(seanet_cfg)

        # Build TransformerConfig
        transformer_cfg = TransformerConfig(
            d_model=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            causal=config.use_causal_conv,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=config.layer_scale_initial_scale,
            positional_embedding="rope",
            use_conv_block=False,
            cross_attention=False,
            conv_kernel_size=3,
            use_conv_bias=True,
            gating=False,
            norm="layer_norm",
            context=config.sliding_window,
            max_period=int(config.rope_theta),
            max_seq_len=config.max_position_embeddings,
            kv_repeat=config.num_attention_heads // config.num_key_value_heads,
            dim_feedforward=config.intermediate_size,
            conv_layout=True,
            rope_traditional=False,
        )
        self.encoder_transformer = ProjectedTransformer(
            transformer_cfg,
            input_dim=config.hidden_size,
            output_dims=[config.hidden_size],
        )

        # ConvDownsample1d: stride = encoder_frame_rate / frame_rate
        encoder_frame_rate = config.sampling_rate / math.prod(config.upsampling_ratios)
        downsample_stride = int(encoder_frame_rate / config.frame_rate)
        self.downsample = ConvDownsample1d(
            stride=downsample_stride,
            dim=config.hidden_size,
            causal=config.use_causal_conv,
        )

        # SplitResidualVectorQuantizer
        self.quantizer = MimiSplitRVQ(
            dim=config.codebook_dim,
            input_dim=config.hidden_size,
            output_dim=config.hidden_size,
            nq=config.num_quantizers,
            bins=config.codebook_size,
        )

        self.encoder_cache = self.encoder_transformer.make_cache()

    def encode(self, audio: mx.array) -> mx.array:
        """Encode audio waveform to codes.

        Args:
            audio: [batch, 1, samples] audio waveform

        Returns:
            codes: [batch, num_quantizers, time]
        """
        self.encoder.reset_state()
        for c in self.encoder_cache:
            _reset_kv_cache(c)
        xs = self.encoder(audio)
        # Create causal attention mask (the model was trained with causal attention)
        seq_len = xs.shape[-1]  # NCL format, time is last dim
        mask = mx.full((seq_len, seq_len), -mx.inf, dtype=xs.dtype)
        mask = mx.triu(mask, k=1)  # Upper triangle = -inf, lower triangle + diag = 0
        mask = mask[None, None, :, :]  # [1, 1, seq_len, seq_len]
        xs = self.encoder_transformer(xs, cache=self.encoder_cache, mask=mask)[0]
        xs = self.downsample(xs)
        codes = self.quantizer.encode(xs)
        return codes[:, : self.valid_num_quantizers, :]


class Qwen3TTSSpeechTokenizer(nn.Module):
    """Full speech tokenizer model."""

    def __init__(self, config: Qwen3TTSTokenizerConfig):
        super().__init__()
        self.config = config
        self.encoder_valid_num_quantizers = config.encoder_valid_num_quantizers
        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate
        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        # Decoder
        self.decoder = Qwen3TTSSpeechTokenizerDecoder(config.decoder_config)

        # Encoder (for ICL voice cloning)
        if config.encoder_config is not None:
            self.encoder_model = Qwen3TTSSpeechTokenizerEncoder(config.encoder_config)
        else:
            self.encoder_model = None

    def encode(self, audio: mx.array) -> mx.array:
        """Encode audio waveform to codes.

        Args:
            audio: [batch, 1, samples] audio waveform

        Returns:
            codes: [batch, num_quantizers, time]
        """
        if self.encoder_model is None:
            raise ValueError("Encoder not available for this speech tokenizer")
        return self.encoder_model.encode(audio)

    @property
    def has_encoder(self) -> bool:
        return self.encoder_model is not None

    def decode(self, audio_codes: mx.array) -> mx.array:
        """
        Decode audio codes to waveform.

        Args:
            audio_codes: [batch, time, num_quantizers]

        Returns:
            audio: [batch, samples]
            audio_lengths: [batch] valid sample lengths
        """
        # Transpose to [batch, num_quantizers, time]
        codes = mx.transpose(audio_codes, (0, 2, 1))
        wav = self.decoder.chunked_decode(codes).squeeze(1)

        # Trim based on valid lengths
        audio_lengths = (audio_codes[..., 0] > 0).sum(
            axis=1
        ) * self.decode_upsample_rate

        return wav, audio_lengths

    def streaming_decode(self, audio_codes: mx.array, chunk_tokens: int = 50):
        """
        Decode audio codes to waveform in a streaming fashion.

        Yields audio chunks as they're decoded, reducing peak memory usage.

        Args:
            audio_codes: [batch, time, num_quantizers]
            chunk_tokens: Number of tokens to decode per chunk

        Yields:
            audio_chunk: [batch, samples] decoded audio for each chunk
        """
        # Transpose to [batch, num_quantizers, time]
        codes = mx.transpose(audio_codes, (0, 2, 1))
        total_tokens = codes.shape[-1]
        left_context_size = 25

        start_index = 0
        while start_index < total_tokens:
            end_index = min(start_index + chunk_tokens, total_tokens)
            context_size = (
                left_context_size
                if start_index - left_context_size > 0
                else start_index
            )
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self.decoder(codes_chunk)
            wav_chunk = wav_chunk[..., context_size * self.decoder.total_upsample :]
            wav_chunk = wav_chunk.squeeze(1)

            # Evaluate immediately to free computation graph
            mx.eval(wav_chunk)

            yield wav_chunk

            # Clear cache after each chunk
            mx.clear_cache()

            start_index = end_index

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights from PyTorch to MLX format."""
        from .qwen3_tts import check_array_shape_qwen3

        sanitized = {}

        # Collect decoder codebook weights to compute embeddings
        codebook_data = {}

        # Collect encoder transformer q/k/v for in_proj concatenation
        encoder_transformer_qkv: Dict[int, Dict[str, mx.array]] = {}

        # Collect encoder codebook data (cluster_usage + embedding_sum)
        encoder_codebook_data: Dict[str, Dict[str, mx.array]] = {}

        # SeanetEncoder layer mapping:
        # N=0: init_conv, N=3,6,9,12: downsample, N=14: final_conv
        # N=1,4,7,10: residual blocks (with .block.1 and .block.3 sub-keys)
        seanet_conv_map = {
            0: "encoder_model.encoder.init_conv1d",
            3: "encoder_model.encoder.layers.0.downsample",
            6: "encoder_model.encoder.layers.1.downsample",
            9: "encoder_model.encoder.layers.2.downsample",
            12: "encoder_model.encoder.layers.3.downsample",
            14: "encoder_model.encoder.final_conv1d",
        }
        # Residual block layers: N -> encoder layer index
        seanet_residual_map = {1: 0, 4: 1, 7: 2, 10: 3}
        # Block index -> residual conv index
        seanet_block_map = {1: 0, 3: 1}

        for k, v in weights.items():
            if k.startswith("encoder."):
                # --- Handle encoder weights ---

                # SeanetEncoder convolutions
                if k.startswith("encoder.encoder.layers."):
                    parts = k.split(".")
                    n = int(parts[3])  # encoder.encoder.layers.{N}...

                    if "block" in k:
                        # Residual block: encoder.encoder.layers.{N}.block.{B}.conv.{w/b}
                        if n not in seanet_residual_map:
                            continue
                        layer_idx = seanet_residual_map[n]
                        block_idx = int(parts[5])  # .block.{B}
                        if block_idx not in seanet_block_map:
                            continue
                        conv_idx = seanet_block_map[block_idx]
                        base_path = f"encoder_model.encoder.layers.{layer_idx}.residuals.0.block.{conv_idx}"
                        suffix = ".".join(parts[6:])  # conv.weight or conv.bias
                    else:
                        # Direct conv: encoder.encoder.layers.{N}.conv.{w/b}
                        if n not in seanet_conv_map:
                            continue
                        base_path = seanet_conv_map[n]
                        suffix = ".".join(parts[4:])  # conv.weight or conv.bias

                    new_key = f"{base_path}.conv.{suffix}"
                    # Transpose conv weights: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
                    if "weight" in suffix and len(v.shape) == 3:
                        v = v.swapaxes(-1, -2)
                    sanitized[new_key] = v

                # Encoder transformer layers
                elif k.startswith("encoder.encoder_transformer.layers."):
                    # encoder.encoder_transformer.layers.{i}.self_attn.q_proj.weight
                    parts = k.split(".")
                    layer_idx = int(
                        parts[3]
                    )  # encoder.encoder_transformer.layers.{i}...
                    rest = ".".join(parts[4:])

                    # Collect q/k/v for in_proj concatenation
                    if "self_attn.q_proj.weight" in rest:
                        if layer_idx not in encoder_transformer_qkv:
                            encoder_transformer_qkv[layer_idx] = {}
                        encoder_transformer_qkv[layer_idx]["q"] = v
                    elif "self_attn.k_proj.weight" in rest:
                        if layer_idx not in encoder_transformer_qkv:
                            encoder_transformer_qkv[layer_idx] = {}
                        encoder_transformer_qkv[layer_idx]["k"] = v
                    elif "self_attn.v_proj.weight" in rest:
                        if layer_idx not in encoder_transformer_qkv:
                            encoder_transformer_qkv[layer_idx] = {}
                        encoder_transformer_qkv[layer_idx]["v"] = v
                    elif "self_attn.o_proj.weight" in rest:
                        new_key = f"encoder_model.encoder_transformer.transformer.layers.{layer_idx}.self_attn.out_proj.weight"
                        sanitized[new_key] = v
                    elif "mlp.fc1.weight" in rest:
                        new_key = f"encoder_model.encoder_transformer.transformer.layers.{layer_idx}.gating.linear1.weight"
                        sanitized[new_key] = v
                    elif "mlp.fc2.weight" in rest:
                        new_key = f"encoder_model.encoder_transformer.transformer.layers.{layer_idx}.gating.linear2.weight"
                        sanitized[new_key] = v
                    elif "input_layernorm.weight" in rest:
                        new_key = f"encoder_model.encoder_transformer.transformer.layers.{layer_idx}.norm1.weight"
                        sanitized[new_key] = v
                    elif "input_layernorm.bias" in rest:
                        new_key = f"encoder_model.encoder_transformer.transformer.layers.{layer_idx}.norm1.bias"
                        sanitized[new_key] = v
                    elif "post_attention_layernorm.weight" in rest:
                        new_key = f"encoder_model.encoder_transformer.transformer.layers.{layer_idx}.norm2.weight"
                        sanitized[new_key] = v
                    elif "post_attention_layernorm.bias" in rest:
                        new_key = f"encoder_model.encoder_transformer.transformer.layers.{layer_idx}.norm2.bias"
                        sanitized[new_key] = v
                    elif "self_attn_layer_scale.scale" in rest:
                        new_key = f"encoder_model.encoder_transformer.transformer.layers.{layer_idx}.layer_scale_1.scale"
                        sanitized[new_key] = v
                    elif "mlp_layer_scale.scale" in rest:
                        new_key = f"encoder_model.encoder_transformer.transformer.layers.{layer_idx}.layer_scale_2.scale"
                        sanitized[new_key] = v

                # Encoder downsample conv
                elif k.startswith("encoder.downsample."):
                    suffix = k.replace("encoder.downsample.", "")
                    new_key = f"encoder_model.downsample.conv.conv.{suffix}"
                    if "weight" in suffix and len(v.shape) == 3:
                        v = v.swapaxes(-1, -2)
                    sanitized[new_key] = v

                # Encoder quantizer
                elif k.startswith("encoder.quantizer."):
                    rest = k.replace("encoder.quantizer.", "")

                    # Codebook data (cluster_usage / embed_sum)
                    # HF format: .layers.{i}.codebook.{cluster_usage,embed_sum,initialized}
                    if (
                        ".codebook.cluster_usage" in rest
                        or ".codebook.embed_sum" in rest
                    ):
                        # Extract base path for grouping
                        base = rest.rsplit(".codebook.", 1)[0]
                        if base not in encoder_codebook_data:
                            encoder_codebook_data[base] = {}
                        if "cluster_usage" in rest:
                            encoder_codebook_data[base]["cluster_usage"] = v
                        elif "embed_sum" in rest:
                            encoder_codebook_data[base]["embedding_sum"] = v
                    elif ".codebook.initialized" in rest:
                        pass  # Skip initialized flag
                    # Input/output projections
                    elif "input_proj.weight" in rest or "output_proj.weight" in rest:
                        if "semantic_residual_vector_quantizer" in rest:
                            proj_type = (
                                "input_proj" if "input_proj" in rest else "output_proj"
                            )
                            new_key = (
                                f"encoder_model.quantizer.rvq_first.{proj_type}.weight"
                            )
                        else:
                            proj_type = (
                                "input_proj" if "input_proj" in rest else "output_proj"
                            )
                            new_key = (
                                f"encoder_model.quantizer.rvq_rest.{proj_type}.weight"
                            )
                        # Conv1d weight: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
                        if len(v.shape) == 3:
                            v = v.swapaxes(-1, -2)
                        sanitized[new_key] = v

            else:
                # --- Handle decoder weights (existing logic) ---

                # Collect codebook cluster_usage and embedding_sum for later processing
                if "_codebook.cluster_usage" in k or "_codebook.embedding_sum" in k:
                    base_path = k.rsplit("._codebook.", 1)[0]
                    if base_path not in codebook_data:
                        codebook_data[base_path] = {}
                    if "cluster_usage" in k:
                        codebook_data[base_path]["cluster_usage"] = v
                    else:
                        codebook_data[base_path]["embedding_sum"] = v
                    continue

                new_key = k

                # Identify ConvTranspose1d weights (for upsampling layers)
                is_transpose_conv = ("upsample" in k and ".0.conv.weight" in k) or (
                    "decoder.decoder" in k and "block.1.conv.weight" in k
                )

                if is_transpose_conv and len(v.shape) == 3:
                    v = v if check_array_shape_qwen3(v) else mx.transpose(v, (1, 2, 0))
                elif "conv.weight" in k and len(v.shape) == 3:
                    v = v if check_array_shape_qwen3(v) else mx.transpose(v, (0, 2, 1))
                elif "_proj.weight" in k and len(v.shape) == 3:
                    v = v if check_array_shape_qwen3(v) else mx.transpose(v, (0, 2, 1))

                sanitized[new_key] = v

        # Process encoder transformer q/k/v into combined in_proj weights
        for layer_idx, qkv in encoder_transformer_qkv.items():
            if "q" in qkv and "k" in qkv and "v" in qkv:
                in_proj_weight = mx.concatenate([qkv["q"], qkv["k"], qkv["v"]], axis=0)
                new_key = f"encoder_model.encoder_transformer.transformer.layers.{layer_idx}.self_attn.in_proj.weight"
                sanitized[new_key] = in_proj_weight

        # Process encoder codebook data into embedding_sum and cluster_usage
        for base_path, data in encoder_codebook_data.items():
            if "cluster_usage" in data and "embedding_sum" in data:
                # Map HF quantizer paths to Mimi paths
                if "semantic_residual_vector_quantizer" in base_path:
                    m = re.search(r"layers\.(\d+)", base_path)
                    if m:
                        layer_idx = int(m.group(1))
                        prefix = f"encoder_model.quantizer.rvq_first.vq.layers.{layer_idx}.codebook"
                        sanitized[f"{prefix}.embedding_sum"] = data["embedding_sum"]
                        sanitized[f"{prefix}.cluster_usage"] = data["cluster_usage"]
                elif "acoustic_residual_vector_quantizer" in base_path:
                    m = re.search(r"layers\.(\d+)", base_path)
                    if m:
                        layer_idx = int(m.group(1))
                        prefix = f"encoder_model.quantizer.rvq_rest.vq.layers.{layer_idx}.codebook"
                        sanitized[f"{prefix}.embedding_sum"] = data["embedding_sum"]
                        sanitized[f"{prefix}.cluster_usage"] = data["cluster_usage"]

        # Compute decoder embeddings from cluster_usage and embedding_sum
        eps = 1e-5
        for base_path, data in codebook_data.items():
            if "cluster_usage" in data and "embedding_sum" in data:
                cluster_usage = data["cluster_usage"]
                embedding_sum = data["embedding_sum"]
                embedding = embedding_sum / mx.clip(cluster_usage[:, None], eps, None)
                new_key = f"{base_path}.codebook.embed.weight"
                sanitized[new_key] = embedding

        return sanitized
