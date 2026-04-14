"""Voxtral TTS Audio Tokenizer — decodes quantized codes to 24kHz waveform.

Decoder: alternating weight-normed conv blocks and ALiBi transformer blocks.
Strides [1,2,2,2] give 8x upsampling; output_proj maps dim -> 240 samples per frame.
"""

import math
from dataclasses import dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn

from .common import FeedForward


@dataclass
class AudioTokenizerArgs:
    channels: int = 1
    sampling_rate: int = 24000
    pretransform_patch_size: int = 240
    patch_proj_kernel_size: int = 7
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36
    conv_weight_norm: bool = True
    causal: bool = True
    attn_sliding_window_size: int = 16
    half_attn_window_upon_downsampling: bool = True
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    qk_norm_eps: float = 1e-06
    qk_norm: bool = True
    use_biases: bool = False
    norm_eps: float = 0.01
    layer_scale: bool = True
    layer_scale_init: float = 0.01
    decoder_transformer_lengths_str: str = "2,2,2,2"
    decoder_convs_kernels_str: str = "3,4,4,4"
    decoder_convs_strides_str: str = "1,2,2,2"

    @property
    def decoder_transformer_lengths(self) -> List[int]:
        return [int(x) for x in self.decoder_transformer_lengths_str.split(",")]

    @property
    def decoder_convs_kernels(self) -> List[int]:
        return [int(x) for x in self.decoder_convs_kernels_str.split(",")]

    @property
    def decoder_convs_strides(self) -> List[int]:
        return [int(x) for x in self.decoder_convs_strides_str.split(",")]

    @property
    def codebook_dim(self) -> int:
        return self.semantic_dim + self.acoustic_dim


# ============================================================================
# Weight-normalized Causal Conv1d
# ============================================================================


class WeightNormConv(nn.Module):
    """Causal Conv1d with weight normalization stored as parametrizations.

    Checkpoint stores:
      conv.parametrizations.weight.original0: (out_ch, 1, 1) -- the gain (g)
      conv.parametrizations.weight.original1: (out_ch, in_ch, K) -- the direction (v)

    Effective weight = g * v / ||v|| applied per output channel.
    """

    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        kernel_size: int,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.parametrizations = {
            "weight": {
                "original0": mx.ones((out_channels, 1, 1)),
                "original1": mx.zeros((out_channels, in_channels, kernel_size)),
            }
        }
        self.pad_mode = pad_mode

    def _get_weight(self) -> mx.array:
        """Reconstruct weight from weight norm parametrization."""
        # Access nested structure
        g = self.parametrizations["weight"]["original0"]  # (out, 1, 1)
        v = self.parametrizations["weight"]["original1"]  # (out, in, K)

        # Weight norm: w = g * v / ||v||
        v_norm = mx.sqrt((v * v).sum(axis=(1, 2), keepdims=True) + 1e-12)
        return g * v / v_norm

    def __call__(
        self, x: mx.array, stride: int = 1, transpose: bool = False
    ) -> mx.array:
        """Apply causal convolution.

        Args:
            x: (batch, seq_len, channels) - MLX channels-last
            stride: convolution stride
            transpose: if True, use transposed convolution (upsampling)
        """
        # weight: (out_ch, in_ch, K) from checkpoint
        weight = self._get_weight()
        if transpose:
            return self._conv_transpose_1d(x, weight, stride)
        return self._conv_1d(x, weight, stride)

    def _pad_1d(
        self,
        x: mx.array,
        padding_left: int,
        padding_right: int,
    ) -> mx.array:
        if padding_left == 0 and padding_right == 0:
            return x

        if self.pad_mode == "constant":
            return mx.pad(x, [(0, 0), (padding_left, padding_right), (0, 0)])

        if self.pad_mode == "replicate":
            return mx.pad(
                x, [(0, 0), (padding_left, padding_right), (0, 0)], mode="edge"
            )

        if self.pad_mode != "reflect":
            raise ValueError(f"Unsupported pad_mode={self.pad_mode!r}")

        length = x.shape[1]
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = mx.pad(x, [(0, 0), (0, extra_pad), (0, 0)])

        parts = []
        if padding_left > 0:
            parts.append(x[:, 1 : padding_left + 1, :][:, ::-1, :])
        parts.append(x)
        if padding_right > 0:
            parts.append(x[:, -padding_right - 1 : -1, :][:, ::-1, :])

        padded = mx.concatenate(parts, axis=1)
        if extra_pad > 0:
            padded = padded[:, : padded.shape[1] - extra_pad, :]
        return padded

    def _conv_1d(self, x: mx.array, weight: mx.array, stride: int) -> mx.array:
        kernel_size = weight.shape[2]
        padding_total = kernel_size - stride
        n_frames = (x.shape[1] - kernel_size + padding_total) / stride + 1
        target_length = (math.ceil(n_frames) - 1) * stride + (
            kernel_size - padding_total
        )
        extra_padding = target_length - x.shape[1]
        x = self._pad_1d(x, padding_total, extra_padding)
        # mx.conv1d weight: (out_ch, K, in_ch)
        w = weight.transpose(0, 2, 1)
        return mx.conv1d(x, w, stride=stride)

    def _conv_transpose_1d(
        self, x: mx.array, weight: mx.array, stride: int
    ) -> mx.array:
        """Causal transposed 1D convolution using native mx.conv_transpose1d."""
        out_ch, in_ch, K = weight.shape
        B, T, C = x.shape

        # mx.conv_transpose1d weight: (C_out, K, C_in)
        # Checkpoint weight: (out_ch, in_ch, K) — for ConvTranspose1d this is (C_in, C_out, K)
        # So we need: (C_out=in_ch, K, C_in=out_ch)
        w = weight.transpose(1, 2, 0)  # (in_ch, K, out_ch) = (C_out, K, C_in)

        # Causal: pad K-1 on the left via output trimming
        out = mx.conv_transpose1d(x, w, stride=stride, padding=0)
        # out shape: (B, (T-1)*stride + K, C_out)

        # Trim for causality: keep first T*stride elements
        out = out[:, : T * stride, :]

        return out


class ConvBlock(nn.Module):
    """A conv block matching decoder_blocks.{even_index}.conv"""

    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        kernel_size: int,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.conv = WeightNormConv(
            out_channels,
            in_channels,
            kernel_size,
            pad_mode=pad_mode,
        )


# ============================================================================
# Transformer components (with ALiBi)
# ============================================================================


def _get_alibi_slopes(n_heads: int) -> mx.array:
    def _get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = _get_slopes_power_of_2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = _get_slopes_power_of_2(closest)
        extra = _get_slopes_power_of_2(2 * closest)
        slopes += extra[0::2][: n_heads - closest]
    return mx.array(slopes, dtype=mx.float32)


class Attention(nn.Module):
    """Multi-head attention with ALiBi bias.

    Weight names: attention.{wq,wk,wv,wo,q_norm,k_norm}.weight
    """

    def __init__(self, args: AudioTokenizerArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        if args.qk_norm:
            # Checkpoint stores norm over full projected dim, not per-head
            self.q_norm = nn.RMSNorm(args.n_heads * args.head_dim, eps=args.qk_norm_eps)
            self.k_norm = nn.RMSNorm(
                args.n_kv_heads * args.head_dim, eps=args.qk_norm_eps
            )
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(
        self, x: mx.array, alibi_slopes: mx.array, window_size: int = 0
    ) -> mx.array:
        B, T, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.n_kv_heads < self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, repeat, axis=1)
            v = mx.repeat(v, repeat, axis=1)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # ALiBi bias: score[h,i,j] += slope[h] * (j - i), matching C reference
        # Negative for past positions (j < i), penalizing distant lookback
        positions = mx.arange(T)
        dist = positions[None, :] - positions[:, None]  # dist[i,j] = j - i
        alibi = alibi_slopes[:, None, None] * dist[None, :, :].astype(mx.float32)

        # Causal mask + sliding window
        causal_mask = mx.triu(mx.full((T, T), -1e9), k=1)
        if window_size > 0:
            # Mask positions outside window (j - i < -window_size)
            window_mask = mx.where(dist < -window_size, -1e9, 0.0)
            causal_mask = causal_mask + window_mask

        scores = scores + alibi + causal_mask[None, None, :, :]

        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(out)


class TokenizerFeedForward(FeedForward):
    """FeedForward initialized from AudioTokenizerArgs."""

    def __init__(self, args: AudioTokenizerArgs):
        super().__init__(dim=args.dim, hidden_dim=args.hidden_dim, bias=False)


class TransformerLayer(nn.Module):
    """Single transformer layer with layer scale.

    Weight names: layers.N.{attention_norm, attention, attention_scale, ffn_norm, feed_forward, ffn_scale}
    """

    def __init__(self, args: AudioTokenizerArgs):
        super().__init__()
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.attention = Attention(args)
        self.feed_forward = TokenizerFeedForward(args)

        if args.layer_scale:
            self.attention_scale = mx.full((args.dim,), args.layer_scale_init)
            self.ffn_scale = mx.full((args.dim,), args.layer_scale_init)
        self.use_layer_scale = args.layer_scale

    def __call__(
        self, x: mx.array, alibi_slopes: mx.array, window_size: int = 0
    ) -> mx.array:
        h = self.attention(
            self.attention_norm(x), alibi_slopes, window_size=window_size
        )
        if self.use_layer_scale:
            h = h * self.attention_scale
        x = x + h

        h = self.feed_forward(self.ffn_norm(x))
        if self.use_layer_scale:
            h = h * self.ffn_scale
        x = x + h
        return x


class TransformerBlock(nn.Module):
    """Block of N transformer layers.

    Weight names: layers.{0..N-1}.*
    """

    def __init__(self, n_layers: int, args: AudioTokenizerArgs):
        super().__init__()
        self.layers = [TransformerLayer(args) for _ in range(n_layers)]

    def __call__(
        self, x: mx.array, alibi_slopes: mx.array, window_size: int = 0
    ) -> mx.array:
        for layer in self.layers:
            x = layer(x, alibi_slopes, window_size=window_size)
        return x


# ============================================================================
# Codebooks
# ============================================================================


class SemanticCodebook(nn.Module):
    """EMA-based semantic codebook.

    Checkpoint stores:
      semantic_codebook.cluster_usage: (codebook_size,)
      semantic_codebook.embedding_sum: (codebook_size, dim)

    The actual codebook vectors are: embedding_sum / cluster_usage.unsqueeze(-1)
    """

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.cluster_usage = mx.ones((codebook_size,))
        self.embedding_sum = mx.zeros((codebook_size, dim))

    @property
    def codebook(self) -> mx.array:
        """Compute actual codebook vectors from EMA statistics (in float32 for precision)."""
        return self.embedding_sum.astype(mx.float32) / mx.maximum(
            self.cluster_usage.astype(mx.float32)[:, None], 1e-8
        )

    def decode(self, indices: mx.array) -> mx.array:
        """Look up codebook entries by index."""
        cb = self.codebook  # (codebook_size, dim)
        return cb[indices]


class AcousticCodebook(nn.Module):
    """Finite Scalar Quantization (FSQ) for acoustic tokens. No learned parameters."""

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim

    def decode(self, indices: mx.array) -> mx.array:
        """Convert FSQ indices [0, codebook_size-1] to continuous values [-1, 1]."""
        return (2.0 * indices.astype(mx.float32) / (self.codebook_size - 1)) - 1.0


class MistralAudioCodebook(nn.Module):
    """Combined semantic + acoustic codebook.

    Weight names: quantizer.semantic_codebook.{cluster_usage, embedding_sum}
    """

    def __init__(self, args: AudioTokenizerArgs):
        super().__init__()
        self.semantic_codebook = SemanticCodebook(
            args.semantic_codebook_size, args.semantic_dim
        )
        self.acoustic_codebook = AcousticCodebook(
            args.acoustic_codebook_size, args.acoustic_dim
        )

    def decode(self, codes: mx.array) -> mx.array:
        """Decode combined codes to continuous representations.

        Args:
            codes: (batch, seq_len, 1 + acoustic_dim) - codes with special token offsets
                   semantic codes in [2, 8193], acoustic codes in [2, 22]
        Returns:
            (batch, seq_len, semantic_dim + acoustic_dim) = (B, T, 292)
        """
        N_SPECIAL = 2
        # Strip special token offset before codebook lookup
        semantic_codes = codes[:, :, 0] - N_SPECIAL  # back to [0, 8191]
        acoustic_codes = codes[:, :, 1:] - N_SPECIAL  # back to [0, 20]

        semantic_emb = self.semantic_codebook.decode(semantic_codes)  # (B, T, 256)
        acoustic_emb = self.acoustic_codebook.decode(acoustic_codes)  # (B, T, 36)

        return mx.concatenate([semantic_emb, acoustic_emb], axis=-1)  # (B, T, 292)


class VoxtralTTSAudioTokenizer(nn.Module):
    """Complete audio tokenizer decoder.

    Architecture: alternating conv blocks and transformer blocks.
    decoder_blocks = [conv0, xformer0, conv1, xformer1, conv2, xformer2, conv3, xformer3]
    Followed by output_proj (another weight-normed conv).

    Strides: [1, 2, 2, 2] -> 8x total upsampling
    Then output_proj maps dim -> patch_size (240 samples per frame)
    Total: each input code -> 8 * 240 = 1920 samples at 24kHz = 80ms
    """

    def __init__(self, args: AudioTokenizerArgs):
        super().__init__()
        self.args = args

        self.quantizer = MistralAudioCodebook(args)

        # ALiBi slopes
        self._alibi_slopes = _get_alibi_slopes(args.n_heads)

        # Decoder blocks: alternating conv and transformer
        strides = args.decoder_convs_strides
        kernels = args.decoder_convs_kernels
        n_transformer_layers = args.decoder_transformer_lengths

        # First conv takes codebook_dim (292) as input, rest take model dim (1024)
        self.decoder_blocks = []
        for i, (stride, kernel, n_layers) in enumerate(
            zip(strides, kernels, n_transformer_layers)
        ):
            # Conv block (even index)
            in_ch = args.codebook_dim if i == 0 else args.dim
            self.decoder_blocks.append(
                ConvBlock(args.dim, in_ch, kernel, pad_mode="replicate")
            )
            # Transformer block (odd index)
            self.decoder_blocks.append(TransformerBlock(n_layers, args))

        # Output projection: dim -> patch_size
        self.output_proj = ConvBlock(
            args.pretransform_patch_size,
            args.dim,
            args.patch_proj_kernel_size,
            pad_mode="reflect",
        )

        # Store strides for the decode pass
        self._strides = strides

    def decode(self, codes: mx.array) -> mx.array:
        """Decode audio codes to waveform.

        Args:
            codes: (batch, seq_len, num_codebooks) - quantized codes [semantic + acoustic]
        Returns:
            (batch, num_samples) - 24kHz waveform
        """
        x = self.quantizer.decode(codes)  # (B, T, 292)

        # Sliding window sizes per decoder stage: [2, 4, 8, 16]
        # The encoder uses [16, 8, 4, 2] with half_attn_window_upon_downsampling
        # The decoder reverses this
        window_sizes = [2, 4, 8, 16]

        for i in range(0, len(self.decoder_blocks), 2):
            conv_block = self.decoder_blocks[i]
            xformer_block = self.decoder_blocks[i + 1]
            stage_idx = i // 2
            stride = self._strides[stage_idx]

            is_transpose = stride > 1
            x = conv_block.conv(x, stride=stride, transpose=is_transpose)

            window = window_sizes[stage_idx] if stage_idx < len(window_sizes) else 16
            x = xformer_block(x, self._alibi_slopes, window_size=window)

        # Output projection
        x = self.output_proj.conv(x, stride=1, transpose=False)  # (B, T_up, 240)

        # Reshape to waveform
        B = x.shape[0]
        return x.reshape(B, -1)
