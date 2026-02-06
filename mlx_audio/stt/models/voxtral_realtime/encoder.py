"""Causal audio encoder for Voxtral Realtime.

32-layer causal transformer with:
- Causal conv1d stem (128 -> 1280, stride 1; 1280 -> 1280, stride 2)
- Interleaved RoPE (theta=1M)
- Sliding window attention (750)
- SwiGLU FFN
- Selective biases (wq/wv/wo yes, wk no; w2 only in FFN)
- 4x downsample + adapter MLP
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .config import EncoderConfig


def _interleaved_rope(x, cos, sin, n_heads, head_dim):
    """Apply interleaved (GPT-J style) RoPE.

    Rotates consecutive pairs: (x[0], x[1]), (x[2], x[3]), ...
    x: [seq, n_heads * head_dim]
    cos, sin: [seq, head_dim // 2]
    """
    seq_len = x.shape[0]
    x = x.reshape(seq_len, n_heads, head_dim)
    x1 = x[..., ::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    cos = cos[:, None, :]  # [seq, 1, hd/2]
    sin = sin[:, None, :]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    # Interleave back
    out = mx.zeros_like(x)
    out[..., ::2] = o1
    out[..., 1::2] = o2
    return out.reshape(seq_len, n_heads * head_dim)


def _compute_rope_freqs(positions, head_dim, theta):
    """Compute cos/sin frequencies for RoPE.

    positions: [seq_len] int array
    Returns: (cos, sin) each [seq_len, head_dim // 2]
    """
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    angles = positions[:, None].astype(mx.float32) * freqs[None, :]
    return mx.cos(angles), mx.sin(angles)


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-only padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size - stride  # left-only padding
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=True
        )

    def __call__(self, x):
        # x: [batch, seq, channels] (MLX conv1d expects NLC)
        # Left-pad only
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
        return self.conv(x)


class EncoderAttention(nn.Module):
    """Multi-head attention for encoder with selective biases."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.rope_theta = config.rope_theta
        attn_dim = config.n_heads * config.head_dim

        # Selective biases: wq, wv, wo have bias; wk does NOT
        self.wq = nn.Linear(config.dim, attn_dim, bias=True)
        self.wk = nn.Linear(config.dim, attn_dim, bias=False)
        self.wv = nn.Linear(config.dim, attn_dim, bias=True)
        self.wo = nn.Linear(attn_dim, config.dim, bias=True)

    def __call__(self, x, positions):
        seq_len = x.shape[0]
        q = self.wq(x)  # [seq, n_heads * head_dim]
        k = self.wk(x)
        v = self.wv(x)

        # RoPE
        cos, sin = _compute_rope_freqs(positions, self.head_dim, self.rope_theta)
        q = _interleaved_rope(q, cos, sin, self.n_heads, self.head_dim)
        k = _interleaved_rope(k, cos, sin, self.n_heads, self.head_dim)

        # Reshape for attention: [1, n_heads, seq, head_dim] (MLX expects B,H,T,D)
        q = q.reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Build causal sliding window mask [1, 1, seq_q, seq_k]
        # MLX scaled_dot_product_attention: additive mask (0 = attend, -inf = block)
        qi = mx.arange(seq_len)[:, None]
        ki = mx.arange(seq_len)[None, :]
        causal = ki <= qi
        window = ki >= (qi - self.sliding_window + 1)
        mask = mx.where(causal & window, mx.array(0.0), mx.array(-1e9))

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask
        )

        # Reshape back: [1, n_heads, seq, head_dim] -> [seq, n_heads * head_dim]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(seq_len, self.n_heads * self.head_dim)
        return self.wo(attn_out)


class EncoderLayer(nn.Module):
    """Single encoder transformer layer."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = EncoderAttention(config)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        # SwiGLU FFN: w1=gate (no bias), w3=up (no bias), w2=down (bias)
        self.feed_forward_w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.feed_forward_w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.feed_forward_w2 = nn.Linear(config.hidden_dim, config.dim, bias=True)

    def __call__(self, x, positions):
        # Attention
        h = self.attention_norm(x)
        h = self.attention(h, positions)
        x = x + h

        # SwiGLU FFN
        h = self.ffn_norm(x)
        gate = nn.silu(self.feed_forward_w1(h))
        up = self.feed_forward_w3(h)
        x = x + self.feed_forward_w2(gate * up)

        return x


class AudioEncoder(nn.Module):
    """Full causal audio encoder: conv stem + transformer + downsample + adapter."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Conv stem
        self.conv_layers_0_conv = CausalConv1d(128, config.dim, kernel_size=3, stride=1)
        self.conv_layers_1_conv = CausalConv1d(config.dim, config.dim, kernel_size=3, stride=2)

        # Transformer layers
        self.transformer_layers = [EncoderLayer(config) for _ in range(config.n_layers)]
        self.transformer_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        # Adapter MLP: Linear(dim*4 -> decoder_dim) -> GELU -> Linear(decoder_dim -> decoder_dim)
        # Note: adapter weights are loaded separately via sanitize, dims set by weight shapes
        adapter_input_dim = config.dim * config.downsample_factor  # 5120
        # Decoder dim will be set from weights; default to 3072
        decoder_dim = 3072
        self.audio_language_projection_0 = nn.Linear(
            adapter_input_dim, decoder_dim, bias=False
        )
        self.audio_language_projection_2 = nn.Linear(
            decoder_dim, decoder_dim, bias=False
        )

    def __call__(self, mel):
        """
        Args:
            mel: [mel_bins, frames] log-mel spectrogram

        Returns:
            mx.array: [seq/4, decoder_dim] adapter output
        """
        # mel is [128, frames], transpose to [frames, 128] for conv (NLC format)
        x = mel.T  # [frames, 128]
        x = x[None, :, :]  # [1, frames, 128]

        # Conv stem
        x = nn.gelu(self.conv_layers_0_conv(x))  # [1, frames, 1280]
        x = nn.gelu(self.conv_layers_1_conv(x))  # [1, frames/2, 1280]

        x = x.squeeze(0)  # [seq, 1280]
        seq_len = x.shape[0]

        # Left-truncate to multiple of downsample_factor
        trunc = seq_len % self.config.downsample_factor
        if trunc > 0:
            x = x[trunc:]
            seq_len = x.shape[0]

        # Transformer layers
        positions = mx.arange(seq_len)
        for layer in self.transformer_layers:
            x = layer(x, positions)

        # Final norm
        x = self.transformer_norm(x)

        # 4x downsample: [seq, 1280] -> [seq/4, 5120]
        ds_len = seq_len // self.config.downsample_factor
        x = x.reshape(ds_len, self.config.dim * self.config.downsample_factor)

        # Adapter MLP
        x = nn.gelu(self.audio_language_projection_0(x))
        x = self.audio_language_projection_2(x)

        return x  # [seq/4, decoder_dim]
