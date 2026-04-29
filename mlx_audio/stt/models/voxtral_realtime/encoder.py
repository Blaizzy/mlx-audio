"""Causal audio encoder for Voxtral Realtime.

32-layer causal transformer with:
- Causal conv1d stem (128 -> 1280, stride 1; 1280 -> 1280, stride 2)
- Interleaved RoPE (theta=1M), fused via mx.fast.rope
- Sliding window attention (750)
- SwiGLU FFN
- Selective biases (wq/wv/wo yes, wk no; w2 only in FFN)
- 4x downsample + adapter MLP

Optimizations:
- RoPE applied via the fused mx.fast.rope kernel (traditional=True) instead of
  a Python interleave — matches the voxmlx path and avoids 32 layers' worth
  of manual reshape/stack overhead per step.
- Attention mask computed once per chunk and shared across all 32 layers.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .config import EncoderConfig


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

    def __call__(self, x, rope_offset, mask, cache=None):
        """
        Args:
            x: [seq, dim]
            rope_offset: absolute position of the first token in ``x``.
            mask: precomputed additive mask, or "causal" string.
            cache: optional RotatingKVCache for chunked encoding.
        """
        seq_len = x.shape[0]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape for attention: [1, n_heads, seq, head_dim]
        q = q.reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Fused RoPE (traditional=True = GPT-J style interleaved pairs).
        q = mx.fast.rope(
            q,
            self.head_dim,
            traditional=True,
            base=self.rope_theta,
            scale=1.0,
            offset=rope_offset,
        )
        k = mx.fast.rope(
            k,
            self.head_dim,
            traditional=True,
            base=self.rope_theta,
            scale=1.0,
            offset=rope_offset,
        )

        # Update KV cache if provided (for chunked encoding)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        # Reshape back: [1, n_heads, seq, head_dim] -> [seq, n_heads * head_dim]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
            seq_len, self.n_heads * self.head_dim
        )
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

    def __call__(self, x, rope_offset, mask, cache=None):
        # Attention
        h = self.attention_norm(x)
        h = self.attention(h, rope_offset, mask, cache=cache)
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
        self.conv_layers_1_conv = CausalConv1d(
            config.dim, config.dim, kernel_size=3, stride=2
        )

        # Transformer layers
        self.transformer_layers = [EncoderLayer(config) for _ in range(config.n_layers)]
        self.transformer_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        # Adapter MLP
        adapter_input_dim = config.dim * config.downsample_factor  # 5120
        decoder_dim = 3072
        self.audio_language_projection_0 = nn.Linear(
            adapter_input_dim, decoder_dim, bias=False
        )
        self.audio_language_projection_2 = nn.Linear(
            decoder_dim, decoder_dim, bias=False
        )

    def conv_stem(self, mel):
        """Run conv layers and align to downsample_factor.

        Args:
            mel: [mel_bins, frames] log-mel spectrogram

        Returns:
            mx.array: [seq, dim] conv output ready for transformer layers
        """
        x = mel.T[None, :, :]  # [1, frames, 128]
        x = nn.gelu(self.conv_layers_0_conv(x))
        x = nn.gelu(self.conv_layers_1_conv(x))
        x = x.squeeze(0)  # [seq, 1280]

        trunc = x.shape[0] % self.config.downsample_factor
        if trunc > 0:
            x = x[trunc:]
        return x

    def encode_chunks(self, conv_out):
        """Generator that encodes conv output in sliding-window-sized chunks.

        Processes through all transformer layers with KV caching, yielding
        layer-normed output per chunk. Each chunk is ready for downsample
        and projection.

        Args:
            conv_out: [seq, dim] output from conv_stem()

        Yields:
            mx.array: [chunk_size, dim] encoded chunk
        """
        from mlx_lm.models.cache import RotatingKVCache

        seq_len = conv_out.shape[0]
        sw = self.config.sliding_window
        n_layers = len(self.transformer_layers)
        caches = [RotatingKVCache(max_size=sw, keep=0) for _ in range(n_layers)]

        for chunk_start in range(0, seq_len, sw):
            chunk_end = min(chunk_start + sw, seq_len)
            x = conv_out[chunk_start:chunk_end]
            chunk_len = x.shape[0]

            # The mask depends only on chunk_len + cache offset, not on the
            # layer weights — compute it once and reuse across all 32 layers.
            mask = caches[0].make_mask(chunk_len, window_size=sw)
            for i, layer in enumerate(self.transformer_layers):
                x = layer(x, chunk_start, mask, cache=caches[i])

            yield self.transformer_norm(x)

    def downsample_and_project(self, encoded):
        """4x downsample encoder output and project to decoder dim.

        Args:
            encoded: [seq, dim] encoder output (from encode_chunks or full encode)

        Returns:
            mx.array: [seq/4, decoder_dim] adapter output
        """
        seq_len = encoded.shape[0]
        ds = self.config.downsample_factor
        ds_len = seq_len // ds
        if ds_len == 0:
            return encoded[:0]  # empty
        x = encoded[: ds_len * ds].reshape(ds_len, self.config.dim * ds)
        x = nn.gelu(self.audio_language_projection_0(x))
        return self.audio_language_projection_2(x)

    def encode_full(self, conv_out):
        """Non-chunked encoding of conv output using optimized causal attention.

        Uses SDPA's native causal mask ("causal" string) which enables Flash
        Attention. Only valid when conv_out fits within the sliding window.

        Args:
            conv_out: [seq, dim] output from conv_stem()

        Returns:
            mx.array: [seq/4, decoder_dim] adapter output
        """
        x = conv_out
        for layer in self.transformer_layers:
            x = layer(x, 0, "causal")
        x = self.transformer_norm(x)
        return self.downsample_and_project(x)

    def __call__(self, mel):
        """Full encode: conv stem + all transformer layers + downsample + project.

        Args:
            mel: [mel_bins, frames] log-mel spectrogram

        Returns:
            mx.array: [seq/4, decoder_dim] adapter output
        """
        conv_out = self.conv_stem(mel)
        seq_len = conv_out.shape[0]
        sw = self.config.sliding_window

        if seq_len <= sw:
            return self.encode_full(conv_out)
        else:
            x = mx.concatenate(list(self.encode_chunks(conv_out)), axis=0)
            return self.downsample_and_project(x)
