"""Conformer CTC encoder for Granite Speech.

Weight key structure (per layer):
  encoder.layers.{i}.ff1.{pre_norm,up_proj,down_proj}.{weight,bias}
  encoder.layers.{i}.attn.{pre_norm,to_q,to_kv,to_out,rel_pos_emb}.{weight,bias}
  encoder.layers.{i}.conv.{up_conv,batch_norm,depth_conv.conv,down_conv,norm}.{weight,bias}
  encoder.layers.{i}.ff2.{pre_norm,up_proj,down_proj}.{weight,bias}
  encoder.layers.{i}.post_norm.{weight,bias}
  encoder.input_linear.{weight,bias}
  encoder.out.{weight,bias}
  encoder.out_mid.{weight,bias}
"""

import mlx.core as mx
import mlx.nn as nn

from .config import EncoderConfig


class BatchNorm1d(nn.Module):
    """Inference-only BatchNorm1d (no running stats update)."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.running_mean) / mx.sqrt(
            self.running_var + self.eps
        ) * self.weight + self.bias


class ConformerFeedForward(nn.Module):
    """Macaron-style feed-forward with pre-norm, SiLU gate, and half-step residual.

    Keys: ff{1,2}.pre_norm, ff{1,2}.up_proj, ff{1,2}.down_proj
    """

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        self.pre_norm = nn.LayerNorm(dim)
        self.up_proj = nn.Linear(dim, inner_dim, bias=True)
        self.down_proj = nn.Linear(inner_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.pre_norm(x)
        h = nn.silu(self.up_proj(h))
        return self.down_proj(h)


class ConformerAttention(nn.Module):
    """Multi-head attention with Shaw relative position encoding and block attention.

    Keys: attn.pre_norm, attn.to_q, attn.to_kv, attn.to_out, attn.rel_pos_emb
    Shapes:
      to_q.weight: (hidden_dim, hidden_dim)
      to_kv.weight: (2*hidden_dim, hidden_dim)  [combined K and V]
      to_out.weight: (hidden_dim, hidden_dim), bias: (hidden_dim,)
      rel_pos_emb.weight: (2*max_pos_emb+1, dim_head)
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim_head = config.dim_head
        self.context_size = config.context_size
        self.max_pos_emb = config.max_pos_emb
        hidden_dim = config.hidden_dim

        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.rel_pos_emb = nn.Embedding(2 * config.max_pos_emb + 1, config.dim_head)

        self.scale = config.dim_head**-0.5

    def _shaw_rel_pos_bias(self, seq_len: int) -> mx.array:
        """Compute Shaw relative position bias for a block of seq_len.

        Returns: (seq_len, seq_len) bias matrix
        """
        positions = mx.arange(seq_len)
        # rel_pos shape: (seq_len, seq_len), values in range [-seq_len+1, seq_len-1]
        rel_pos = positions[:, None] - positions[None, :]
        rel_pos_clipped = mx.clip(rel_pos, -self.max_pos_emb, self.max_pos_emb)
        rel_pos_idx = (rel_pos_clipped + self.max_pos_emb).astype(mx.int32)

        # Get embeddings: (seq_len, seq_len, dim_head)
        rel_emb = self.rel_pos_emb(rel_pos_idx)
        return rel_emb

    def _attend_block(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """Attend within a single block.

        Args:
            q, k, v: (batch, heads, seq_len, dim_head)
        Returns: (batch, heads, seq_len, dim_head)
        """
        B, H, S, D = q.shape

        # Standard attention logits: (B, H, S, S)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Shaw relative position bias
        # q: (B, H, S, D), rel_emb: (S, S, D)
        # For each query position i and key position j: q[i] · rel_emb[i,j]
        rel_emb = self._shaw_rel_pos_bias(S)  # (S, S, D)
        # Compute: q @ rel_emb^T for each (i,j) pair
        # q: (B, H, S, D) -> (B*H, S, D)
        # rel_emb: (S, S, D) -> for each query pos i, (S, D)
        # Result: (B, H, S, S) rel position bias
        rel_bias = mx.einsum("bhid,ijd->bhij", q, rel_emb) * self.scale
        attn = attn + rel_bias

        attn = mx.softmax(attn, axis=-1)
        return attn @ v

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        h = self.pre_norm(x)

        q = self.to_q(h)
        kv = self.to_kv(h)
        k, v = mx.split(kv, 2, axis=-1)

        # Reshape to multi-head: (B, T, C) -> (B, H, T, D)
        q = q.reshape(B, T, self.num_heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.dim_head).transpose(0, 2, 1, 3)

        ctx = self.context_size
        if T <= ctx:
            out = self._attend_block(q, k, v)
        else:
            # Block attention: split sequence into blocks of context_size
            num_blocks = (T + ctx - 1) // ctx
            pad_len = num_blocks * ctx - T

            if pad_len > 0:
                q = mx.pad(q, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
                k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
                v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])

            # Reshape to blocks: (B, H, num_blocks, ctx, D)
            q = q.reshape(B, self.num_heads, num_blocks, ctx, self.dim_head)
            k = k.reshape(B, self.num_heads, num_blocks, ctx, self.dim_head)
            v = v.reshape(B, self.num_heads, num_blocks, ctx, self.dim_head)

            # Process each block
            block_outs = []
            for i in range(num_blocks):
                bq = q[:, :, i, :, :]  # (B, H, ctx, D)
                bk = k[:, :, i, :, :]
                bv = v[:, :, i, :, :]
                block_outs.append(self._attend_block(bq, bk, bv))

            out = mx.concatenate(block_outs, axis=2)  # (B, H, num_blocks*ctx, D)

            if pad_len > 0:
                out = out[:, :, :T, :]

        # (B, H, T, D) -> (B, T, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.to_out(out)


class ConformerConvModule(nn.Module):
    """Conformer convolution module with GLU, depthwise conv, and batch norm.

    Order: LayerNorm -> up_conv -> GLU -> depth_conv -> BatchNorm -> SiLU -> down_conv

    Keys: conv.norm, conv.up_conv, conv.batch_norm, conv.depth_conv.conv, conv.down_conv
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        dim = config.hidden_dim
        inner_dim = dim * config.conv_expansion_factor  # 2048
        kernel_size = config.conv_kernel_size  # 15
        padding = (kernel_size - 1) // 2

        self.norm = nn.LayerNorm(dim)
        self.up_conv = nn.Conv1d(dim, inner_dim * 2, kernel_size=1, bias=True)
        self.depth_conv = DepthWiseConv1d(inner_dim, kernel_size, padding=padding)
        self.batch_norm = BatchNorm1d(inner_dim)
        self.down_conv = nn.Conv1d(inner_dim, dim, kernel_size=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        h = self.up_conv(h)  # (B, T, inner_dim*2)

        # GLU: split and gate
        h1, h2 = mx.split(h, 2, axis=-1)
        h = h1 * nn.sigmoid(h2)  # (B, T, inner_dim)

        h = self.depth_conv(h)
        h = nn.silu(self.batch_norm(h))
        h = self.down_conv(h)

        return h


class DepthWiseConv1d(nn.Module):
    """Depthwise separable 1D convolution.

    Keys: depth_conv.conv.weight
    PyTorch weight shape: (channels, 1, kernel_size), groups=channels
    MLX weight shape: (channels, kernel_size, 1) after transposition
    """

    def __init__(self, channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class ConformerBlock(nn.Module):
    """Single Conformer block: ff1 -> attn -> conv -> ff2 -> post_norm.

    Uses half-step residual for feed-forward modules.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.ff1 = ConformerFeedForward(config.hidden_dim, config.feedforward_mult)
        self.attn = ConformerAttention(config)
        self.conv = ConformerConvModule(config)
        self.ff2 = ConformerFeedForward(config.hidden_dim, config.feedforward_mult)
        self.post_norm = nn.LayerNorm(config.hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # Macaron-style: half-step FFN, attention, conv, half-step FFN
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.post_norm(x)


class CTCEncoder(nn.Module):
    """CTC Conformer encoder with mid-layer self-conditioning.

    Keys:
      encoder.input_linear.{weight,bias}
      encoder.layers.{i}.*
      encoder.out.{weight,bias}     (hidden_dim -> output_dim CTC projection)
      encoder.out_mid.{weight,bias} (output_dim -> hidden_dim self-conditioning)
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.mid_layer = config.num_layers // 2  # Layer 8 for 16 layers

        self.input_linear = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.layers = [ConformerBlock(config) for _ in range(config.num_layers)]

        # CTC heads
        self.out = nn.Linear(config.hidden_dim, config.output_dim, bias=True)
        self.out_mid = nn.Linear(config.output_dim, config.hidden_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """Encode audio features.

        Args:
            x: (batch, seq_len, input_dim) stacked mel features

        Returns:
            (batch, seq_len, hidden_dim) encoder output
        """
        x = self.input_linear(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Mid-layer CTC self-conditioning
            if i == self.mid_layer - 1:
                ctc_out = mx.softmax(self.out(x), axis=-1)
                x = x + self.out_mid(ctc_out)

        return x
