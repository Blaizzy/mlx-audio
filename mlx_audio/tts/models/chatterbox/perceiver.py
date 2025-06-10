import math

import mlx.core as mx
import mlx.nn as nn
import mlx.nn.functional as F
from einops import rearrange


class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = mx.abs(n)
        else:
            n = mx.max(n, mx.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                mx.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = mx.min(val_if_large, mx.full_like(val_if_large, num_buckets - 1))

        ret += mx.where(is_small, n, val_if_large)
        return ret

    def __call__(self, qk_dots):
        i, j = qk_dots.shape[-2:]
        q_pos = mx.arange(i, dtype=mx.long)
        k_pos = mx.arange(j, dtype=mx.long)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)

        bias = mx.transpose(values, (2, 0, 1))[
            None, ...
        ]  # bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + (bias * self.scale)


class AttentionQKV(nn.Module):
    def __init__(self, n_heads, head_dim, dropout_rate=0.1, scale=None, flash=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim**-0.5
        self.flash = flash
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def __call__(self, q, k, v, mask=None):
        q, k, v = [self.split_heads(tensor) for tensor in [q, k, v]]
        out = self.scaled_dot_product_attention(q, k, v, mask=mask)

        return self.combine_heads(out)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        sim = mx.einsum("bhlt,bhls->bhts", q, k) * self.scale
        if mask is not None:
            sim = mx.where(
                mask == 0, float("-inf"), sim
            )  # sim = sim.masked_fill(mask == 0, float('-inf'))
        attn = mx.softmax(sim, axis=-1)
        return mx.einsum("bhts,bhls->bhlt", attn, v)

    def split_heads(self, x):
        bs, length, _ = x.shape
        x = x.reshape(bs, length, self.n_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        bs, _, length, _ = x.shape
        x = x.transpose(0, 2, 1, 3).contiguous()
        return x.reshape(bs, length, -1)


class AttentionBlock2(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other,
    using AttentionQKV and separate linear transformations for Q, K, and V.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        relative_pos_embeddings=False,
        flash_attention=True,
        dropout_rate=0.2,
        scale=None,
    ):
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = nn.LayerNorm(channels)

        # Separate linear layers for Q, K, and V
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)

        self.attention = AttentionQKV(
            self.num_heads,
            channels // self.num_heads,
            dropout_rate=dropout_rate,
            flash=flash_attention,
            scale=scale,
        )

        self.proj_out = nn.Linear(channels, channels)

        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(
                scale=(channels // self.num_heads) ** 0.5,
                causal=False,
                heads=num_heads,
                num_buckets=32,
                max_distance=64,
            )
        else:
            self.relative_pos_embeddings = None

    def __call__(self, x1, x2, mask=None):
        b1, c1, *spatial1 = x1.shape
        b2, c2, *spatial2 = x2.shape

        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        q = self.to_q(x1_norm)
        k = self.to_k(x2_norm)
        v = self.to_v(x2_norm)

        h = self.attention(q, k, v, mask=mask)
        h = self.proj_out(h)

        return (x1 + h).reshape(b1, c1, *spatial1)


class Perceiver(nn.Module):
    """Inspired by https://arxiv.org/abs/2103.03206"""

    def __init__(
        self,
        pre_attention_query_token=32,
        pre_attention_query_size=1024,
        embedding_dim=1024,
        num_attn_heads=4,
    ):
        """
        Initialize the perceiver module.

        :param pre_attention_query_token: Number of query tokens for pre-attention
        :param pre_attention_query_size: Size of each query token
        :param embedding_dim: Dimension of the embedding space
        :param num_attn_heads: Number of attention heads
        """
        super().__init__()

        # Initialize the pre-attention query parameter
        self.pre_attention_query = mx.zeros(
            (1, pre_attention_query_token, pre_attention_query_size)
        )

        # Initialize the attention block
        self.attn = AttentionBlock2(embedding_dim, num_attn_heads)

    def __call__(self, h):
        """
        Forward pass of the perceiver module.
        :param h: Input tensor
        :return: Output after applying attention mechanisms
        """
        # Expand the pre-attention query to match the batch size of the input (mlx)
        query_ = mx.broadcast_to(
            self.pre_attention_query, (h.shape[0],) + self.pre_attention_query.shape[1:]
        )  # self.pre_attention_query.expand(h.shape[0], -1, -1)
        # Apply the first attention mechanism (cross-attention)
        pre_att = self.attn(query_, h)
        # Apply the second attention mechanism (self-attention)
        attn = self.attn(pre_att, pre_att)
        return attn
