import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer."""

    def __init__(
        self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True
    ):
        """
        Args:
            n_head: Number of attention heads
            n_feat: Number of features
            dropout_rate: Dropout rate
            key_bias: Whether to use bias in key projection
        """
        super().__init__()
        assert n_feat % n_head == 0, "n_feat must be divisible by n_head"

        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout_rate = dropout_rate

    def forward_qkv(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Transform query, key and value.

        Args:
            query: (B, T1, D)
            key: (B, T2, D)
            value: (B, T2, D)

        Returns:
            q: (B, n_head, T1, d_k)
            k: (B, n_head, T2, d_k)
            v: (B, n_head, T2, d_k)
        """
        n_batch = query.shape[0]
        q = self.linear_q(query).reshape(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).reshape(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).reshape(n_batch, -1, self.h, self.d_k)

        q = mx.transpose(q, (0, 2, 1, 3))  # (B, h, T1, d_k)
        k = mx.transpose(k, (0, 2, 1, 3))  # (B, h, T2, d_k)
        v = mx.transpose(v, (0, 2, 1, 3))  # (B, h, T2, d_k)

        return q, k, v

    def forward_attention(
        self, value: mx.array, scores: mx.array, mask: mx.array = None
    ) -> mx.array:
        """
        Compute attention context vector.

        Args:
            value: (B, n_head, T2, d_k)
            scores: (B, n_head, T1, T2)
            mask: (B, 1, T2) or (B, T1, T2) or None

        Returns:
            output: (B, T1, d_model)
        """
        n_batch = value.shape[0]

        if mask is not None and mask.shape[2] > 0:
            # Expand mask: (B, 1, T2) -> (B, 1, 1, T2)
            mask = mx.expand_dims(mask, 1)
            # Truncate mask to match scores length
            mask = mask[:, :, :, : scores.shape[-1]]
            # Apply mask
            scores = mx.where(mask == 0, -float("inf"), scores)
            attn = mx.softmax(scores, axis=-1)
            attn = mx.where(mask == 0, 0.0, attn)
        else:
            attn = mx.softmax(scores, axis=-1)

        # Apply dropout during training
        if self.training and self.dropout_rate > 0:
            attn = nn.Dropout(self.dropout_rate)(attn)

        x = attn @ value  # (B, h, T1, d_k)
        x = mx.transpose(x, (0, 2, 1, 3))  # (B, T1, h, d_k)
        x = mx.reshape(x, (n_batch, -1, self.h * self.d_k))  # (B, T1, d_model)

        return self.linear_out(x)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array = None,
        pos_emb: mx.array = None,
        cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Scaled dot product attention.

        Args:
            query: (B, T1, size)
            key: (B, T2, size)
            value: (B, T2, size)
            mask: (B, 1, T2) or (B, T1, T2) or None
            pos_emb: Not used in base class
            cache: (1, head, cache_t, d_k * 2) for KV caching

        Returns:
            output: (B, T1, d_model)
            new_cache: (1, head, cache_t + T1, d_k * 2)
        """
        q, k, v = self.forward_qkv(query, key, value)

        # Handle KV caching
        if cache is not None and cache.shape[0] > 0:
            key_cache, value_cache = mx.split(cache, 2, axis=-1)
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        new_cache = mx.concatenate([k, v], axis=-1)

        scores = (q @ mx.swapaxes(k, -2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention with relative positional encoding."""

    def __init__(
        self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True
    ):
        super().__init__(n_head, n_feat, dropout_rate, key_bias)

        # Linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        # Learnable biases for relative position
        self.pos_bias_u = mx.random.uniform(
            low=-1.0, high=1.0, shape=(self.h, self.d_k)
        ) * math.sqrt(6.0 / (self.h + self.d_k))
        self.pos_bias_v = mx.random.uniform(
            low=-1.0, high=1.0, shape=(self.h, self.d_k)
        ) * math.sqrt(6.0 / (self.h + self.d_k))

    def rel_shift(self, x: mx.array) -> mx.array:
        """
        Compute relative positional encoding.

        Args:
            x: (B, head, T1, 2*T1-1)

        Returns:
            shifted: (B, head, T1, T1)
        """
        zero_pad = mx.zeros((x.shape[0], x.shape[1], x.shape[2], 1))
        x_padded = mx.concatenate([zero_pad, x], axis=-1)

        x_padded = mx.reshape(
            x_padded, (x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2])
        )
        x = mx.reshape(x_padded[:, :, 1:], x.shape)[:, :, :, : x.shape[-1] // 2 + 1]

        return x

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array = None,
        pos_emb: mx.array = None,
        cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Scaled dot product attention with relative positional encoding.

        Args:
            query: (B, T1, size)
            key: (B, T2, size)
            value: (B, T2, size)
            mask: (B, 1, T2) or (B, T1, T2)
            pos_emb: (B, T2, size) positional embeddings
            cache: (1, head, cache_t, d_k * 2)

        Returns:
            output: (B, T1, d_model)
            new_cache: (1, head, cache_t + T1, d_k * 2)
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = mx.transpose(q, (0, 2, 1, 3))  # (B, T1, h, d_k)

        # Handle KV caching
        if cache is not None and cache.shape[0] > 0:
            key_cache, value_cache = mx.split(cache, 2, axis=-1)
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        new_cache = mx.concatenate([k, v], axis=-1)

        # Process positional embeddings
        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb).reshape(n_batch_pos, -1, self.h, self.d_k)
        p = mx.transpose(p, (0, 2, 1, 3))  # (B, h, T1, d_k)

        # Add biases to query
        q_with_bias_u = mx.transpose(
            q + self.pos_bias_u, (0, 2, 1, 3)
        )  # (B, h, T1, d_k)
        q_with_bias_v = mx.transpose(
            q + self.pos_bias_v, (0, 2, 1, 3)
        )  # (B, h, T1, d_k)

        # Compute attention scores with relative position
        matrix_ac = q_with_bias_u @ mx.swapaxes(k, -2, -1)
        matrix_bd = q_with_bias_v @ mx.swapaxes(p, -2, -1)

        # Apply relative shift if needed
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask), new_cache
