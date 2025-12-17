import math

import mlx.core as mx
import mlx.nn as nn


class AttentionQKV(nn.Module):
    """Multi-head attention with separate Q, K, V projections."""

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        dropout_rate: float = 0.1,
        scale: float = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim**-0.5
        self.dropout_rate = dropout_rate

    def __call__(
        self, q: mx.array, k: mx.array, v: mx.array, mask: mx.array = None
    ) -> mx.array:
        """
        Args:
            q: Query tensor (B, T_q, n_heads * head_dim)
            k: Key tensor (B, T_k, n_heads * head_dim)
            v: Value tensor (B, T_v, n_heads * head_dim)
            mask: Optional attention mask (boolean, True=attend)

        Returns:
            Output tensor (B, T_q, n_heads * head_dim)
        """
        # Split heads: (B, T, D) -> (B, n_heads, T, head_dim)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Use MLX fast attention (fused kernel)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        return self.combine_heads(out)

    def split_heads(self, x: mx.array) -> mx.array:
        """(B, T, D) -> (B, n_heads, T, head_dim)"""
        B, T, _ = x.shape
        x = mx.reshape(x, (B, T, self.n_heads, self.head_dim))
        return mx.transpose(x, (0, 2, 1, 3))

    def combine_heads(self, x: mx.array) -> mx.array:
        """(B, n_heads, T, head_dim) -> (B, T, D)"""
        B, _, T, _ = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return mx.reshape(x, (B, T, -1))


class AttentionBlock(nn.Module):
    """
    Cross-attention block with separate Q, K, V linear transformations.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
        scale: float = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(channels)

        # Separate linear layers for Q, K, V
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)

        self.attention = AttentionQKV(
            num_heads, channels // num_heads, dropout_rate=dropout_rate, scale=scale
        )

        self.proj_out = nn.Linear(channels, channels)

    def __call__(self, x1: mx.array, x2: mx.array, mask: mx.array = None) -> mx.array:
        """
        Cross-attention from x1 to x2.

        Args:
            x1: Query source (B, T1, C)
            x2: Key/Value source (B, T2, C)
            mask: Optional attention mask

        Returns:
            Output (B, T1, C)
        """
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        q = self.to_q(x1_norm)
        k = self.to_k(x2_norm)
        v = self.to_v(x2_norm)

        h = self.attention(q, k, v, mask=mask)
        h = self.proj_out(h)

        return x1 + h


class Perceiver(nn.Module):
    """
    Perceiver-style resampler for conditioning embeddings.
    Reduces variable-length input to fixed-length latent representation.

    Note: Uses a single shared attention block for both cross-attention
    and self-attention, matching the original PyTorch implementation.
    """

    def __init__(
        self,
        pre_attention_query_token: int = 32,
        pre_attention_query_size: int = 1024,
        embedding_dim: int = 1024,
        num_attn_heads: int = 4,
    ):
        """
        Args:
            pre_attention_query_token: Number of query tokens (output length)
            pre_attention_query_size: Size of each query token
            embedding_dim: Dimension of the embedding space
            num_attn_heads: Number of attention heads
        """
        super().__init__()

        # Learnable query tokens - initialize with uniform distribution
        # This is stored as a module attribute that will be tracked as a parameter
        query_variance = math.sqrt(3.0) * math.sqrt(
            2.0 / (pre_attention_query_token + pre_attention_query_token)
        )
        # Use a linear layer with no bias as a workaround to store learnable params
        # We'll initialize with our custom values
        self._query_shape = (1, pre_attention_query_token, pre_attention_query_size)
        self.pre_attention_query = mx.random.uniform(
            low=-query_variance, high=query_variance, shape=self._query_shape
        )

        # Single shared attention block (used for both cross and self attention)
        # This matches the original PyTorch implementation
        self.attn = AttentionBlock(embedding_dim, num_attn_heads)

    def __call__(self, h: mx.array) -> mx.array:
        """
        Args:
            h: Input embeddings (B, T, D) - variable length T

        Returns:
            Fixed-length output (B, query_tokens, D)
        """
        B = h.shape[0]

        # Expand query to batch size
        query = mx.broadcast_to(
            self.pre_attention_query, (B,) + self.pre_attention_query.shape[1:]
        )

        # Cross-attention: query attends to input
        pre_att = self.attn(query, h)

        # Self-attention: query attends to itself
        attn_out = self.attn(pre_att, pre_att)

        return attn_out

    def parameters(self):
        """Return learnable parameters including the query tokens."""
        params = super().parameters()
        # Add pre_attention_query as a learnable parameter
        params["pre_attention_query"] = self.pre_attention_query
        return params
