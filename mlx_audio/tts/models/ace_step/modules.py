# Copyright 2025 The ACESTEP Team and MLX-Audio Contributors.
# Licensed under the Apache License, Version 2.0.

"""Core modules for ACE-Step model."""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class KVCache:

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Store K,V and return them.

        Args:
            keys: Key tensor [batch, num_heads, seq_len, head_dim]
            values: Value tensor [batch, num_heads, seq_len, head_dim]

        Returns:
            Tuple of (keys, values)
        """
        self.keys = keys
        self.values = values
        return self.keys, self.values

    def fetch(self) -> Tuple[mx.array, mx.array]:
        """Return cached K,V."""
        return self.keys, self.values

    @property
    def is_set(self) -> bool:
        """Check if cache has been populated."""
        return self.keys is not None

    def reset(self):
        """Clear the cache for a new generation."""
        self.keys = None
        self.values = None


def make_cache(num_layers: int) -> List[KVCache]:
    """Create a list of KVCache instances for all layers.

    This follows the mlx-lm pattern of using a simple list of caches.

    Args:
        num_layers: Number of transformer layers

    Returns:
        List of KVCache instances, one per layer
    """
    return [KVCache() for _ in range(num_layers)]


def reset_cache(cache: List[KVCache]):
    """Reset all caches in the list for a new generation."""
    for c in cache:
        c.reset()


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return (self.weight * x).astype(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1000000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self._inv_freq = None

    @property
    def inv_freq(self) -> mx.array:
        if self._inv_freq is None:
            self._inv_freq = 1.0 / (
                self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
            )
        return self._inv_freq

    def __call__(
        self, x: mx.array, position_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Compute rotary position embeddings.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            position_ids: Position indices [batch, seq_len]

        Returns:
            Tuple of (cos, sin) embeddings
        """
        # inv_freq: [dim//2]
        inv_freq = self.inv_freq

        # freqs: [batch, seq_len, dim//2]
        freqs = position_ids[..., None].astype(mx.float32) * inv_freq[None, None, :]

        # emb: [batch, seq_len, dim]
        emb = mx.concatenate([freqs, freqs], axis=-1)

        cos = mx.cos(emb).astype(x.dtype)
        sin = mx.sin(emb).astype(x.dtype)

        return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    """Rotate half the hidden dims of x."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to queries and keys.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine embeddings [batch, seq_len, head_dim]
        sin: Sine embeddings [batch, seq_len, head_dim]

    Returns:
        Tuple of rotated (queries, keys)
    """
    # Expand cos/sin for heads: [batch, 1, seq_len, head_dim]
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class MLP(nn.Module):
    """MLP with SiLU gated activation (Qwen3-style)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    """Multi-head attention for ACE-Step.

    ACE-Step turbo uses standard scaled dot-product attention (softmax)
    for both self-attention and cross-attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        is_cross_attention: bool = False,
        is_causal: bool = False,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.is_cross_attention = is_cross_attention
        self.is_causal = is_causal if not is_cross_attention else False
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=attention_bias
        )

        # QK normalization
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        encoder_hidden_states: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        """Forward pass for attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_embeddings: Optional (cos, sin) tuple for RoPE
            encoder_hidden_states: Optional cross-attention input
            cache: Optional KVCache for cross-attention (stores K,V on first call)

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project queries
        queries = self.q_proj(hidden_states)
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = self.q_norm(queries)
        queries = queries.transpose(0, 2, 1, 3)  # [B, H, S, D]

        # For cross-attention with cache: skip K,V projection if already cached
        if self.is_cross_attention and cache is not None and cache.is_set:
            # Reuse cached K,V (already projected and normalized)
            keys, values = cache.fetch()
        else:
            # Determine key/value source
            if self.is_cross_attention and encoder_hidden_states is not None:
                kv_input = encoder_hidden_states
                kv_len = encoder_hidden_states.shape[1]
            else:
                kv_input = hidden_states
                kv_len = seq_len

            # Project keys and values
            keys = self.k_proj(kv_input)
            values = self.v_proj(kv_input)

            keys = keys.reshape(batch_size, kv_len, self.num_kv_heads, self.head_dim)
            values = values.reshape(
                batch_size, kv_len, self.num_kv_heads, self.head_dim
            )

            keys = self.k_norm(keys)

            keys = keys.transpose(0, 2, 1, 3)  # [B, H, S, D]
            values = values.transpose(0, 2, 1, 3)

            # Apply RoPE for self-attention (not cross-attention)
            if position_embeddings is not None and not self.is_cross_attention:
                cos, sin = position_embeddings
                queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

            # Store in cache for cross-attention
            if self.is_cross_attention and cache is not None:
                cache.update_and_fetch(keys, values)

        kv_len = keys.shape[2]

        # Repeat KV heads for grouped query attention
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            keys = mx.repeat(keys, n_rep, axis=1)
            values = mx.repeat(values, n_rep, axis=1)

        # ACE-Step turbo uses standard scaled dot-product attention (softmax)
        # for both self-attention and cross-attention
        # queries, keys, values: [B, H, S, D]
        scale = 1.0 / math.sqrt(self.head_dim)

        # Compute attention scores
        # scores: [B, H, seq_len, kv_len]
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(queries.dtype)

        # Apply to values
        output = weights @ values  # [B, H, S, D]

        # Reshape output
        output = output.transpose(0, 2, 1, 3)  # [B, S, H, D]
        output = output.reshape(batch_size, seq_len, -1)

        output = self.o_proj(output)

        return output


class TimestepEmbedding(nn.Module):
    """Timestep embedding for diffusion models."""

    def __init__(
        self,
        in_channels: int = 256,
        time_embed_dim: int = 2048,
        scale: float = 1000.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        self.scale = scale

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6, bias=True)

    def timestep_embedding(
        self, t: mx.array, dim: int, max_period: float = 10000.0
    ) -> mx.array:
        """Create sinusoidal timestep embeddings.

        Args:
            t: Timestep tensor [batch]
            dim: Embedding dimension
            max_period: Maximum period for embeddings

        Returns:
            Embeddings of shape [batch, dim]
        """
        t = t * self.scale
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def __call__(self, t: mx.array) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            t: Timestep tensor [batch]

        Returns:
            Tuple of (temb, timestep_proj) where timestep_proj has shape [batch, 6, time_embed_dim]
        """
        t_freq = self.timestep_embedding(t, self.in_channels)
        t_freq = t_freq.astype(t.dtype)
        temb = self.linear_1(t_freq)
        temb = nn.silu(temb)
        temb = self.linear_2(temb)
        timestep_proj = self.time_proj(nn.silu(temb))
        timestep_proj = timestep_proj.reshape(t.shape[0], 6, -1)
        return temb, timestep_proj


class EncoderLayer(nn.Module):
    """Encoder layer with self-attention and MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
            is_cross_attention=False,
            is_causal=False,  # Bidirectional for encoder
            sliding_window=sliding_window,
        )
        self.mlp = MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) tuple for RoPE
            attention_mask: Optional attention mask

        Returns:
            Output hidden states
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DiTLayer(nn.Module):
    """DiT (Diffusion Transformer) layer with adaptive layer norm."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        use_cross_attention: bool = True,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_cross_attention = use_cross_attention

        # Self-attention
        self.self_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
            is_cross_attention=False,
            is_causal=False,  # Bidirectional for DiT
            sliding_window=sliding_window,
        )

        # Cross-attention (optional)
        if use_cross_attention:
            self.cross_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.cross_attn = Attention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                attention_bias=attention_bias,
                rms_norm_eps=rms_norm_eps,
                is_cross_attention=True,
                is_causal=False,
            )

        # MLP
        self.mlp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size)

        # Scale-shift table for adaptive layer norm [1, 6, hidden_size]
        self.scale_shift_table = mx.random.normal((1, 6, hidden_size)) / math.sqrt(
            hidden_size
        )

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        timestep_proj: mx.array,
        attention_mask: Optional[mx.array] = None,
        encoder_hidden_states: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) tuple for RoPE
            timestep_proj: Timestep projection [batch, 6, hidden_size]
            attention_mask: Optional self-attention mask
            encoder_hidden_states: Optional cross-attention input
            encoder_attention_mask: Optional cross-attention mask
            cache: Optional KVCache for cross-attention

        Returns:
            Output hidden states [batch, seq_len, hidden_size]
        """
        # Extract scale-shift parameters from timestep embeddings
        # Shape: [batch, 6, hidden_size]
        modulation = self.scale_shift_table + timestep_proj

        shift_msa = modulation[:, 0:1, :]  # [B, 1, D]
        scale_msa = modulation[:, 1:2, :]
        gate_msa = modulation[:, 2:3, :]
        c_shift_msa = modulation[:, 3:4, :]
        c_scale_msa = modulation[:, 4:5, :]
        c_gate_msa = modulation[:, 5:6, :]

        # Self-attention with adaptive layer norm
        norm_hidden = self.self_attn_norm(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_msa) + shift_msa

        attn_output = self.self_attn(
            norm_hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + attn_output * gate_msa

        # Cross-attention (if enabled)
        if self.use_cross_attention and encoder_hidden_states is not None:
            norm_hidden = self.cross_attn_norm(hidden_states)
            attn_output = self.cross_attn(
                norm_hidden,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                cache=cache,
            )
            hidden_states = hidden_states + attn_output

        # MLP with adaptive layer norm
        norm_hidden = self.mlp_norm(hidden_states)
        norm_hidden = norm_hidden * (1 + c_scale_msa) + c_shift_msa
        ff_output = self.mlp(norm_hidden)
        hidden_states = hidden_states + ff_output * c_gate_msa

        return hidden_states


def create_4d_mask(
    seq_len: int,
    dtype: mx.Dtype,
    attention_mask: Optional[mx.array] = None,
    sliding_window: Optional[int] = None,
    is_sliding_window: bool = False,
    is_causal: bool = True,
) -> mx.array:
    """Create a 4D attention mask.

    Args:
        seq_len: Sequence length
        dtype: Data type for mask
        attention_mask: Optional [batch, seq_len] padding mask
        sliding_window: Window size for sliding window attention
        is_sliding_window: Whether to use sliding window
        is_causal: Whether to use causal masking

    Returns:
        4D mask of shape [batch, 1, seq_len, seq_len]
    """
    # Create base mask
    indices = mx.arange(seq_len)
    diff = indices[:, None] - indices[None, :]  # [seq_len, seq_len]

    # Initialize all True (all positions visible)
    valid_mask = mx.ones((seq_len, seq_len), dtype=mx.bool_)

    # Apply causality
    if is_causal:
        valid_mask = valid_mask & (diff >= 0)

    # Apply sliding window
    if is_sliding_window and sliding_window is not None:
        if is_causal:
            valid_mask = valid_mask & (diff <= sliding_window)
        else:
            valid_mask = valid_mask & (mx.abs(diff) <= sliding_window)

    # Expand to [1, 1, seq_len, seq_len]
    valid_mask = valid_mask[None, None, :, :]

    # Apply padding mask if provided
    if attention_mask is not None:
        # attention_mask: [batch, seq_len]
        # Expand to [batch, 1, 1, seq_len]
        padding_mask = attention_mask[:, None, None, :].astype(mx.bool_)
        valid_mask = valid_mask & padding_mask

    # Convert to additive mask
    min_val = -1e9 if dtype == mx.float32 else -1e4

    # Create mask tensor
    mask = mx.where(
        valid_mask,
        mx.zeros(valid_mask.shape, dtype=dtype),
        mx.full(valid_mask.shape, min_val, dtype=dtype),
    )

    return mask


def pack_sequences(
    hidden1: mx.array,
    hidden2: mx.array,
    mask1: mx.array,
    mask2: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Pack two sequences by concatenating and sorting by mask values.

    Args:
        hidden1: First hidden states [batch, len1, dim]
        hidden2: Second hidden states [batch, len2, dim]
        mask1: First mask [batch, len1]
        mask2: Second mask [batch, len2]

    Returns:
        Tuple of (packed_hidden, new_mask)
    """
    # Concatenate
    hidden_cat = mx.concatenate([hidden1, hidden2], axis=1)
    mask_cat = mx.concatenate([mask1, mask2], axis=1)

    batch_size, total_len, _ = hidden_cat.shape

    # Sort indices so mask=1 comes before mask=0
    sort_idx = mx.argsort(-mask_cat, axis=1)  # Descending sort

    # Gather hidden states
    batch_idx = mx.arange(batch_size)[:, None]
    hidden_sorted = hidden_cat[batch_idx, sort_idx]

    # Create new mask based on valid lengths
    lengths = mx.sum(mask_cat, axis=1)
    new_mask = mx.arange(total_len)[None, :] < lengths[:, None]

    return hidden_sorted, new_mask.astype(mask1.dtype)
