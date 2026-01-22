# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from .config import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig


class RMSNorm(nn.Module):
    """RMS Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Cast to float32 for stability
        x_float = x.astype(mx.float32)
        variance = mx.mean(x_float**2, axis=-1, keepdims=True)
        x_normed = x_float * mx.rsqrt(variance + self.eps)
        return (self.weight * x_normed).astype(x.dtype)


def rotate_half(x: mx.array) -> mx.array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Applies Rotary Position Embedding to query and key tensors."""
    # cos, sin: [batch, seq_len, head_dim]
    # Expand for heads dimension
    cos = mx.expand_dims(cos, axis=1)  # [batch, 1, seq_len, head_dim]
    sin = mx.expand_dims(sin, axis=1)  # [batch, 1, seq_len, head_dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    unsqueeze_dim: int = 1,
) -> Tuple[mx.array, mx.array]:
    """Applies Multimodal RoPE to query and key tensors.

    The interleaved MRoPE combination is done in TalkerRotaryEmbedding,
    so here we just apply the combined cos/sin.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine embeddings [batch, seq_len, head_dim]
        sin: Sine embeddings [batch, seq_len, head_dim]
        unsqueeze_dim: Dimension to unsqueeze for broadcasting
    """
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self._inv_freq = inv_freq

    def __call__(
        self, x: mx.array, position_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            position_ids: Position indices [batch, seq_len]

        Returns:
            cos, sin: [batch, seq_len, head_dim]
        """
        # Expand inv_freq: [1, head_dim/2, 1]
        inv_freq = mx.expand_dims(self._inv_freq, axis=(0, 2))
        # position_ids: [batch, 1, seq_len]
        pos = mx.expand_dims(position_ids.astype(mx.float32), axis=1)

        # Compute frequencies: [batch, head_dim/2, seq_len]
        freqs = inv_freq * pos
        # Transpose: [batch, seq_len, head_dim/2]
        freqs = mx.transpose(freqs, (0, 2, 1))

        # Concatenate for full head_dim
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).astype(x.dtype)
        sin = mx.sin(emb).astype(x.dtype)

        return cos, sin


class TalkerRotaryEmbedding(nn.Module):
    """Multimodal Rotary Embedding for 3D positions (temporal, height, width).

    Uses interleaved MRoPE layout for better spatial-temporal modeling.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        mrope_section: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.mrope_section = mrope_section or [24, 20, 20]

        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self._inv_freq = inv_freq

    def apply_interleaved_mrope(
        self, freqs: mx.array, mrope_section: List[int]
    ) -> mx.array:
        """Apply interleaved MRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.

        Args:
            freqs: [3, batch, seq_len, head_dim // 2]
            mrope_section: [temporal_dims, height_dims, width_dims]

        Returns:
            freqs_combined: [batch, seq_len, head_dim // 2]
        """
        head_dim_half = freqs.shape[-1]

        # Start with temporal frequencies as base
        freqs_t = freqs[0]  # [batch, seq_len, head_dim // 2]
        freqs_h = freqs[1]  # [batch, seq_len, head_dim // 2]
        freqs_w = freqs[2]  # [batch, seq_len, head_dim // 2]

        # Create masks for interleaved positions
        # H gets positions 1, 4, 7, ... (offset 1, step 3)
        # W gets positions 2, 5, 8, ... (offset 2, step 3)
        # T keeps positions 0, 3, 6, ... and everything after the interleave region

        indices = mx.arange(head_dim_half)

        # H mask: positions where index % 3 == 1, up to length mrope_section[1] * 3
        h_length = mrope_section[1] * 3
        h_mask = (indices % 3 == 1) & (indices < h_length)

        # W mask: positions where index % 3 == 2, up to length mrope_section[2] * 3
        w_length = mrope_section[2] * 3
        w_mask = (indices % 3 == 2) & (indices < w_length)

        # Expand masks for broadcasting: [1, 1, head_dim // 2]
        h_mask = h_mask.reshape(1, 1, -1)
        w_mask = w_mask.reshape(1, 1, -1)

        # Apply interleaved combination
        freqs_combined = mx.where(h_mask, freqs_h, freqs_t)
        freqs_combined = mx.where(w_mask, freqs_w, freqs_combined)

        return freqs_combined

    def __call__(
        self, x: mx.array, position_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            position_ids: Position indices [3, batch, seq_len] for 3D positions

        Returns:
            cos, sin: [batch, seq_len, head_dim] (already combined via interleaved MRoPE)
        """
        # Ensure position_ids has 3D shape
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )

        # Expand inv_freq: [1, 1, head_dim/2, 1]
        inv_freq = mx.broadcast_to(
            self._inv_freq[None, None, :, None].astype(mx.float32),
            (3, position_ids.shape[1], self._inv_freq.shape[0], 1),
        )

        # position_ids: [3, batch, 1, seq_len]
        pos = mx.expand_dims(position_ids.astype(mx.float32), axis=2)

        # Compute frequencies: [3, batch, head_dim/2, seq_len]
        freqs = inv_freq @ pos
        # Transpose: [3, batch, seq_len, head_dim/2]
        freqs = mx.swapaxes(freqs, 2, 3)

        # Apply interleaved MRoPE to combine the 3 modalities
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)

        # Concatenate for full head_dim: [batch, seq_len, head_dim]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).astype(x.dtype)
        sin = mx.sin(emb).astype(x.dtype)

        return cos, sin


class TalkerAttention(nn.Module):
    """Multi-head attention with MRoPE support."""

    def __init__(self, config: Qwen3TTSTalkerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5
        self.rope_scaling = config.rope_scaling

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # QK normalization (like Qwen3)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, seq_len, num_heads, head_dim]
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Apply rotary embeddings (interleaving already done in TalkerRotaryEmbedding)
        cos, sin = position_embeddings
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache using mlx_lm's KVCache
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Use fast scaled dot product attention with GQA support
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        # Reshape back
        output = mx.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, -1)

        return self.o_proj(output)


class TalkerMLP(nn.Module):
    """MLP with SwiGLU activation."""

    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class ResizeMLP(nn.Module):
    """MLP for resizing hidden dimensions."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)

        if hidden_act == "silu":
            self.act_fn = nn.silu
        elif hidden_act == "gelu":
            self.act_fn = nn.gelu
        elif hidden_act == "relu":
            self.act_fn = nn.relu
        else:
            self.act_fn = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class TalkerDecoderLayer(nn.Module):
    """Transformer decoder layer for talker model."""

    def __init__(self, config: Qwen3TTSTalkerConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = TalkerAttention(config, layer_idx)
        self.mlp = TalkerMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        # Self attention
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_embeddings, mask, cache)
        x = residual + x

        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Qwen3TTSTalkerModel(nn.Module):
    """Main talker transformer model."""

    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Embeddings
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(
            config.text_vocab_size, config.text_hidden_size
        )

        # Transformer layers
        self.layers = [
            TalkerDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embeddings with MRoPE section from config
        mrope_section = None
        if config.rope_scaling and "mrope_section" in config.rope_scaling:
            mrope_section = config.rope_scaling["mrope_section"]

        self.rotary_emb = TalkerRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            mrope_section=mrope_section,
        )

    def __call__(
        self,
        inputs_embeds: mx.array,
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
    ) -> mx.array:
        batch, seq_len, _ = inputs_embeds.shape

        # Get offset from cache for position calculation
        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset

        # Generate position ids if not provided
        if position_ids is None:
            # 3D position for MRoPE: [3, batch, seq_len]
            pos = mx.arange(offset, offset + seq_len)[None, :].astype(mx.int32)
            pos = mx.broadcast_to(pos, (batch, seq_len))
            position_ids = mx.stack([pos, pos, pos], axis=0)

        # Compute position embeddings
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Create causal mask if not provided
        if mask is None and seq_len > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(inputs_embeds.dtype)

        x = inputs_embeds

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, position_embeddings, mask, layer_cache)

        x = self.norm(x)
        return x

    def make_cache(self) -> List[KVCache]:
        """Create KV cache for all layers."""
        return [KVCache() for _ in self.layers]


class CodePredictorAttention(nn.Module):
    """Attention for the code predictor with standard RoPE."""

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache using mlx_lm's KVCache
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Use fast scaled dot product attention with GQA support
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        output = mx.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, -1)

        return self.o_proj(output)


class CodePredictorMLP(nn.Module):
    """MLP for code predictor."""

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig):
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


class CodePredictorDecoderLayer(nn.Module):
    """Decoder layer for code predictor."""

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, layer_idx: int):
        super().__init__()
        self.self_attn = CodePredictorAttention(config, layer_idx)
        self.mlp = CodePredictorMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
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
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class CodePredictorModel(nn.Module):
    """Inner model for code predictor (to match PyTorch weight structure)."""

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_hidden_size: int,
    ):
        super().__init__()
        self.config = config

        # Embeddings for each code group (except first)
        # Note: These are indexed 0 to num_code_groups-2 in weights
        self.codec_embedding = [
            nn.Embedding(config.vocab_size, talker_hidden_size)
            for _ in range(config.num_code_groups - 1)
        ]

        # Transformer layers
        self.layers = [
            CodePredictorDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def __call__(
        self,
        inputs_embeds: mx.array,
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
    ) -> mx.array:
        batch, seq_len, _ = inputs_embeds.shape

        # Get offset from cache for position calculation
        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset

        # Position ids
        if position_ids is None:
            position_ids = mx.arange(offset, offset + seq_len)[None, :]
            position_ids = mx.broadcast_to(position_ids, (batch, seq_len))

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        if mask is None and seq_len > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(inputs_embeds.dtype)

        x = inputs_embeds

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, position_embeddings, mask, layer_cache)

        x = self.norm(x)
        return x

    def make_cache(self) -> List[KVCache]:
        """Create KV cache for all layers."""
        return [KVCache() for _ in self.layers]


class Qwen3TTSTalkerCodePredictor(nn.Module):
    """Code predictor sub-model for multi-codebook token prediction."""

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.num_code_groups = config.num_code_groups

        # Inner model (matches PyTorch weight structure: code_predictor.model.*)
        self.model = CodePredictorModel(config, talker_hidden_size)

        # LM heads for each code group (except first)
        self.lm_head = [
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        ]

    @property
    def codec_embedding(self):
        """Access codec embeddings from inner model."""
        return self.model.codec_embedding

    def __call__(
        self,
        inputs_embeds: mx.array,
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
        generation_step: int = 0,
    ) -> Tuple[mx.array, Optional[List[KVCache]], int]:
        # Forward through inner model
        x = self.model(inputs_embeds, position_ids, mask, cache)

        # Get logits from appropriate head
        logits = self.lm_head[generation_step](x)

        return logits, cache, generation_step + 1

    def make_cache(self) -> List[KVCache]:
        """Create KV cache for all layers."""
        return self.model.make_cache()


class Qwen3TTSTalkerForConditionalGeneration(nn.Module):
    """Full talker model for conditional generation."""

    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__()
        self.config = config

        self.model = Qwen3TTSTalkerModel(config)

        # Text projection MLP
        self.text_projection = ResizeMLP(
            config.text_hidden_size,
            config.text_hidden_size,
            config.hidden_size,
            config.hidden_act,
            bias=True,
        )

        # Codec head for first token prediction
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Code predictor for remaining tokens
        self.code_predictor = Qwen3TTSTalkerCodePredictor(
            config.code_predictor_config, config.hidden_size
        )

    def get_input_embeddings(self):
        return self.model.codec_embedding

    def get_text_embeddings(self):
        return self.model.text_embedding

    def __call__(
        self,
        inputs_embeds: mx.array,
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Forward pass for the talker model.

        Returns:
            logits: Logits for next token prediction
            hidden_states: Last hidden states
        """
        hidden_states = self.model(inputs_embeds, position_ids, mask, cache)
        logits = self.codec_head(hidden_states)
        return logits, hidden_states

    def make_cache(self) -> List[KVCache]:
        """Create KV cache for all layers."""
        return self.model.make_cache()

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights from PyTorch to MLX format."""
        sanitized = {}
        for k, v in weights.items():
            if not k.startswith("talker."):
                continue

            new_key = k.replace("talker.", "")

            # No transpositions needed for linear layers in MLX
            sanitized[new_key] = v

        return sanitized
