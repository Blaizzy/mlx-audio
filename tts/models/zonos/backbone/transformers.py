# Based on gpt-fast converted to MLX
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Tuple, Optional, Any


class BackboneConfig:
    def __init__(
        self,
        d_model: int = 2048,
        n_layer: int = 24,
        attn_cfg: Dict[str, int] = None,
        attn_mlp_d_intermediate: int = 5632,
        norm_epsilon: float = 1e-5,
        ssm_cfg=None,
    ):
        self.d_model = d_model
        self.n_layer = n_layer
        self.attn_cfg = attn_cfg or {"num_heads": 16, "num_heads_kv": 8}
        self.attn_mlp_d_intermediate = attn_mlp_d_intermediate
        self.norm_epsilon = norm_epsilon
        self.ssm_cfg = ssm_cfg


class InferenceParams:
    def __init__(
        self,
        seqlen_offset: int = 0,
        batch_size_offset: int = 0,
        lengths_per_sample=None,
        key_value_memory_dict: Dict[int, Tuple[mx.array, Any]] = None,
    ):
        self.seqlen_offset = seqlen_offset
        self.batch_size_offset = batch_size_offset
        self.lengths_per_sample = lengths_per_sample
        self.key_value_memory_dict = key_value_memory_dict or {}


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000) -> mx.array:
    """Precompute the frequency cis for rotary embeddings."""
    freqs = 1.0 / (base ** (mx.arange(0, n_elem, 2)[: (n_elem // 2)] / n_elem))
    t = mx.arange(seq_len)
    freqs = mx.outer(t, freqs)

    # Complex exponential in polar form
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)

    # Stack real and imaginary parts
    cache = mx.stack([cos, sin], axis=-1)
    return cache


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Apply rotary embeddings to input tensors."""
    # Reshape input tensor to separate real and imaginary parts
    xshaped = x.reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.reshape(-1, xshaped.shape[1], 1, xshaped.shape[3], 2)

    # Apply complex multiplication in rectangular form
    x_real = xshaped[..., 0]
    x_imag = xshaped[..., 1]
    freqs_cos = freqs_cis[..., 0]
    freqs_sin = freqs_cis[..., 1]

    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    out_real = x_real * freqs_cos - x_imag * freqs_sin
    out_imag = x_imag * freqs_cos + x_real * freqs_sin

    # Stack and reshape back
    x_out = mx.stack([out_real, out_imag], axis=-1)
    x_out = x_out.reshape(*x.shape)
    return x_out


def _update_kv_cache(
    k: mx.array, v: mx.array, inference_params: InferenceParams, layer_idx: int
) -> mx.array:
    """Update the KV cache for inference."""
    assert layer_idx in inference_params.key_value_memory_dict
    kv_cache, _ = inference_params.key_value_memory_dict[layer_idx]

    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + k.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + k.shape[1]

    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None

    # MLX doesn't have in-place operations, so we need to create a new array
    updated_cache = kv_cache.copy()

    # Update key values
    updated_cache = mx.array_scatter(
        updated_cache,
        mx.array([batch_start, sequence_start, 0]),
        (batch_end - batch_start, sequence_end - sequence_start, 1),
        k
    )

    # Update value values
    updated_cache = mx.array_scatter(
        updated_cache,
        mx.array([batch_start, sequence_start, 1]),
        (batch_end - batch_start, sequence_end - sequence_start, 1),
        v
    )

    # Return the relevant slice
    return updated_cache[batch_start:batch_end, :sequence_end, ...]


class MLXZonosBackbone(nn.Module):
    """MLX implementation of the transformer backbone."""
    supported_architectures = ["transformer"]

    def __init__(self, config: BackboneConfig):
        assert not config.ssm_cfg, "This backbone implementation only supports the Transformer model."
        super().__init__()
        self.config = config

        self.layers = [TransformerBlock(config, i) for i in range(config.n_layer)]
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=mx.bfloat16):
        """Allocate cache for inference."""
        head_dim = self.config.d_model // self.config.attn_cfg["num_heads"]
        self.freqs_cis = precompute_freqs_cis(16384, head_dim)

        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.layers)
        }

    def __call__(self, hidden_states: mx.array, inference_params: InferenceParams) -> mx.array:
        """Forward pass."""
        input_pos = mx.arange(0, hidden_states.shape[1])
        input_pos = input_pos + inference_params.lengths_per_sample.reshape(-1, 1)

        freqs_cis = self.freqs_cis[input_pos].repeat(hidden_states.shape[0], axis=0)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, inference_params, freqs_cis)

        return self.norm_f(hidden_states)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward network."""
    def __init__(self, config: BackboneConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mixer = Attention(config, layer_idx)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mlp = FeedForward(config)

        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // config.attn_cfg["num_heads"]

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=mx.bfloat16):
        """Allocate cache for inference in this layer."""
        return mx.zeros((batch_size, max_seqlen, 2, self.num_heads_kv, self.head_dim), dtype=dtype), None

    def __call__(self, x: mx.array, inference_params: InferenceParams, freqs_cis: mx.array) -> mx.array:
        """Forward pass through the transformer block."""
        x = x + self.mixer(self.norm(x), inference_params, freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """Multi-head attention with grouped query attention support."""
    def __init__(self, config: BackboneConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.attn_cfg["num_heads"]
        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // self.num_heads
        self.layer_idx = layer_idx

        total_head_dim = (self.num_heads + 2 * self.num_heads_kv) * self.head_dim
        self.in_proj = nn.Linear(config.d_model, total_head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False)

    def __call__(self, x: mx.array, inference_params: InferenceParams, freqs_cis: mx.array) -> mx.array:
        """Forward pass through the attention layer."""
        batch_size, seqlen, _ = x.shape

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_heads_kv * self.head_dim

        # Project input to query, key, value
        qkv = self.in_proj(x)
        q, k, v = mx.split(qkv, [q_size, kv_size, kv_size], axis=-1)

        # Reshape for attention computation
        q = q.reshape(batch_size, seqlen, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seqlen, self.num_heads_kv, self.head_dim)
        v = v.reshape(batch_size, seqlen, self.num_heads_kv, self.head_dim)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Update KV cache
        kv = _update_kv_cache(k, v, inference_params, self.layer_idx)
        k, v = mx.split(kv, indices_or_sections=2, axis=2)
        k = k.squeeze(2)  # Remove the KV dimension after splitting
        v = v.squeeze(2)

        # Prepare for attention
        q = q.transpose(0, 2, 1, 3)  # [batch, heads, seqlen, head_dim]
        k = k.transpose(0, 2, 1, 3)  # [batch, heads_kv, seqlen, head_dim]
        v = v.transpose(0, 2, 1, 3)  # [batch, heads_kv, seqlen, head_dim]

        # Implement scaled dot-product attention
        scale = 1.0 / mx.sqrt(mx.array(self.head_dim))

        # For GQA, repeat k and v heads
        if self.num_heads > self.num_heads_kv:
            repeats = self.num_heads // self.num_heads_kv
            k = k.repeat(repeats, axis=1)
            v = v.repeat(repeats, axis=1)

        # Compute attention scores
        attn_scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # [batch, heads, seqlen, seqlen]

        # Apply causal mask for autoregressive decoding
        if seqlen > 1:
            mask = mx.triu(mx.ones((seqlen, seqlen)), k=1) * -1e9
            attn_scores = attn_scores + mask.reshape(1, 1, seqlen, seqlen)

        # Apply softmax to get attention weights
        attn_weights = mx.softmax(attn_scores, axis=-1)

        # Apply attention to values
        y = mx.matmul(attn_weights, v)  # [batch, heads, seqlen, head_dim]

        # Reshape back
        y = y.transpose(0, 2, 1, 3)  # [batch, seqlen, heads, head_dim]
        y = y.reshape(batch_size, seqlen, q_size)

        # Final projection
        y = self.out_proj(y)
        return y


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 2 * config.attn_mlp_d_intermediate, bias=False)
        self.fc2 = nn.Linear(config.attn_mlp_d_intermediate, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through the feed-forward network."""
        x_proj = self.fc1(x)
        y, gate = mx.split(x_proj, 2, axis=-1)
        # SwiGLU activation
        gate_activated = mx.sigmoid(gate) * gate
        return self.fc2(y * gate_activated)