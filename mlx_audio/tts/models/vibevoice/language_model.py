"""Qwen2-based language model backbone for VibeVoice."""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import Qwen2DecoderConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self, dim: int, max_position_embeddings: int = 8192, base: float = 1000000.0
    ):
        super().__init__()
        self._dim = dim
        self._max_position_embeddings = max_position_embeddings
        self._base = base

    def _compute_inv_freq(self) -> mx.array:
        """Compute inverse frequencies on the fly."""
        return 1.0 / (
            self._base ** (mx.arange(0, self._dim, 2, dtype=mx.float32) / self._dim)
        )

    def __call__(self, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        """Compute cos and sin for rotary embeddings.

        Args:
            position_ids: Position indices, shape (L,) or (B, L)

        Returns:
            Tuple of (cos, sin) each of shape matching positions x dim
        """
        # Ensure position_ids is at least 1D
        if position_ids.ndim == 0:
            position_ids = mx.expand_dims(position_ids, 0)

        # IMPORTANT: RoPE must use *absolute* positions, especially when KV cache is used.
        # Our model passes `position_ids = arange(offset, offset+L)`.
        if position_ids.ndim > 1:
            # Assume all batch rows share the same positions
            t = position_ids[0].astype(mx.float32)
        else:
            t = position_ids.astype(mx.float32)

        inv_freq = self._compute_inv_freq()
        freqs = mx.outer(t, inv_freq)  # (L, dim/2)

        # Concatenate to get full dimension
        emb = mx.concatenate([freqs, freqs], axis=-1)  # (L, dim)

        cos = mx.cos(emb)
        sin = mx.sin(emb)

        return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor, shape (B, L, H, D)
        k: Key tensor, shape (B, L, H_kv, D)
        cos: Cosine embeddings, shape (B, L, 1, D)
        sin: Sine embeddings, shape (B, L, 1, D)

    Returns:
        Tuple of rotated (q, k)
    """
    # Expand dims for head dimension
    cos = cos[:, :, None, :]  # (B, L, 1, D)
    sin = sin[:, :, None, :]  # (B, L, 1, D)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with grouped query attention support."""

    def __init__(self, config: Qwen2DecoderConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = (
            config.head_dim if config.head_dim else config.hidden_size // self.num_heads
        )
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache handling
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        new_cache = (k, v)

        # Transpose for attention: (B, L, H, D) -> (B, H, L, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        # Reshape output
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(out), new_cache


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: Qwen2DecoderConfig):
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


class DecoderLayer(nn.Module):
    """A single transformer decoder layer."""

    def __init__(self, config: Qwen2DecoderConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # Self attention
        residual = x
        x = self.input_layernorm(x)
        h, new_cache = self.self_attn(x, cos, sin, mask, cache)
        x = residual + h

        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        h = self.mlp(x)
        x = residual + h

        return x, new_cache


class SpeechConnector(nn.Module):
    """Connector to project speech latents to LM hidden size."""

    def __init__(self, input_dim: int, output_dim: int, eps: float = 1e-6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = RMSNorm(output_dim, eps=eps)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def __call__(self, features: mx.array) -> mx.array:
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class BinaryClassifier(nn.Module):
    """Binary classifier for TTS end-of-speech detection."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Qwen2Model(nn.Module):
    """Qwen2 transformer model for text and speech processing."""

    def __init__(self, config: Qwen2DecoderConfig, use_norm: bool = True):
        super().__init__()
        self.config = config
        self.use_norm = use_norm

        if config.vocab_size > 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = None

        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]

        # Only add norm if requested (base LM doesn't have it)
        if use_norm:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = None

        # Rotary embeddings
        head_dim = (
            config.head_dim
            if config.head_dim
            else config.hidden_size // config.num_attention_heads
        )
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def __call__(
        self,
        inputs_embeds: Optional[mx.array] = None,
        input_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        is_causal: bool = True,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        """Forward pass.

        Args:
            inputs_embeds: Embedded inputs, shape (B, L, D)
            input_ids: Token IDs, shape (B, L) - used if inputs_embeds is None
            mask: Attention mask
            cache: KV cache from previous steps
            is_causal: Whether to apply causal masking

        Returns:
            Tuple of (hidden_states, new_cache)
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        _, L, _ = inputs_embeds.shape

        # Compute position offset from cache
        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0][0].shape[1]

        # Position IDs
        position_ids = mx.arange(offset, offset + L, dtype=mx.int32)

        # Get rotary embeddings
        cos, sin = self.rotary_emb(position_ids)
        cos = cos[None, :, :]  # (1, L, D)
        sin = sin[None, :, :]  # (1, L, D)

        # Create causal mask if needed.
        # If we have cached KV, key length is (offset + L). The mask must match that.
        if mask is None and is_causal and L > 1:
            # pylint: disable=unsubscriptable-object
            k_len = offset + L
            q_pos = mx.expand_dims(
                mx.arange(offset, offset + L, dtype=mx.int32), axis=1
            )  # (L, 1)
            k_pos = mx.expand_dims(
                mx.arange(0, k_len, dtype=mx.int32), axis=0
            )  # (1, K)
            allow = q_pos >= k_pos  # (L, K)
            neg_inf = mx.array(float("-inf"), dtype=mx.float32)
            causal_mask = mx.where(allow, mx.array(0.0, dtype=mx.float32), neg_inf)
            mask = causal_mask[None, None, :, :]  # (1, 1, L, K)

        h = inputs_embeds
        new_caches = []

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, c = layer(h, cos, sin, mask=mask, cache=layer_cache)
            new_caches.append(c)

        if self.norm is not None:
            h = self.norm(h)
        return h, new_caches


class VibeVoiceLanguageModel(nn.Module):
    """Combined language model for VibeVoice with text LM and TTS LM portions.

    The model is split into:
    - language_model: Lower transformer layers for text encoding
    - tts_language_model: Upper transformer layers for TTS generation
    """

    def __init__(
        self, config: Qwen2DecoderConfig, tts_backbone_num_hidden_layers: int = 20
    ):
        super().__init__()
        self.config = config
        self.tts_backbone_num_hidden_layers = tts_backbone_num_hidden_layers

        # Calculate layer split
        lm_num_layers = config.num_hidden_layers - tts_backbone_num_hidden_layers

        # Create base LM config
        lm_config = Qwen2DecoderConfig(
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            num_hidden_layers=lm_num_layers,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            head_dim=config.head_dim,
        )

        # Create TTS LM config
        tts_config = Qwen2DecoderConfig(
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            num_hidden_layers=tts_backbone_num_hidden_layers,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=0,  # TTS LM doesn't need token embeddings
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            head_dim=config.head_dim,
        )

        # Initialize models
        self.language_model = Qwen2Model(lm_config)
        self.tts_language_model = Qwen2Model(tts_config)

        # Remove the norm from base LM (it's applied in TTS LM)
        self.language_model.norm = None

        # TTS input type embeddings (text=1, speech=0)
        self.tts_input_types = nn.Embedding(2, config.hidden_size)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the token embedding layer."""
        return self.language_model.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding):
        """Set the token embedding layer."""
        self.language_model.embed_tokens = embeddings
