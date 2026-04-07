# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
import time
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen2 import ModelArgs as MlxLmQwen2Args
from mlx_lm.models.qwen2 import Qwen2Model as MlxLmQwen2Backbone

from .config import Qwen2DecoderConfig

LayerCache = Union[Tuple[mx.array, mx.array], KVCache]


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

        # IMPORTANT: RoPE must use absolute positions, especially when KV cache is
        # used. Support both shared positions (L,) and per-batch positions (B, L).
        t = position_ids.astype(mx.float32)
        inv_freq = self._compute_inv_freq()  # (D/2,)

        if t.ndim == 1:
            freqs = mx.outer(t, inv_freq)  # (L, D/2)
            emb = mx.concatenate([freqs, freqs], axis=-1)  # (L, D)
        elif t.ndim == 2:
            # (B, L, 1) * (1, 1, D/2) -> (B, L, D/2)
            freqs = mx.expand_dims(t, axis=-1) * mx.reshape(inv_freq, (1, 1, -1))
            emb = mx.concatenate([freqs, freqs], axis=-1)  # (B, L, D)
        else:
            raise ValueError(
                f"Unsupported position_ids rank {t.ndim}; expected 1D or 2D."
            )

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

    # Keep RoPE phase computation in fp32 for stability, but run the
    # actual q/k rotation in the model dtype (bf16 on Apple Silicon).
    if cos.dtype != q.dtype:
        cos = cos.astype(q.dtype)
        sin = sin.astype(q.dtype)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
    """Repeat KV heads to match the number of query heads.

    Mirrors Hugging Face Qwen2 eager attention semantics:
    (B, H_kv, L, D) -> (B, H_q, L, D)
    """
    if n_rep == 1:
        return hidden_states
    return mx.repeat(hidden_states, n_rep, axis=1)


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
        cache: Optional[LayerCache] = None,
        timing_info: Optional[dict[str, float]] = None,
    ) -> Tuple[mx.array, LayerCache]:
        sync_timing = bool(timing_info is not None and timing_info.get("__sync__", False))
        attn_t0 = time.perf_counter() if timing_info is not None else None
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose query for attention: (B, L, H, D) -> (B, H, L, D)
        q = q.transpose(0, 2, 1, 3)

        # KV cache handling.
        # Prefer mlx_lm.KVCache to avoid O(T^2) concatenation during long decoding.
        if cache is not None and hasattr(cache, "update_and_fetch"):
            k_t = k.transpose(0, 2, 1, 3)  # (B, H_kv, L, D)
            v_t = v.transpose(0, 2, 1, 3)  # (B, H_kv, L, D)
            k_full, v_full = cache.update_and_fetch(k_t, v_t)
            new_cache: LayerCache = cache
        else:
            if cache is not None:
                # Legacy tuple cache path
                k_cache, v_cache = cache
                k = mx.concatenate([k_cache, k], axis=1)
                v = mx.concatenate([v_cache, v], axis=1)
            new_cache = (k, v)
            k_full = k.transpose(0, 2, 1, 3)  # (B, H_kv, L_total, D)
            v_full = v.transpose(0, 2, 1, 3)  # (B, H_kv, L_total, D)

        # Match Qwen2 eager attention semantics by explicitly repeating KV heads
        # before the attention kernel. Relying on implicit GQA handling here
        # produces numerically different outputs from the original model.
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k_full = repeat_kv(k_full, n_rep)
            v_full = repeat_kv(v_full, n_rep)

        # Scaled dot-product attention
        out = mx.fast.scaled_dot_product_attention(
            q, k_full, v_full, scale=self.scale, mask=mask
        )

        # Reshape output
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.o_proj(out)
        if sync_timing:
            mx.eval(out)
        if timing_info is not None:
            timing_info["attn"] = timing_info.get("attn", 0.0) + (
                time.perf_counter() - attn_t0
            )
        return out, new_cache


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
        cache: Optional[LayerCache] = None,
        timing_info: Optional[dict[str, float]] = None,
    ) -> Tuple[mx.array, LayerCache]:
        sync_timing = bool(timing_info is not None and timing_info.get("__sync__", False))
        # Self attention
        residual = x
        x = self.input_layernorm(x)
        h, new_cache = self.self_attn(
            x, cos, sin, mask, cache, timing_info=timing_info
        )
        x = residual + h

        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_t0 = time.perf_counter() if timing_info is not None else None
        h = self.mlp(x)
        if sync_timing:
            mx.eval(h)
        if timing_info is not None:
            timing_info["mlp"] = timing_info.get("mlp", 0.0) + (
                time.perf_counter() - mlp_t0
            )
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
        cache: Optional[List[LayerCache]] = None,
        is_causal: bool = True,
        return_layer_last_hidden: bool = False,
        timing_info: Optional[dict[str, float]] = None,
    ) -> Union[Tuple[mx.array, List[LayerCache]], Tuple[mx.array, List[LayerCache], List[mx.array]]]:
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

        B, L, _ = inputs_embeds.shape

        # Compute position offset(s) from cache.
        offsets: Optional[mx.array] = None
        if cache is not None and cache[0] is not None:
            first_cache = cache[0]
            if hasattr(first_cache, "offset"):
                raw_offset = first_cache.offset
                if isinstance(raw_offset, mx.array):
                    if raw_offset.ndim == 0:
                        offsets = mx.array([raw_offset.item()], dtype=mx.int32)
                    else:
                        offsets = raw_offset.astype(mx.int32)
                else:
                    offsets = mx.array([int(raw_offset)], dtype=mx.int32)
            else:
                # Legacy tuple cache path.
                offsets = mx.array([int(first_cache[0].shape[1])], dtype=mx.int32)

        # Position IDs: shared (L,) for scalar offsets or per-batch (B, L) when
        # cache offsets are batched (e.g., BatchKVCache).
        if offsets is None:
            position_ids = mx.arange(0, L, dtype=mx.int32)
        else:
            if offsets.ndim == 0:
                offsets = mx.expand_dims(offsets, axis=0)
            if int(offsets.shape[0]) == 1 and B > 1:
                offsets = mx.broadcast_to(offsets, (B,))
            pos = mx.arange(0, L, dtype=mx.int32)[None, :]
            position_ids = mx.expand_dims(offsets, axis=1) + pos

        # Get rotary embeddings
        cos, sin = self.rotary_emb(position_ids)
        if cos.ndim == 2:
            cos = cos[None, :, :]  # (1, L, D)
            sin = sin[None, :, :]  # (1, L, D)

        # Create causal mask if needed. This uses cache-specific masking when
        # available (KVCache or BatchKVCache), including left padding support.
        if mask is None and is_causal:
            cache0 = cache[0] if (cache is not None and len(cache) > 0) else None
            mask = create_attention_mask(inputs_embeds, cache0, return_array=True)

        h = inputs_embeds
        new_caches = []
        layer_last_hidden = [] if return_layer_last_hidden else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, c = layer(
                h,
                cos,
                sin,
                mask=mask,
                cache=layer_cache,
                timing_info=timing_info,
            )
            new_caches.append(c)
            if layer_last_hidden is not None:
                layer_last_hidden.append(h[:, -1, :].astype(mx.float32))

        if self.norm is not None:
            h = self.norm(h)
        if layer_last_hidden is not None:
            return h, new_caches, layer_last_hidden
        return h, new_caches

    def make_cache(self) -> List[KVCache]:
        """Create efficient KV caches for all decoder layers."""
        return [KVCache() for _ in self.layers]


class MlxLmQwen2Model(nn.Module):
    """Compatibility wrapper around `mlx_lm.models.qwen2.Qwen2Model`.

    Exposes the subset of the local Qwen2Model interface used by VibeVoice so we
    can A/B the optimized mlx_lm decode path under an env flag.
    """

    def __init__(self, config: Qwen2DecoderConfig, use_norm: bool = True):
        super().__init__()
        self.config = config
        args = MlxLmQwen2Args(
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        backbone = MlxLmQwen2Backbone(args)
        self.embed_tokens = backbone.embed_tokens
        self.layers = backbone.layers
        self.norm = backbone.norm if use_norm else None

    def __call__(
        self,
        inputs_embeds: Optional[mx.array] = None,
        input_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[LayerCache]] = None,
        is_causal: bool = True,
        return_layer_last_hidden: bool = False,
        timing_info: Optional[dict[str, float]] = None,
    ) -> Union[
        Tuple[mx.array, List[LayerCache]],
        Tuple[mx.array, List[LayerCache], List[mx.array]],
    ]:
        del timing_info
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None and is_causal:
            cache0 = cache[0] if len(cache) > 0 else None
            mask = create_attention_mask(inputs_embeds, cache0)

        h = inputs_embeds
        layer_last_hidden = [] if return_layer_last_hidden else None
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, mask, layer_cache)
            if layer_last_hidden is not None:
                layer_last_hidden.append(h[:, -1, :].astype(mx.float32))

        if self.norm is not None:
            h = self.norm(h)
        if layer_last_hidden is not None:
            return h, cache, layer_last_hidden
        return h, cache

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]


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
