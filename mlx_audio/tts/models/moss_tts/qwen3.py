# pyright: reportMissingImports=false

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache


def _get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    v = cfg.get(key, default)
    return default if v is None else v


def create_additive_causal_mask(N: int, offset: int = 0) -> mx.array:
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


class Qwen3Attention(nn.Module):
    def __init__(self, config: Dict[str, Any], layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = int(_get(config, "hidden_size", 4096))
        self.num_heads = int(_get(config, "num_attention_heads", 32))
        self.num_kv_heads = int(_get(config, "num_key_value_heads", 8))
        self.head_dim = int(
            _get(config, "head_dim", self.hidden_size // self.num_heads)
        )
        self.scale = self.head_dim**-0.5

        attn_bias = bool(_get(config, "attention_bias", False))

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=attn_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=attn_bias
        )

        eps = float(_get(config, "rms_norm_eps", 1e-6))
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps)

        rope_theta = float(_get(config, "rope_theta", 10000.0))
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_theta)

    def _apply_rope_fp32(
        self, x: mx.array, *, offset: Optional[int] = None
    ) -> mx.array:
        orig_dtype = x.dtype
        x_fp32 = x.astype(mx.float32)
        if offset is None:
            out = self.rope(x_fp32)
        else:
            out = self.rope(x_fp32, offset=offset)
        return out.astype(orig_dtype)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, S, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, S, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, S, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        if cache is not None:
            offset = cache.offset
            q = self._apply_rope_fp32(q, offset=offset)
            k = self._apply_rope_fp32(k, offset=offset)
        else:
            q = self._apply_rope_fp32(q)
            k = self._apply_rope_fp32(k)

        if cache is not None:
            if S > 1 and hasattr(self.q_proj, "bits"):
                k_full = None
                v_full = None
                for t in range(S):
                    k_full, v_full = cache.update_and_fetch(
                        k[:, :, t : t + 1, :],
                        v[:, :, t : t + 1, :],
                    )
                if k_full is None or v_full is None:
                    raise RuntimeError(
                        "KV cache update failed during quantized prefill"
                    )
                k, v = k_full, v_full
            else:
                k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, S, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        hidden_size = int(_get(config, "hidden_size", 4096))
        intermediate_size = int(_get(config, "intermediate_size", 12288))

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Dict[str, Any], layer_idx: int):
        super().__init__()
        eps = float(_get(config, "rms_norm_eps", 1e-6))
        hidden_size = int(_get(config, "hidden_size", 4096))

        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen3Model(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        self.hidden_size = int(_get(config, "hidden_size", 4096))
        self.vocab_size = int(_get(config, "vocab_size", 155648))
        self.num_hidden_layers = int(_get(config, "num_hidden_layers", 36))

        eps = float(_get(config, "rms_norm_eps", 1e-6))

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = [
            Qwen3DecoderLayer(config, i) for i in range(self.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(self.hidden_size, eps=eps)

    def __call__(
        self,
        inputs_embeds: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
    ) -> Tuple[mx.array, Optional[List[KVCache]]]:
        B, S, _ = inputs_embeds.shape

        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset

        attn_mask: Optional[Any] = None
        use_native_causal = attention_mask is None
        if not use_native_causal and (S > 1 or offset > 0):
            attn_mask = create_additive_causal_mask(S, offset=offset).astype(
                inputs_embeds.dtype
            )

        if use_native_causal:
            if S > 1:
                attn_mask = "causal"
            else:
                attn_mask = None

        if attention_mask is not None:
            key_len = offset + S
            key_mask = attention_mask
            if key_mask.ndim != 2:
                raise ValueError("attention_mask must be rank-2 [B, T]")
            if key_mask.shape[1] == S and key_len != S:
                pad = mx.ones((B, offset), dtype=key_mask.dtype)
                key_mask = mx.concatenate([pad, key_mask], axis=1)
            if key_mask.shape[1] != key_len:
                raise ValueError(
                    f"attention_mask length {key_mask.shape[1]} != key_len {key_len}"
                )

            pad_mask = (~key_mask.astype(mx.bool_)).astype(inputs_embeds.dtype) * -1e9
            pad_mask = pad_mask.reshape(B, 1, 1, key_len)

            if attn_mask is None:
                attn_mask = pad_mask
            else:
                attn_mask = attn_mask.reshape(1, 1, S, key_len) + pad_mask

        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, mask=attn_mask, cache=layer_cache)

        x = self.norm(x)
        return x, cache

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]
