import math
from typing import Any, Optional

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.llama import ModelArgs


class Llama3ScaledRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 500_000.0,
        scale_factor: float = 32.0,
        low_freq_factor: int = 1,
        high_freq_factor: int = 4,
        old_context_len: int = 8192,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim = dim
        self.d2 = dim // 2
        self.base = float(base)
        self.max_seq_len = int(max_seq_len)

        self.scale_factor = float(scale_factor)
        self.low_freq_factor = int(low_freq_factor)
        self.high_freq_factor = int(high_freq_factor)
        self.old_context_len = int(old_context_len)

        self._cos_f32 = None
        self._sin_f32 = None
        self._cos_by_dtype = {}
        self._sin_by_dtype = {}

        self.rope_init()

    def rope_init(self):
        freqs = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        theta = self._apply_scaling(freqs)
        seq = mx.arange(self.max_seq_len, dtype=mx.float32).reshape(-1, 1)
        idx_theta = seq * theta.reshape(1, -1)

        self._cos_f32 = mx.cos(idx_theta)
        self._sin_f32 = mx.sin(idx_theta)

        self._cos_by_dtype.clear()
        self._sin_by_dtype.clear()

    def _apply_scaling(self, freqs: mx.array) -> mx.array:
        wavelen = 2.0 * math.pi / freqs  # (D/2,)

        low = self.old_context_len / self.low_freq_factor
        high = self.old_context_len / self.high_freq_factor

        smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        smooth = mx.clip(smooth, 0.0, 1.0)

        scaled = freqs / self.scale_factor
        blended = (1.0 - smooth) * scaled + smooth * freqs

        cond_A = wavelen < high
        cond_B = wavelen > low

        out = mx.where(cond_A, freqs, mx.where(cond_B, scaled, blended))
        return out.astype(freqs.dtype)

    def _get_cache(self, dtype, seq_len: int, offset: Optional[int]):
        start = 0 if (offset is None) else int(offset)
        end = start + int(seq_len)
        assert end <= self.max_seq_len, "RoPE cache length exceeded"

        if dtype == self._cos_f32.dtype:
            cos = self._cos_f32[start:end]
            sin = self._sin_f32[start:end]
        else:
            if dtype not in self._cos_by_dtype:
                self._cos_by_dtype[dtype] = self._cos_f32.astype(dtype)
                self._sin_by_dtype[dtype] = self._sin_f32.astype(dtype)
            cos = self._cos_by_dtype[dtype][start:end]
            sin = self._sin_by_dtype[dtype][start:end]

        return cos.reshape(1, -1, 1, self.d2), sin.reshape(1, -1, 1, self.d2)

    def __call__(self, x: mx.array, *, offset: Optional[int]) -> mx.array:
        B, S, H, D = x.shape
        assert D == self.dim

        x_dtype = x.dtype
        x_even = x[..., 0::2]  # (B, S, H, D/2)
        x_odd = x[..., 1::2]  # (B, S, H, D/2)

        cos, sin = self._get_cache(x_dtype, S, offset)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_odd * cos + x_even * sin
        out = mx.stack([out_even, out_odd], axis=-1).reshape(B, S, H, D)
        return out  # already in x's dtype


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads or n_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = Llama3ScaledRoPE(
            self.head_dim,
            base=args.rope_theta,
            scale_factor=args.rope_scaling.get("factor", 1.0),
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        b, s_x, _ = x.shape
        y = x

        s_y = y.shape[1] if y is not None else 0

        q = self.q_proj(x)

        q_per_kv = self.n_heads // self.n_kv_heads
        q = q.reshape(b, s_x, self.n_kv_heads * q_per_kv, self.head_dim)

        if self.rope is not None:
            q = self.rope(q, offset=cache.offset if cache else 0)

        q = q.swapaxes(1, 2)

        k = self.k_proj(y)
        v = self.v_proj(y)

        k = k.reshape(b, s_y, -1, self.head_dim)
        v = v.reshape(b, s_y, -1, self.head_dim)
        if self.rope is not None:
            k = self.rope(k, offset=cache.offset if cache else 0)

        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)

        if cache:
            k, v = cache.update_and_fetch(k, v)

        if self.n_heads != self.n_kv_heads:
            k = mx.expand_dims(k, axis=2)
            v = mx.expand_dims(v, axis=2)

            k_expand_shape = (b, self.n_kv_heads, q_per_kv) + k.shape[3:]
            v_expand_shape = (b, self.n_kv_heads, q_per_kv) + v.shape[3:]

            k = mx.broadcast_to(k, k_expand_shape)
            v = mx.broadcast_to(v, v_expand_shape)

            k = k.reshape(b, self.n_kv_heads * q_per_kv, *k.shape[3:])
            v = v.reshape(b, self.n_kv_heads * q_per_kv, *v.shape[3:])

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )

        output = output.swapaxes(1, 2).reshape(b, s_x, -1)
        return self.o_proj(output)
