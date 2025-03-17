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
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.is_cache_built = False
        self.rope_init()

    def rope_init(self):
        freqs = 1.0 / (
            self.base
            ** (
                mx.arange(0, self.dim, 2)[: (self.dim // 2)].astype(mx.float32)
                / self.dim
            )
        )

        theta = self.apply_scaling(
            freqs,
            self.scale_factor,
            self.low_freq_factor,
            self.high_freq_factor,
            self.old_context_len,
        )
        self._theta = theta
        self.build_rope_cache(self.max_seq_len)
        self.is_cache_built = True

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = mx.arange(max_seq_len, dtype=self._theta.dtype)
        idx_theta = mx.einsum("i, j -> ij", seq_idx, self._theta).astype(mx.float32)
        cache = mx.stack([mx.cos(idx_theta), mx.sin(idx_theta)], axis=-1)
        self._cache = cache

    def apply_scaling(
        self,
        freqs: mx.array,
        scale_factor: float,
        low_freq_factor: int,
        high_freq_factor: int,
        old_context_len: int,
    ):
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return mx.array(new_freqs, dtype=freqs.dtype)

    def __call__(self, x: mx.array, *, offset: int) -> mx.array:
        if not self.is_cache_built:
            raise RuntimeError(
                "RoPE cache is not built. Please call rope_init() first."
            )

        seq_len = x.shape[1]
        rope_cache = (
            self._cache[:seq_len]
            if offset is None
            else self._cache[None, offset : offset + seq_len]
        )
        xshaped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.reshape(-1, xshaped.shape[1], 1, xshaped.shape[3], 2)

        x_out = mx.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out = x_out.flatten(3)
        return x_out.astype(x.dtype)


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
            q_per_kv = self.n_heads // self.n_kv_heads

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
