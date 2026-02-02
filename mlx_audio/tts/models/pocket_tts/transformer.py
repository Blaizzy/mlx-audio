import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from .rope import RotaryEmbedding


def create_additive_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float):
        super().__init__()
        self.scale = mx.full((channels,), init)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale


class StreamingMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, rope: RotaryEmbedding):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.rope = rope
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def __call__(self, query: mx.array, cache: KVCache | None):
        b, t, _ = query.shape
        projected = self.in_proj(query)
        projected = projected.reshape(b, t, 3, self.num_heads, self.head_dim)
        q = projected[:, :, 0]
        k = projected[:, :, 1]
        v = projected[:, :, 2]

        offset = 0 if cache is None else cache.offset
        if self.rope is not None:
            q, k = self.rope(q, k, offset)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is None:
            k_full, v_full = k, v
        else:
            k_full, v_full = cache.update_and_fetch(k, v)

        mask = create_additive_causal_mask(t, offset=offset).astype(query.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q, k_full, v_full, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.embed_dim)
        return self.out_proj(out)


class StreamingTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        rope: RotaryEmbedding,
        layer_scale: float | None = None,
    ):
        super().__init__()
        self.self_attn = StreamingMultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, rope=rope
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        if layer_scale is None:
            self.layer_scale_1 = None
            self.layer_scale_2 = None
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale)
            self.layer_scale_2 = LayerScale(d_model, layer_scale)

    def _apply_scale(self, x: mx.array, layer_scale: LayerScale | None) -> mx.array:
        return x if layer_scale is None else layer_scale(x)

    def __call__(self, x: mx.array, cache: KVCache | None) -> mx.array:
        attn_out = self.self_attn(self.norm1(x), cache)
        x = x + self._apply_scale(attn_out, self.layer_scale_1)
        ff = self.linear2(nn.gelu(self.linear1(self.norm2(x))))
        x = x + self._apply_scale(ff, self.layer_scale_2)
        return x


class StreamingTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        max_period: float = 10000.0,
        layer_scale: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = d_model // num_heads
        self.rope = RotaryEmbedding(max_period=max_period)
        self.layers = [
            StreamingTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                rope=self.rope,
                layer_scale=layer_scale,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array, cache: list[KVCache] | None):
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, layer_cache)
        return x

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]
