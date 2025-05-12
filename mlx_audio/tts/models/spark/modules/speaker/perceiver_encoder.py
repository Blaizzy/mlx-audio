# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/lucidrains/naturalspeech2-pytorch/blob/659bec7f7543e7747e809e950cc2f84242fbeec7/naturalspeech2_pytorch/naturalspeech2_pytorch.py#L532

from collections import namedtuple
from functools import wraps

import mlx.core as mx
import mlx.nn as nn
from einops import rearrange, repeat


def exists(val):
    return val is not None


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


class Attend(nn.Module):
    def __init__(self, dropout=0.0, causal=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.mask = None

    def get_mask(self, n, device=None):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = mx.triu(mx.ones((n, n), dtype=mx.bool_), 1)
        self.mask = mask
        return mask

    def __call__(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        n = q.shape[-2]

        scale = q.shape[-1] ** -0.5

        # Handle different dimensions for k and v
        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        # similarity
        if k.ndim == 3:
            k = mx.expand_dims(k, axis=1)
            k = mx.broadcast_to(k, q.shape)

        if v.ndim == 3:
            v = mx.expand_dims(v, axis=1)
            v = mx.broadcast_to(v, q.shape[:-1] + (v.shape[-1],))

        # q: [b h i d], k: [b h j d]
        sim = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale

        # key padding mask
        if exists(mask):
            mask = mx.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))
            sim = mx.where(mask, sim, -1e9)

        # causal mask
        if self.causal:
            causal_mask = self.get_mask(n)
            sim = mx.where(causal_mask, -1e9, sim)

        # attention
        attn = mx.softmax(sim, axis=-1)

        if self.dropout > 0 and self.training:
            attn = self.attn_dropout(attn)

        # aggregate values
        out = mx.matmul(attn, v)

        return out


def Sequential(*mods):
    return nn.Sequential(*[mod for mod in mods if exists(mod)])


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True, dim_cond=None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim**0.5
        self.gamma = mx.ones((dim,)) if scale else None

    def __call__(self, x, cond=None):
        def normalize(input, p=2.0, dim=1, eps=1e-12):
            norm = mx.power(
                mx.sum(mx.power(mx.abs(input), p), axis=dim, keepdims=True), 1 / p
            )
            return input / mx.maximum(norm, eps)

        gamma = default(self.gamma, 1)
        out = normalize(x, dim=-1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = mx.split(self.to_gamma_beta(cond), 2, axis=-1)
        gamma = mx.expand_dims(gamma, axis=1)
        beta = mx.expand_dims(beta, axis=1)
        return out * gamma + beta


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def __call__(self, x):
        causal_padded_x = mx.pad(x, [(0, 0), (0, 0), (self.causal_padding, 0)])
        return self.conv(causal_padded_x)


class GEGLU(nn.Module):
    def __call__(self, x):
        x, gate = mx.split(x, 2, axis=-1)
        return nn.gelu(gate) * x


def FeedForward(dim, mult=4, causal_conv=False):
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = [
            lambda x: mx.transpose(x, (0, 2, 1)),  # b n d -> b d n
            CausalConv1d(dim_inner, dim_inner, 3),
            lambda x: mx.transpose(x, (0, 2, 1)),  # b d n -> b n d
        ]

        return [
            nn.Linear(dim, dim_inner * 2),
            GEGLU(),
            conv,
            nn.Linear(dim_inner, dim),
        ]
    else:
        return [
            nn.Linear(dim, dim_inner * 2),
            GEGLU(),
            nn.Linear(dim_inner, dim),
        ]


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        dropout=0.0,
        cross_attn_include_queries=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal=causal, dropout=dropout)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def __call__(self, x, context=None, mask=None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = mx.concatenate([x, context], axis=-2)

        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = mx.split(kv, 2, axis=-1)

        # Reshape for multi-head attention
        q = mx.reshape(q, (q.shape[0], q.shape[1], h, -1))
        q = mx.transpose(q, (0, 2, 1, 3))  # b n (h d) -> b h n d

        k = mx.reshape(k, (k.shape[0], k.shape[1], h, -1))
        k = mx.transpose(k, (0, 2, 1, 3))  # b n (h d) -> b h n d

        v = mx.reshape(v, (v.shape[0], v.shape[1], h, -1))
        v = mx.transpose(v, (0, 2, 1, 3))  # b n (h d) -> b h n d

        out = self.attend(q, k, v, mask=mask)

        out = mx.transpose(out, (0, 2, 1, 3))  # b h n d -> b n h d
        out = mx.reshape(out, (out.shape[0], out.shape[1], -1))  # b n h d -> b n (h d)

        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=2,
        dim_context=None,
        num_latents=32,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = (
            nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        )

        self.latents = mx.random.normal(shape=(num_latents, dim), scale=0.02)

        self.layers = []
        for _ in range(depth):
            self.layers.append(
                [
                    Attention(
                        dim=dim,
                        dim_head=dim_head,
                        heads=heads,
                        cross_attn_include_queries=True,
                    ),
                    FeedForward(dim=dim, mult=ff_mult),
                ]
            )

        self.norm = RMSNorm(dim)

    def __call__(self, x, mask=None):
        batch = x.shape[0]

        x = self.proj_context(x)

        latents = mx.broadcast_to(self.latents, (batch,) + self.latents.shape)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            skip_connect = latents
            for module in ff:
                latents = module(latents)

            latents = skip_connect + latents

        return self.norm(latents)


if __name__ == "__main__":
    from mlx.utils import tree_flatten

    model = PerceiverResampler(dim=256, dim_context=80)
    x = mx.random.normal(shape=(8, 200, 80))
    out = model(x)
    print("Output shape:", out.shape)  # [8, 32, 80]

    # Count parameters for MLX model
    num_params = 0

    weights = dict(tree_flatten(model.parameters()))

    for k, v in weights.items():
        num_params += v.size
    print("{} M".format(num_params / 1e6))
