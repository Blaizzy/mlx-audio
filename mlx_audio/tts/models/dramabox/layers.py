from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .rope import LTXRopeType, apply_rotary_emb


def rms_norm(
    x: mx.array, weight: mx.array | None = None, eps: float = 1e-6
) -> mx.array:
    x_float = x.astype(mx.float32)
    output = x_float * mx.rsqrt(
        mx.mean(mx.square(x_float), axis=-1, keepdims=True) + eps
    )
    if weight is not None:
        output = output * weight
    return output.astype(x.dtype)


def gelu_approx(x: mx.array) -> mx.array:
    return (
        0.5
        * x
        * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3))))
    )


class GELUApprox(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        return gelu_approx(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = [
            GELUApprox(dim, inner_dim),
            nn.Identity(),
            nn.Linear(inner_dim, dim_out),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.net:
            x = layer(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType | str = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = query_dim if context_dim is None else context_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.rope_type = rope_type
        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_gate_logits = (
            nn.Linear(query_dim, heads, bias=True) if apply_gated_attention else None
        )
        self.to_out = [nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()]

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
        mask: mx.array | None = None,
        pe: tuple[mx.array, mx.array] | None = None,
        k_pe: tuple[mx.array, mx.array] | None = None,
        perturbation_mask: mx.array | None = None,
        all_perturbed: bool = False,
    ) -> mx.array:
        context = x if context is None else context
        value = self.to_v(context)

        if all_perturbed:
            out = value
        else:
            q = self.q_norm(self.to_q(x))
            k = self.k_norm(self.to_k(context))

            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

            batch = q.shape[0]
            q = q.reshape(batch, -1, self.heads, self.dim_head).transpose(0, 2, 1, 3)
            k = k.reshape(batch, -1, self.heads, self.dim_head).transpose(0, 2, 1, 3)
            v = value.reshape(batch, -1, self.heads, self.dim_head).transpose(
                0, 2, 1, 3
            )

            if mask is not None:
                if mask.ndim == 2:
                    mask = mask[None, None, :, :]
                elif mask.ndim == 3:
                    mask = mask[:, None, :, :]
                mask = mask.astype(q.dtype)

            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=mask
            )
            out = out.transpose(0, 2, 1, 3).reshape(
                batch, -1, self.heads * self.dim_head
            )

            if perturbation_mask is not None:
                out = out * perturbation_mask + value * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            gates = 2.0 * mx.sigmoid(self.to_gate_logits(x))
            out = out.reshape(out.shape[0], out.shape[1], self.heads, self.dim_head)
            out = out * gates[:, :, :, None]
            out = out.reshape(out.shape[0], out.shape[1], self.heads * self.dim_head)

        for layer in self.to_out:
            out = layer(out)
        return out
