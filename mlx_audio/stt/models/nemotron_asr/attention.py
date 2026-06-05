"""Relative-positional multi-head attention for Nemotron FastConformer.

Same rel_pos math as NeMo/Parakeet (pos_bias_u/v, linear_pos, untied per layer),
but: no bias on q/k/v/out, and an additive float mask (0 / -inf) of shape
(Tq, Tk) for the chunked_limited context restriction.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int):
        super().__init__()
        self.n_head = n_head
        self.n_feat = n_feat
        self.head_dim = n_feat // n_head
        self.scale = self.head_dim**-0.5

        self.linear_q = nn.Linear(n_feat, n_feat, bias=False)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=False)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=False)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=False)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        self.pos_bias_u = mx.zeros((n_head, self.head_dim))
        self.pos_bias_v = mx.zeros((n_head, self.head_dim))

    def rel_shift(self, x: mx.array) -> mx.array:
        B, H, Tq, pos_len = x.shape
        x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(1, 0)])
        x = x.reshape(B, H, pos_len + 1, Tq)
        x = x[:, :, 1:, :].reshape(B, H, Tq, pos_len)
        return x

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        p = self.linear_pos(pos_emb)

        B, T, _ = q.shape
        _, pos_len, _ = p.shape

        q = q.reshape(B, T, self.n_head, self.head_dim)
        q_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_v = (q + self.pos_bias_v).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        p = p.reshape(1, pos_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)[:, :, :, : k.shape[-2]] * self.scale

        if mask is not None:
            matrix_bd = matrix_bd + mask  # additive (0 / -inf), shape (1,1,Tq,Tk)

        o = mx.fast.scaled_dot_product_attention(
            q_u, k, v, scale=self.scale, mask=matrix_bd
        )
        o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.linear_out(o)

    def stream(self, q_in: mx.array, kv_in: mx.array, pos_emb: mx.array) -> mx.array:
        """Cache-aware step: q_in (B,c,d) attends to kv_in (B,L,d), no mask
        (the L-window IS the allowed context). pos_emb is for length L (2L-1)."""
        q = self.linear_q(q_in)
        k = self.linear_k(kv_in)
        v = self.linear_v(kv_in)
        p = self.linear_pos(pos_emb)
        B, c, _ = q.shape
        L = kv_in.shape[1]
        pos_len = p.shape[1]
        q = q.reshape(B, c, self.n_head, self.head_dim)
        q_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_v = (q + self.pos_bias_v).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        p = p.reshape(1, pos_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)[:, :, :, :L] * self.scale
        o = mx.fast.scaled_dot_product_attention(
            q_u, k, v, scale=self.scale, mask=matrix_bd
        )
        return self.linear_out(o.transpose(0, 2, 1, 3).reshape(B, c, -1))


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._build(max_len)

    def _build(self, max_len: int):
        positions = mx.arange(max_len - 1, -max_len, -1, dtype=mx.int32)
        positions = mx.expand_dims(positions, axis=1).astype(mx.float32)
        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = mx.zeros((2 * max_len - 1, self.d_model), dtype=mx.float32)
        pe[:, 0::2] = mx.sin(positions * div_term)
        pe[:, 1::2] = mx.cos(positions * div_term)
        self._pe = mx.expand_dims(pe, axis=0)
        self.max_len = max_len

    def __call__(self, x: mx.array) -> mx.array:
        return self.pos_emb_for(x.shape[1], x.dtype)

    def pos_emb_for(self, length: int, dtype=mx.float32) -> mx.array:
        """Positional embedding for a window of `length` frames (2*length-1)."""
        if length > self.max_len:
            self._build(length + 1)
        center = self._pe.shape[1] // 2
        return self._pe[:, center - (length - 1) : center + length].astype(dtype)
