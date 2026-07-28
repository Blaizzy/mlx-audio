"""Relative-position multi-head attention for the FastConformer encoder.

Matches NeMo's ``RelPositionMultiHeadAttention`` (Transformer-XL style) — the same
convention Parakeet uses — but takes an *additive* attention mask so the encoder can
inject the cache-aware ``chunked_limited`` look-ahead pattern, and supports the
``use_bias=False`` projections this model was trained with.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int, bias: bool = False):
        super().__init__()
        assert n_feat % n_head == 0
        self.n_head = n_head
        self.n_feat = n_feat
        self.head_dim = n_feat // n_head
        self.scale = self.head_dim**-0.5

        self.linear_q = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        # Per-layer (untied) position biases; populated from the checkpoint.
        self.pos_bias_u = mx.zeros((self.n_head, self.head_dim))
        self.pos_bias_v = mx.zeros((self.n_head, self.head_dim))

    def rel_shift(self, x: mx.array) -> mx.array:
        B, H, Tq, pos_len = x.shape
        padding = [(0, 0)] * (x.ndim - 1) + [(1, 0)]
        x = mx.pad(x, padding)
        x = x.reshape(B, H, pos_len + 1, Tq)
        x = x[:, :, 1:, :]
        x = x.reshape(B, H, Tq, pos_len)
        return x

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        p = self.linear_pos(pos_emb)

        batch, seq, _ = q.shape
        _, pos_len, _ = p.shape

        q = q.reshape(batch, seq, self.n_head, self.head_dim)
        q_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_v = (q + self.pos_bias_v).transpose(0, 2, 1, 3)

        k = k.reshape(batch, seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        p = p.reshape(batch, pos_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        # matrix_bd: position-dependent term, scaled and rel-shifted to (B, H, Tq, Tk).
        matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, : k.shape[-2]] * self.scale

        if mask is not None:
            # mask is additive (0 where visible, -inf where blocked), broadcastable
            # to (B, H, Tq, Tk). It is applied at the same (post-scale) point SDPA
            # adds it, so -inf reliably zeroes out blocked positions.
            matrix_bd = matrix_bd + mask

        o = mx.fast.scaled_dot_product_attention(
            q_u, k, v, scale=self.scale, mask=matrix_bd
        )
        o = o.transpose(0, 2, 1, 3).reshape(batch, seq, self.n_feat)
        return self.linear_out(o)

    def stream(self, q_in: mx.array, kv_in: mx.array, pos_emb: mx.array) -> mx.array:
        """Cache-aware step: q_in (B,c,d) attends to kv_in (B,L,d), no mask (the
        L-window IS the allowed context). pos_emb is for length L (2L-1)."""
        q = self.linear_q(q_in)
        k = self.linear_k(kv_in)
        v = self.linear_v(kv_in)
        p = self.linear_pos(pos_emb)
        batch, c, _ = q.shape
        ksz = kv_in.shape[1]
        pos_len = p.shape[1]
        q = q.reshape(batch, c, self.n_head, self.head_dim)
        q_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_v = (q + self.pos_bias_v).transpose(0, 2, 1, 3)
        k = k.reshape(batch, ksz, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, ksz, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        p = p.reshape(1, pos_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)[:, :, :, :ksz] * self.scale
        o = mx.fast.scaled_dot_product_attention(
            q_u, k, v, scale=self.scale, mask=matrix_bd
        )
        return self.linear_out(o.transpose(0, 2, 1, 3).reshape(batch, c, -1))


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, scale_input: bool = False):
        assert d_model % 2 == 0 and max_len > 0
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.scale = math.sqrt(self.d_model) if scale_input else 1.0
        self.calculate_pe()

    def calculate_pe(self):
        positions = mx.arange(self.max_len - 1, -self.max_len, -1, dtype=mx.int32)
        positions = mx.expand_dims(positions, axis=1).astype(mx.float32)
        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = mx.zeros((2 * self.max_len - 1, self.d_model), dtype=mx.float32)
        pe[:, 0::2] = mx.sin(positions * div_term)
        pe[:, 1::2] = mx.cos(positions * div_term)
        self._pe = mx.expand_dims(pe, axis=0).astype(mx.float32)
        mx.eval(self._pe)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        input_len = x.shape[1]
        if input_len > self.max_len:
            self.max_len = input_len + 1
            self.calculate_pe()

        x = x * self.scale
        buffer_len = self._pe.shape[1]
        start_idx = buffer_len // 2 - (input_len - 1)
        end_idx = buffer_len // 2 + (input_len - 1) + 1
        pos_emb = self._pe[:, start_idx:end_idx].astype(x.dtype)
        return x, pos_emb

    def pos_emb_for(self, length: int, dtype: mx.Dtype = mx.float32) -> mx.array:
        """Positional embedding for a window of ``length`` frames (2*length-1).

        Used by cache-aware streaming where queries (the new chunk) attend to a
        longer key window (cache + chunk).
        """
        if length > self.max_len:
            self.max_len = length + 1
            self.calculate_pe()
        center = self._pe.shape[1] // 2
        return self._pe[:, center - (length - 1) : center + length].astype(dtype)
