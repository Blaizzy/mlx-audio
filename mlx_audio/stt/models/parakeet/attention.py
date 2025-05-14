import math

import mlx.core as mx
import mlx.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        bias=True,
    ):
        super().__init__()

        self.n_head = n_head
        self.head_dim = n_feat // n_head
        self.scale = self.head_dim**-0.5

        self.linear_q = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=bias)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        batch, q_seq, _ = q.shape
        _, k_seq, _ = k.shape

        q = q.reshape(batch, q_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        if cache:
            k, v = cache.update_and_fetch(k, v)

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(batch, q_seq, self.n_feat)

        return self.linear_out(o)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        bias: bool = True,
        pos_bias_u: mx.array | None = None,
        pos_bias_v: mx.array | None = None,
    ):
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            bias=bias,
        )

        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        if pos_bias_u is None:
            self._pos_bias_u_init = mx.zeros((self.n_head, self.head_dim))
        else:
            self._pos_bias_u_init = pos_bias_u

        if pos_bias_v is None:
            self._pos_bias_v_init = mx.zeros((self.n_head, self.head_dim))
        else:
            self._pos_bias_v_init = pos_bias_v

        self.pos_bias_u = self._pos_bias_u_init
        self.pos_bias_v = self._pos_bias_v_init

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
        q: mx.array,
        k: mx.array,
        v: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        if pos_emb is None:
            raise ValueError("pos_emb is necessary!")

        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        p = self.linear_pos(pos_emb)  # p stands for position

        batch, q_seq, _ = q.shape
        _, k_seq, _ = k.shape
        _, pos_len, _ = p.shape

        q = q.reshape(batch, q_seq, self.n_head, self.head_dim)
        q_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_v = (q + self.pos_bias_v).transpose(0, 2, 1, 3)

        k = k.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        p = p.reshape(batch, pos_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, : k.shape[-2]] * self.scale

        if mask is not None:
            mask = mx.expand_dims(mask, 0)
            matrix_bd[mask] = -mx.inf

        o = mx.fast.scaled_dot_product_attention(
            q_u, k, v, scale=self.scale, mask=matrix_bd
        )
        o = o.transpose(0, 2, 1, 3).reshape(batch, q_seq, -1)

        return self.linear_out(o)


class RelPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        scale_input: bool = True,
    ):
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

    def __call__(self, x: mx.array, offset: int = 0) -> tuple[mx.array, mx.array]:
        input_len = x.shape[1] + offset

        if input_len > self.max_len:
            self.max_len = input_len + 1
            self.calculate_pe()

        x = x * self.scale

        buffer_len = self._pe.shape[1]
        start_idx = buffer_len // 2 - (input_len - 1)
        end_idx = buffer_len // 2 + (input_len - 1) + 1

        pos_emb = self._pe[:, start_idx:end_idx].astype(x.dtype)

        return x, pos_emb
