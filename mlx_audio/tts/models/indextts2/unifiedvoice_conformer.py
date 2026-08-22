from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.max_len = max_len

        position = mx.arange(max_len).astype(mx.float32)[:, None]
        div_term = mx.exp(
            mx.arange(0, d_model, 2, dtype=mx.float32)
            * (-(math.log(10000.0) / d_model))
        )
        pe = mx.zeros((max_len, d_model), dtype=mx.float32)
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        self.pe = pe[None, :, :]

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        if offset + x.shape[1] > self.max_len:
            # extend
            self.__init__(self.d_model, max_len=offset + x.shape[1] + 1)
        pos_emb = self.pe[:, offset : offset + x.shape[1]].astype(x.dtype)
        return x * self.xscale, pos_emb


class Conv2dSubsampling2(nn.Module):
    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.conv = [nn.Conv2d(1, odim, 3, 2), nn.ReLU()]
        self.out = [nn.Linear(odim * ((idim - 1) // 2), odim)]
        self.pos_enc = RelPositionalEncoding(odim)

    def __call__(
        self, x: mx.array, x_mask: mx.array, offset: int = 0
    ) -> Tuple[mx.array, mx.array, mx.array]:
        # x: (B, T, F)
        x = x[:, :, :, None]
        for layer in self.conv:
            x = layer(x)
        b, t, f, c = x.shape
        # Match torch path: (B, C, T, F) -> transpose(1,2) -> (B, T, C, F) -> flatten C*F.
        # MLX conv is channel-last (B, T, F, C), so reorder to (B, T, C, F) before flatten.
        x = x.transpose(0, 1, 3, 2).reshape(b, t, c * f)
        for layer in self.out:
            x = layer(x)
        x, pos = self.pos_enc(x, offset)
        # mask: (B, 1, T) -> (B,1,T')
        return x, pos, x_mask[:, :, 2::2]


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

    def forward_qkv(self, query: mx.array, key: mx.array, value: mx.array):
        n_batch = query.shape[0]
        q = self.linear_q(query).reshape(n_batch, -1, self.h, self.d_k).transpose(
            0, 2, 1, 3
        )
        k = self.linear_k(key).reshape(n_batch, -1, self.h, self.d_k).transpose(
            0, 2, 1, 3
        )
        v = self.linear_v(value).reshape(n_batch, -1, self.h, self.d_k).transpose(
            0, 2, 1, 3
        )
        return q, k, v

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
        pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        del pos_emb
        q, k, v = self.forward_qkv(query, key, value)
        attn_mask = None
        if mask is not None and mask.size > 0:
            # mask: (B,1,T)
            attn_mask = (1.0 - mask.astype(mx.float32))[:, None, None, :] * (-1e9)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.d_k**-0.5, mask=attn_mask)
        o = o.transpose(0, 2, 1, 3).reshape(query.shape[0], -1, self.h * self.d_k)
        return self.linear_out(o)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self, n_head: int, n_feat: int):
        super().__init__(n_head, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = mx.zeros((self.h, self.d_k), dtype=mx.float32)
        self.pos_bias_v = mx.zeros((self.h, self.d_k), dtype=mx.float32)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
        pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        if pos_emb is None:
            raise ValueError("pos_emb required")
        q, k, v = self.forward_qkv(query, key, value)
        # q: (B,H,T,D) -> (B,T,H,D)
        q_t = q.transpose(0, 2, 1, 3)

        p = self.linear_pos(pos_emb)
        p = p.reshape(pos_emb.shape[0], -1, self.h, self.d_k).transpose(0, 2, 1, 3)

        q_u = (q_t + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_v = (q_t + self.pos_bias_v).transpose(0, 2, 1, 3)

        matrix_ac = mx.matmul(q_u, k.swapaxes(-2, -1))
        matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))
        scores = (matrix_ac + matrix_bd) * (self.d_k**-0.5)

        if mask is not None and mask.size > 0:
            # mask: (B, 1, T)
            m = (mask == 0)[:, :, None, :]  # (B, 1, 1, T)
            scores = mx.where(m, -1e9, scores)

        probs = mx.softmax(scores, axis=-1)
        out = mx.matmul(probs, v)
        out = out.transpose(0, 2, 1, 3).reshape(query.shape[0], -1, self.h * self.d_k)
        return self.linear_out(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.activation = nn.SiLU()

    def __call__(self, xs: mx.array) -> mx.array:
        return self.w_2(self.activation(self.w_1(xs)))


class ConvolutionModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 15, bias: bool = True):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1, 1, 0, bias=bias)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1, 1, 0, bias=bias)
        self.activation = nn.SiLU()

    def __call__(self, x: mx.array, mask_pad: Optional[mx.array] = None) -> mx.array:
        # x: (B, T, C)  (MLX Conv1d expects channel-last)
        if mask_pad is not None and mask_pad.size > 0:
            # mask_pad: (B,1,T)
            x = mx.where(mask_pad.transpose(0, 2, 1), x, 0.0)

        x = self.pointwise_conv1(x)
        x = nn.glu(x, axis=-1)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        if mask_pad is not None and mask_pad.size > 0:
            x = mx.where(mask_pad.transpose(0, 2, 1), x, 0.0)
        return x


class ConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        conv_module: nn.Module,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        self.norm_conv = nn.LayerNorm(size, eps=1e-5)
        self.norm_final = nn.LayerNorm(size, eps=1e-5)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        pos_emb: mx.array,
        mask_pad: mx.array,
    ) -> mx.array:
        residual = x
        x = self.norm_mha(x)
        x = residual + self.self_attn(x, x, x, mask=mask_pad, pos_emb=pos_emb)

        residual = x
        x = self.norm_conv(x)
        x = residual + self.conv_module(x, mask_pad=mask_pad)

        residual = x
        x = self.norm_ff(x)
        x = residual + self.feed_forward(x)

        return self.norm_final(x)


@dataclass
class ConformerEncoderConfig:
    input_size: int = 1024
    output_size: int = 512
    attention_heads: int = 8
    linear_units: int = 2048
    num_blocks: int = 6
    input_layer: str = "conv2d2"


class ConformerEncoder(nn.Module):
    def __init__(self, cfg: ConformerEncoderConfig):
        super().__init__()
        self.embed = Conv2dSubsampling2(cfg.input_size, cfg.output_size)
        self.after_norm = nn.LayerNorm(cfg.output_size, eps=1e-5)

        self.encoders = [
            ConformerEncoderLayer(
                cfg.output_size,
                RelPositionMultiHeadedAttention(cfg.attention_heads, cfg.output_size),
                PositionwiseFeedForward(cfg.output_size, cfg.linear_units),
                ConvolutionModule(cfg.output_size, kernel_size=15),
            )
            for _ in range(cfg.num_blocks)
        ]

    def __call__(self, xs: mx.array, xs_lens: mx.array) -> Tuple[mx.array, mx.array]:
        # xs: (B, T, F)
        T = xs.shape[1]
        # masks: (B,1,T)
        mask = mx.arange(T)[None, :] < xs_lens[:, None]
        mask = mask[:, None, :]
        xs, pos_emb, mask = self.embed(xs, mask, 0)
        for layer in self.encoders:
            xs = layer(xs, mask, pos_emb, mask)
        xs = self.after_norm(xs)
        return xs, mask
