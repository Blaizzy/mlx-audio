# Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""T2S prefix encoders in MLX: text_projector + ECAPA-TDNN speaker_encoder.

Produces the [condition_emb | text_emb] prefix from (w2v condition_vector, token_ids),
so the T2S decode no longer needs the torch prefix. Weights read from
t2s_model.safetensors (text_projector.*, speaker_encoder.*, text_position_embedding.*).
"""

import mlx.core as mx

from .t2s import find_ckpt

EPS = 1e-12


def lin(x, W, b=None):
    y = x @ W.T
    return y + b if b is not None else y


def relu(x):
    return mx.maximum(x, 0)


def silu(x):
    return x * mx.sigmoid(x)


def reflect_pad(x, p):
    """reflect pad along time axis=1 of (B,T,C), matching torch 'reflect'."""
    if p == 0:
        return x
    T = x.shape[1]
    li = p - mx.arange(p)  # [p, p-1, ..., 1]
    ri = (T - 2) - mx.arange(p)  # [T-2, T-3, ..., T-1-p]
    left = mx.take(x, li, axis=1)
    right = mx.take(x, ri, axis=1)
    return mx.concatenate([left, x, right], axis=1)


def conv_same(x, W, b, dilation=1, groups=1):
    """conv1d 'same' padding (reflect). x (B,T,Cin); W torch (Cout,Cin,k)."""
    k = W.shape[2]
    p = dilation * (k - 1) // 2
    xp = reflect_pad(x, p)
    w = mx.transpose(W, (0, 2, 1))
    return mx.conv1d(xp, w, stride=1, padding=0, dilation=dilation, groups=groups) + b


class T2SPrefixMLX:
    def __init__(self, ckpt=None):
        self.W = mx.load(ckpt or find_ckpt())

    def g(self, k):
        return self.W[k]

    # ---------- text path ----------
    def text_emb(self, token_ids):
        e = self.g("text_projector.embed.weight")[token_ids]  # (1,T,4096)
        e = silu(
            lin(
                e,
                self.g("text_projector.text_projection_fc1.weight"),
                self.g("text_projector.text_projection_fc1.bias"),
            )
        )
        e = lin(
            e,
            self.g("text_projector.text_projection_fc2.weight"),
            self.g("text_projector.text_projection_fc2.bias"),
        )  # (1,T,1280)
        T = token_ids.shape[1]
        return e + self.g("text_position_embedding.embedding.weight")[:T][None]

    # ---------- ECAPA speaker encoder ----------
    def _tdnn(self, x, p, dilation=1):
        return relu(
            conv_same(
                x,
                self.g(p + ".conv.weight"),
                self.g(p + ".conv.bias"),
                dilation=dilation,
            )
        )

    def _res2net(self, x, p, dilation, scale=8):
        chunks = mx.split(x, scale, axis=2)
        outs = []
        prev = None
        for i in range(scale):
            if i == 0:
                o = chunks[0]
            elif i == 1:
                o = self._tdnn(chunks[1], f"{p}.blocks.0", dilation=dilation)
            else:
                o = self._tdnn(chunks[i] + prev, f"{p}.blocks.{i-1}", dilation=dilation)
            outs.append(o)
            prev = o
        return mx.concatenate(outs, axis=2)

    def _se(self, x, p):
        s = x.mean(axis=1, keepdims=True)  # (B,1,C)
        s = relu(conv_same(s, self.g(p + ".conv1.weight"), self.g(p + ".conv1.bias")))
        s = mx.sigmoid(
            conv_same(s, self.g(p + ".conv2.weight"), self.g(p + ".conv2.bias"))
        )
        return x * s

    def _se_res2net(self, x, p, dilation):
        residual = x
        h = self._tdnn(x, p + ".tdnn1")
        h = self._res2net(h, p + ".res2net_block", dilation=dilation)
        h = self._tdnn(h, p + ".tdnn2")
        h = self._se(h, p + ".se_block")
        return h + residual

    def _stats(self, x, w):
        mean = (w * x).sum(axis=1)  # (B,C)
        std = mx.sqrt(mx.maximum((w * (x - mean[:, None]) ** 2).sum(axis=1), EPS))
        return mean, std

    def _asp(self, x):
        B, T, C = x.shape
        m = mx.full((B, T, 1), 1.0 / T)
        mean, std = self._stats(x, m)
        att_in = mx.concatenate(
            [
                x,
                mx.broadcast_to(mean[:, None], (B, T, C)),
                mx.broadcast_to(std[:, None], (B, T, C)),
            ],
            axis=2,
        )
        h = self._tdnn(att_in, "speaker_encoder.asp.tdnn")
        h = mx.tanh(h)
        h = conv_same(
            h,
            self.g("speaker_encoder.asp.conv.weight"),
            self.g("speaker_encoder.asp.conv.bias"),
        )
        att = mx.softmax(h, axis=1)
        mean, std = self._stats(x, att)
        return mx.concatenate([mean, std], axis=1)[:, None]  # (B,1,2C)

    def cond_emb(self, cond_vec):
        """cond_vec (1,Tf,1024) -> (1,1,1280)."""
        x = cond_vec
        x = self._tdnn(x, "speaker_encoder.blocks.0", dilation=1)
        feats = []
        for i in range(1, 4):
            x = self._se_res2net(x, f"speaker_encoder.blocks.{i}", dilation=i + 1)
            feats.append(x)
        x = mx.concatenate(feats, axis=2)
        x = self._tdnn(x, "speaker_encoder.mfa")
        x = self._asp(x)
        x = conv_same(
            x, self.g("speaker_encoder.fc.weight"), self.g("speaker_encoder.fc.bias")
        )
        return x  # (1,1,1280)
