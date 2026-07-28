# Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Wav2Vec2-BERT (w2v-bert-2.0) conformer encoder -> hidden_states[17], in MLX.

feature_projection + 17 conformer layers. Each layer: ffn1(*0.5) -> self_attn
(relative_key) -> conv_module (causal depthwise) -> ffn2(*0.5) -> final LN.
Weights: b4/w2vbert_mlx.safetensors (fp32, torch layout).
"""

import math
import os

import mlx.core as mx

H, NH, HD, NLAYERS = 1024, 16, 64, 17
LEFT, RIGHT = 64, 8
ST = os.path.join(os.path.dirname(__file__), "..", "weights", "w2vbert_mlx.safetensors")


def lin(x, W, b=None):
    y = x @ W.T
    return y + b if b is not None else y


def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + eps) * w + b


def swish(x):
    return x * mx.sigmoid(x)


class W2VBertMLX:
    def __init__(self, st=ST, group_size=64, bits=8):
        self.W = mx.load(st)
        self.qgs, self.qbits = group_size, bits  # for optional int8/int4 linears

    def g(self, k):
        return self.W[k]

    # Linear (weight [out,in]); 8-bit quantized_matmul when sibling .scales present.
    def _lin(self, x, wk, bk=None):
        W = self.W
        sk = wk[:-7] + ".scales"
        if sk in W:
            y = mx.quantized_matmul(
                x,
                W[wk],
                W[sk],
                W[wk[:-7] + ".biases"],
                transpose=True,
                group_size=self.qgs,
                bits=self.qbits,
            )
        else:
            y = x @ W[wk].T
        return y + W[bk] if bk is not None else y

    def _ffn(self, x, p):
        h = swish(
            self._lin(
                x, p + ".intermediate_dense.weight", p + ".intermediate_dense.bias"
            )
        )
        return self._lin(h, p + ".output_dense.weight", p + ".output_dense.bias")

    def _attn(self, x, p):
        B, T, _ = x.shape
        q = (
            self._lin(x, p + ".linear_q.weight", p + ".linear_q.bias")
            .reshape(B, T, NH, HD)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self._lin(x, p + ".linear_k.weight", p + ".linear_k.bias")
            .reshape(B, T, NH, HD)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self._lin(x, p + ".linear_v.weight", p + ".linear_v.bias")
            .reshape(B, T, NH, HD)
            .transpose(0, 2, 1, 3)
        )
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(HD)
        # relative_key: distance = r - l, clamp[-LEFT,RIGHT], +LEFT -> distance_embedding (T,T,HD)
        l = mx.arange(T).reshape(T, 1)
        r = mx.arange(T).reshape(1, T)
        dist = mx.clip(r - l, -LEFT, RIGHT) + LEFT  # (T,T) in [0,72]
        pe = self.g(p + ".distance_embedding.weight")[dist]  # (T,T,HD)
        rel = mx.einsum("bhld,lrd->bhlr", q, pe) / math.sqrt(HD)  # (B,NH,T,T)
        scores = scores + rel
        a = mx.softmax(scores, axis=-1) @ v  # (B,NH,T,HD)
        a = a.transpose(0, 2, 1, 3).reshape(B, T, H)
        return self._lin(a, p + ".linear_out.weight", p + ".linear_out.bias")

    def _conv(self, x, p):
        B, T, _ = x.shape
        h = layernorm(
            x, self.g(p + ".layer_norm.weight"), self.g(p + ".layer_norm.bias")
        )
        # pointwise_conv1 (k1, no bias): 1024->2048
        h = h @ self.g(p + ".pointwise_conv1.weight")[:, :, 0].T  # (B,T,2048)
        a, bb = mx.split(h, 2, axis=-1)
        h = a * mx.sigmoid(bb)  # GLU -> (B,T,1024)
        # causal depthwise k31: left pad 30, groups=1024
        wdw = mx.transpose(
            self.g(p + ".depthwise_conv.weight"), (0, 2, 1)
        )  # (1024,31,1)
        hp = mx.concatenate([mx.zeros((B, 30, H)), h], axis=1)
        h = mx.conv1d(hp, wdw, stride=1, padding=0, groups=H)  # (B,T,1024)
        h = layernorm(
            h,
            self.g(p + ".depthwise_layer_norm.weight"),
            self.g(p + ".depthwise_layer_norm.bias"),
        )
        h = swish(h)
        h = h @ self.g(p + ".pointwise_conv2.weight")[:, :, 0].T  # (B,T,1024)
        return h

    def _layer(self, x, i):
        p = f"encoder.layers.{i}."
        x = x + 0.5 * self._ffn(
            layernorm(
                x,
                self.g(p + "ffn1_layer_norm.weight"),
                self.g(p + "ffn1_layer_norm.bias"),
            ),
            p + "ffn1",
        )
        x = x + self._attn(
            layernorm(
                x,
                self.g(p + "self_attn_layer_norm.weight"),
                self.g(p + "self_attn_layer_norm.bias"),
            ),
            p + "self_attn",
        )
        x = x + self._conv(x, p + "conv_module")
        x = x + 0.5 * self._ffn(
            layernorm(
                x,
                self.g(p + "ffn2_layer_norm.weight"),
                self.g(p + "ffn2_layer_norm.bias"),
            ),
            p + "ffn2",
        )
        return layernorm(
            x,
            self.g(p + "final_layer_norm.weight"),
            self.g(p + "final_layer_norm.bias"),
        )

    def hidden17(self, input_features):
        """input_features (1,T,160) -> hidden_states[17] (1,T,1024)."""
        x = layernorm(
            input_features,
            self.g("feature_projection.layer_norm.weight"),
            self.g("feature_projection.layer_norm.bias"),
        )
        x = lin(
            x,
            self.g("feature_projection.projection.weight"),
            self.g("feature_projection.projection.bias"),
        )
        for i in range(NLAYERS):
            x = self._layer(x, i)
        return x
