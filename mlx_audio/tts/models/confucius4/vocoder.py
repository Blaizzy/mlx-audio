# Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""BigVGAN v2 vocoder (mel -> waveform) ported to MLX.

snakebeta logscale + anti-aliased activation (upsample->snake->downsample with
fixed FIR filters). Weights b2/bigvgan_mlx.safetensors (weight_norm folded).
Config: ups [4,4,2,2,2,2], init ch 1536, resblock k[3,7,11] dil[1,3,5].
"""

import os

import mlx.core as mx

ST = os.path.join(os.path.dirname(__file__), "..", "weights", "bigvgan_mlx.safetensors")
UP_RATES = [4, 4, 2, 2, 2, 2]
UP_KERNELS = [8, 8, 4, 4, 4, 4]
RES_K = [3, 7, 11]
RES_D = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


def conv1d(x, W_oik, b=None, pad=0, dilation=1, stride=1, groups=1):
    w = mx.transpose(W_oik, (0, 2, 1))  # (Cout,k,Cin/groups)
    y = mx.conv1d(x, w, stride=stride, padding=pad, dilation=dilation, groups=groups)
    return y + b if b is not None else y


def rep_pad(x, l, r):
    B, T, C = x.shape
    left = mx.broadcast_to(x[:, :1, :], (B, l, C))
    right = mx.broadcast_to(x[:, -1:, :], (B, r, C))
    return mx.concatenate([left, x, right], axis=1)


class BigVGANMLX:
    def __init__(self, st=ST):
        self.W = mx.load(st)

    def _snakebeta(self, x, prefix):
        a = mx.exp(self.W[prefix + ".alpha"]).reshape(1, 1, -1)
        b = mx.exp(self.W[prefix + ".beta"]).reshape(1, 1, -1)
        return x + (1.0 / (b + 1e-9)) * mx.sin(x * a) ** 2

    def _aa_act(self, x, prefix):
        """anti-aliased snakebeta on (B,T,C). up x2 -> snake -> down x2."""
        B, T, C = x.shape
        # upsample x2 (k=12): replicate pad 5, grouped conv_transpose, *2, crop 15
        fu = self.W[prefix + ".upsample.filter"].reshape(1, 12, 1)
        wu = mx.broadcast_to(fu, (C, 12, 1))  # (Cout=C,k,Cin/g=1)
        xu = rep_pad(x, 5, 5)
        xu = 2.0 * mx.conv_transpose1d(xu, wu, stride=2, padding=0, groups=C)
        xu = xu[:, 15:-15, :]  # (B,2T,C)
        xu = self._snakebeta(xu, prefix + ".act")
        # downsample x2 (k=12): replicate pad (5,6), grouped conv stride2
        fd = self.W[prefix + ".downsample.lowpass.filter"].reshape(1, 12, 1)
        wd = mx.broadcast_to(fd, (C, 12, 1))
        xd = rep_pad(xu, 5, 6)
        xd = mx.conv1d(xd, wd, stride=2, padding=0, groups=C)
        return xd  # (B,T,C)

    def _resblock(self, x, idx, k):
        for j, d in enumerate(RES_D[0]):
            p = f"resblocks.{idx}."
            xt = self._aa_act(x, p + f"activations.{2*j}")
            xt = conv1d(
                xt,
                self.W[p + f"convs1.{j}.weight"],
                self.W[p + f"convs1.{j}.bias"],
                pad=d * (k - 1) // 2,
                dilation=d,
            )
            xt = self._aa_act(xt, p + f"activations.{2*j+1}")
            xt = conv1d(
                xt,
                self.W[p + f"convs2.{j}.weight"],
                self.W[p + f"convs2.{j}.bias"],
                pad=(k - 1) // 2,
                dilation=1,
            )
            x = x + xt
        return x

    def __call__(self, mel_bct):
        """mel (1,80,T) -> wav (1, T*256)."""
        x = mel_bct.transpose(0, 2, 1)  # (1,T,80)
        x = conv1d(
            x, self.W["conv_pre.weight"], self.W["conv_pre.bias"], pad=3
        )  # (1,T,1536)
        for i in range(6):
            # upsample (ConvTranspose1d ch->ch/2)
            wt = self.W[f"ups.{i}.0.weight"]  # torch (Cin,Cout,k)
            wt = mx.transpose(wt, (1, 2, 0))  # (Cout,k,Cin)
            k, u = UP_KERNELS[i], UP_RATES[i]
            x = (
                mx.conv_transpose1d(x, wt, stride=u, padding=(k - u) // 2)
                + self.W[f"ups.{i}.0.bias"]
            )
            xs = None
            for j in range(3):
                r = self._resblock(x, i * 3 + j, RES_K[j])
                xs = r if xs is None else xs + r
            x = xs / 3.0
            mx.eval(x)
        x = self._aa_act(x, "activation_post")
        x = conv1d(x, self.W["conv_post.weight"], pad=3)  # no bias (1,T*256,1)
        x = mx.clip(x, -1.0, 1.0)
        return x.transpose(0, 2, 1).reshape(1, -1)  # (1, T*256)
