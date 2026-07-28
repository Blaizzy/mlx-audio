# Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""S2A flow-matching estimator (DiT + WaveNet) ported to MLX.

Validate-driven: phan DiT estimator truoc (RoPE interleaved fp32, adaLN/RMSNorm,
SwiGLU, U-Net skip, WaveNet gated). Weights: b2/s2a_mlx.safetensors (weight_norm
da fold, layout torch giu nguyen: Linear (out,in), Conv1d (out,in,k)).
"""

import math
import os

import mlx.core as mx

HID, NH, HD, DEPTH = 512, 8, 64, 13
ST = os.path.join(os.path.dirname(__file__), "..", "weights", "s2a_mlx.safetensors")


def lin(x, W, b=None):
    y = x @ W.T
    return y + b if b is not None else y


def conv1d(x_btc, W_oik, b=None, pad=0, dilation=1):
    """x_btc (B,T,Cin); W_oik torch layout (Cout,Cin,k) -> (B,T,Cout)."""
    w = mx.transpose(W_oik, (0, 2, 1))  # (Cout,k,Cin) for mx
    y = mx.conv1d(x_btc, w, stride=1, padding=pad, dilation=dilation)
    return y + b if b is not None else y


def rms_norm(x, w, eps=1e-5):
    return x * mx.rsqrt((x * x).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x * mx.sigmoid(x)


def mish(x):
    return x * mx.tanh(mx.logaddexp(x, mx.zeros_like(x)))  # x*tanh(softplus(x))


class S2AEstimator:
    def __init__(self, st=ST):
        self.W = mx.load(st)
        self.freqs = self.W["decoder.estimator.freqs_cis"]  # (4096,32,2)

    def g(self, name):
        return self.W["decoder.estimator." + name]

    # ---- timestep embedding ----
    def _t_embed(self, t, prefix):
        half = 128
        emb = mx.exp(mx.arange(half) * (-math.log(10000) / half))
        emb = 1000.0 * t[:, None] * emb[None]  # (B,256/2)
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)  # (B,256)
        emb = silu(
            lin(
                emb,
                self.g(prefix + ".time_mlp.0.weight"),
                self.g(prefix + ".time_mlp.0.bias"),
            )
        )
        return lin(
            emb,
            self.g(prefix + ".time_mlp.2.weight"),
            self.g(prefix + ".time_mlp.2.bias"),
        )

    def _extend_freqs(self, T):
        # freqs_cis is precomputed only up to 4096 positions; a long ref mel
        # (T_ref + target > 4096 frames) would slice short and crash _rope's
        # reshape. RoPE angle is linear in position (angle(p)=p*theta), so we
        # recover theta from row 1 and append the missing rows, leaving the
        # original 4096 rows byte-identical.
        import numpy as np

        f = np.array(self.freqs)  # (P0, hd//2, 2)
        P0 = f.shape[0]
        theta = np.arctan2(f[1, :, 1], f[1, :, 0])
        extra = np.arange(P0, T)[:, None] * theta[None, :]
        tail = np.stack([np.cos(extra), np.sin(extra)], -1).astype(f.dtype)
        self.freqs = mx.array(np.concatenate([f, tail], 0))

    def _rope(self, x):
        # x (B,T,nh,hd) interleaved pairs
        B, T, nh, hd = x.shape
        if T > self.freqs.shape[0]:
            self._extend_freqs(T)
        xs = x.reshape(B, T, nh, hd // 2, 2)
        c = self.freqs[:T, :, 0].reshape(1, T, 1, hd // 2)
        s = self.freqs[:T, :, 1].reshape(1, T, 1, hd // 2)
        xr, xi = xs[..., 0], xs[..., 1]
        o0 = xr * c - xi * s
        o1 = xi * c + xr * s
        return mx.stack([o0, o1], axis=-1).reshape(B, T, nh, hd)

    def _adaln(self, x, cond, prefix):
        mod = lin(
            cond,
            self.g(prefix + ".modulation.weight"),
            self.g(prefix + ".modulation.bias"),
        )
        w, b = mx.split(mod, 2, axis=-1)
        xn = rms_norm(x, self.g(prefix + ".norm.weight"))
        return xn * w[:, None] + b[:, None]

    def _attn(self, x, prefix):
        B, T, _ = x.shape
        qkv = lin(x, self.g(prefix + ".wqkv.weight"))
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = self._rope(q.reshape(B, T, NH, HD))
        k = self._rope(k.reshape(B, T, NH, HD))
        v = v.reshape(B, T, NH, HD)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        sc = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(HD))
        a = mx.softmax(sc, axis=-1) @ v
        a = a.transpose(0, 2, 1, 3).reshape(B, T, HID)
        return lin(a, self.g(prefix + ".wo.weight"))

    def _ff(self, x, prefix):
        return lin(
            silu(lin(x, self.g(prefix + ".w1.weight")))
            * lin(x, self.g(prefix + ".w3.weight")),
            self.g(prefix + ".w2.weight"),
        )

    def _block(self, h, cond, idx, skip_in):
        p = f"transformer_blocks.{idx}."
        if skip_in is not None:
            h = lin(
                mx.concatenate([h, skip_in], axis=-1),
                self.g(p + "skip_in_linear.weight"),
                self.g(p + "skip_in_linear.bias"),
            )
        h = h + self._attn(self._adaln(h, cond, p + "attention_norm"), p + "attention")
        h = h + self._ff(self._adaln(h, cond, p + "ffn_norm"), p + "feed_forward")
        return h

    def _wavenet(self, x_bct, g_b1):
        # x_bct (B,512,T) torch layout; work in (B,T,512)
        x = x_bct.transpose(0, 2, 1)  # (B,T,512)
        gp = self.g("wavenet.cond_layer.conv.weight")  # (8192,512,1)
        gcond = conv1d(
            g_b1.transpose(0, 2, 1), gp, self.g("wavenet.cond_layer.conv.bias")
        )  # (B,1,8192)
        out = mx.zeros_like(x)
        n = 8
        for i in range(n):
            xin = conv1d(
                x,
                self.g(f"wavenet.in_layers.{i}.conv.weight"),
                self.g(f"wavenet.in_layers.{i}.conv.bias"),
                pad=2,
            )  # k5 dil1 pad2 -> (B,T,1024)
            gl = gcond[:, :, i * 1024 : (i + 1) * 1024]  # (B,1,1024)
            ina = xin + gl
            acts = mx.tanh(ina[..., :512]) * mx.sigmoid(ina[..., 512:])  # (B,T,512)
            rs = conv1d(
                acts,
                self.g(f"wavenet.res_skip_layers.{i}.conv.weight"),
                self.g(f"wavenet.res_skip_layers.{i}.conv.bias"),
            )  # k1
            if i < n - 1:
                x = x + rs[..., :512]
                out = out + rs[..., 512:]
            else:
                out = out + rs
        return out.transpose(0, 2, 1)  # (B,512,T)

    def _final_layer(self, x, c):
        mod = lin(
            silu(c),
            self.g("final_layer.adaLN_modulation.1.weight"),
            self.g("final_layer.adaLN_modulation.1.bias"),
        )
        shift, scale = mx.split(mod, 2, axis=-1)
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        xn = (x - mu) * mx.rsqrt(var + 1e-6)  # LayerNorm no affine
        x = xn * (1.0 + scale[:, None]) + shift[:, None]
        return lin(
            x, self.g("final_layer.linear.weight"), self.g("final_layer.linear.bias")
        )

    # ---- preamble: semantic tokens + lm_latent -> mu (cat_condition) ----
    def _group_norm1(self, x, w, b, eps=1e-5):
        # GroupNorm(1, C) on (B,T,C): normalize over (T,C) per sample, affine per channel
        mu = x.mean(axis=(1, 2), keepdims=True)
        var = ((x - mu) ** 2).mean(axis=(1, 2), keepdims=True)
        return (x - mu) * mx.rsqrt(var + eps) * w + b

    def build_mu(self, codes, latent, T_ref):
        """codes int (1,T), latent (1,T,1280), T_ref int -> mu (1,T_ref+target,512)."""
        W = self.W
        T = codes.shape[1]
        emb = W["input_embedding.embedding.weight"][codes]  # (1,T,8)
        sem = conv1d(
            emb,
            W["input_embedding.out_project.weight"],
            W["input_embedding.out_project.bias"],
        )  # (1,T,1024)
        text_cond = lin(
            mx.concatenate([latent, sem], axis=-1),
            W["encoder_proj.weight"],
            W["encoder_proj.bias"],
        )  # (1,T,1024)
        # length regulator
        x = lin(
            text_cond,
            W["length_regulator.content_in_proj.weight"],
            W["length_regulator.content_in_proj.bias"],
        )  # (1,T,512)
        out_len = int(T * 1.72)
        idx = mx.minimum((mx.arange(out_len) * (T / out_len)).astype(mx.int32), T - 1)
        x = x[:, idx, :]  # nearest interp (1,out_len,512)
        for ci, gi in [(0, 1), (3, 4), (6, 7), (9, 10)]:
            x = conv1d(
                x,
                W[f"length_regulator.model.{ci}.weight"],
                W[f"length_regulator.model.{ci}.bias"],
                pad=1,
            )  # k3
            x = self._group_norm1(
                x,
                W[f"length_regulator.model.{gi}.weight"],
                W[f"length_regulator.model.{gi}.bias"],
            )
            x = mish(x)
        cond_target = conv1d(
            x,
            W["length_regulator.model.12.weight"],
            W["length_regulator.model.12.bias"],
        )  # k1 (1,out_len,512)
        prompt_cond = mx.broadcast_to(W["prompt_cond"], (1, T_ref, 512))
        return mx.concatenate(
            [prompt_cond, cond_target], axis=1
        )  # (1,T_ref+out_len,512)

    def solve_euler(self, z, prompt, mu, spks, t_span, cfg):
        """z (1,80,Ttot), prompt (1,80,T_ref), mu (1,Ttot,512), spks (1,192).
        Returns full mel (1,80,Ttot) via Euler ODE + CFG (mirrors torch)."""
        Ttot = z.shape[-1]
        T_ref = prompt.shape[-1]
        zeros_tail = mx.zeros((1, 80, Ttot - T_ref))
        prompt_x = mx.concatenate([prompt, zeros_tail], axis=-1)  # (1,80,Ttot)
        x = mx.concatenate([mx.zeros((1, 80, T_ref)), z[..., T_ref:]], axis=-1)
        z80 = mx.zeros_like(x)
        zmu = mx.zeros_like(mu)
        zspk = mx.zeros_like(spks)
        t = float(t_span[0])
        dt = float(t_span[1] - t_span[0])
        for step in range(1, t_span.shape[0]):
            x_in = mx.concatenate([x, x], axis=0)
            px_in = mx.concatenate([prompt_x, z80], axis=0)
            mu_in = mx.concatenate([mu, zmu], axis=0)
            spk_in = mx.concatenate([spks, zspk], axis=0)
            t_in = mx.array([t, t])
            dphi = self.forward(x_in, mu_in, t_in, spk_in, px_in)  # (2,80,Ttot)
            cond_d, uncond_d = dphi[:1], dphi[1:]
            d = (1.0 + cfg) * cond_d - cfg * uncond_d
            x = x + dt * d
            t = t + dt
            if step < t_span.shape[0] - 1:
                dt = float(t_span[step + 1] - t)
            # torch re-zeros prompt region of x every step (line 189)
            x = mx.concatenate([mx.zeros((1, 80, T_ref)), x[..., T_ref:]], axis=-1)
            mx.eval(x)
        return x

    def forward(self, x_bct, mu, t, spks, cond_bct):
        """x,cond (B,80,T); mu (B,T,512); t (B,); spks (B,192) -> (B,80,T)."""
        B = x_bct.shape[0]
        x = x_bct.transpose(0, 2, 1)  # (B,T,80)
        cond = cond_bct.transpose(0, 2, 1)
        T = x.shape[1]
        t1 = self._t_embed(t, "t_embedder")
        # input embed
        mu_proj = lin(
            mu,
            self.g("input_embed.mu_projection.weight"),
            self.g("input_embed.mu_projection.bias"),
        )
        spks_seq = mx.broadcast_to(spks[:, None, :], (B, T, spks.shape[-1]))
        xin = mx.concatenate([x, cond, mu_proj, spks_seq], axis=-1)  # (B,T,864)
        h = lin(xin, self.g("input_embed.proj.weight"), self.g("input_embed.proj.bias"))
        # transformer with U-Net skip
        emit = set(range(DEPTH // 2))
        recv = set(i for i in range(DEPTH) if i > DEPTH // 2)
        stack = []
        for idx in range(DEPTH):
            skip_in = stack.pop() if (idx in recv and stack) else None
            h = self._block(h, t1, idx, skip_in)
            if idx in emit:
                stack.append(h)
        x_res = self._adaln(h, t1, "transformer_norm")
        x_res = lin(
            mx.concatenate([x_res, x], axis=-1),
            self.g("skip_linear.weight"),
            self.g("skip_linear.bias"),
        )
        # wavenet final
        x_out = lin(x_res, self.g("conv1.weight"), self.g("conv1.bias")).transpose(
            0, 2, 1
        )  # (B,512,T)
        t2 = self._t_embed(t, "t_embedder2")
        x_out = self._wavenet(x_out, t2[:, :, None])
        x_out = x_out.transpose(0, 2, 1) + lin(
            x_res, self.g("res_projection.weight"), self.g("res_projection.bias")
        )
        x_out = self._final_layer(x_out, t1).transpose(0, 2, 1)  # (B,512,T)
        x_out = conv1d(
            x_out.transpose(0, 2, 1), self.g("conv2.weight"), self.g("conv2.bias")
        ).transpose(0, 2, 1)
        return x_out  # (B,80,T)
