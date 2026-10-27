# Portions derived from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya
# Research, licensed under the Apache License 2.0 (see LICENSE in this directory).
"""Maya-2's depth decoder: 31 codebook levels per 80 ms frame on top of the backbone's hidden state.

Module layout and parameter names follow DepthV2 in MayaResearch/maya2-native (tts/model/depth_v2.py). The frame body
is the serving path of serve/fast_depth.FastDepth: prefix position k is embedded from slot k alone; levels that share a
trunk group reuse the group's K/V (exact, the self-attention is causal); the last layer of a group's first pass reads
only the final position; and the speaker cross-attention is a per-voice constant (one reference token => the softmax
is exactly one, so each layer adds o(v)).
"""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

# level -> private trunk group: levels 1-8 own groups 0-7; 9-16, 17-24 and 25-31 share groups 8, 9 and 10
GROUP_OF_LEVEL = [None] + list(range(8)) + [8] * 8 + [9] * 8 + [10] * 7


class SelfBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.f1, self.f2 = nn.Linear(dim, ffn, bias=False), nn.Linear(
            ffn, dim, bias=False
        )

    def __call__(self, x, kv_prev=None, causal=True, last_only=False):
        """x [B, T, D]; kv_prev: this layer's (k, v) of the group's earlier positions. Returns (x, (k, v))."""
        B, T, D = x.shape
        H = self.heads
        qkv = self.qkv(self.n1(x)).reshape(B, T, 3, H, D // H)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        if kv_prev is not None:
            k, v = mx.concatenate([kv_prev[0], k], 2), mx.concatenate(
                [kv_prev[1], v], 2
            )
        if last_only:
            q, x = q[:, :, -1:], x[:, -1:]
        a = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(D // H) ** -0.5, mask="causal" if causal else None
        )
        x = x + self.o(a.transpose(0, 2, 1, 3).reshape(B, x.shape[1], D))
        return x + self.f2(nn.gelu(self.f1(self.n2(x)))), (k, v)


class CrossBlock(nn.Module):
    """Speaker cross-attention, shared across groups at one layer index (only kv/o are used when serving)."""

    def __init__(self, dim: int):
        super().__init__()
        self.n = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)


class DepthDecoder(nn.Module):
    def __init__(
        self,
        d_backbone=2048,
        dim=1024,
        heads=16,
        ffn=3072,
        layers=4,
        groups=11,
        rank=128,
        n_q=32,
        n_codes=2048,
        spk_dim=192,
    ):
        super().__init__()
        self.dim, self.n_q, self.n_codes, self.n_layers = dim, n_q, n_codes, layers
        self.in_proj = nn.Linear(d_backbone, dim, bias=False)
        self.code_lo = nn.Embedding((n_q - 1) * n_codes, rank)
        self.code_up = nn.Linear(rank, dim, bias=False)
        self.level_emb = nn.Embedding(n_q, dim)
        self.groups = [
            [SelfBlock(dim, heads, ffn) for _ in range(layers)] for _ in range(groups)
        ]
        self.cross = [CrossBlock(dim) for _ in range(layers)]
        self.norm = nn.LayerNorm(dim)
        self.heads = [nn.Linear(dim, n_codes, bias=False) for _ in range(n_q - 1)]
        self.spk_proj_campp = nn.Linear(
            1024, dim, bias=False
        )  # unused when serving; kept so loading is strict
        self.spk_proj_ecapa = nn.Linear(spk_dim, dim, bias=False)
        self.null_mem = mx.zeros((1, 1, dim))
        self._table: Optional[mx.array] = None
        self._spk_proj: Optional[mx.array] = None

    def to_serving_dtype(self, dtype):
        """Cast to the serving dtype, deriving the prefix table and keeping the speaker projection from the
        source-precision weights first: casting bf16 -> fp16 before deriving rounds small values and was measured to
        flip near-tie codes at the deepest levels."""
        lo = self.code_lo.weight.astype(mx.float32).reshape(
            self.n_q - 1, self.n_codes, -1
        )
        table = (lo @ self.code_up.weight.astype(mx.float32).T).astype(dtype)
        self._table = (
            table + self.level_emb.weight[1 : self.n_q, None].astype(dtype)
        ).reshape(-1, self.dim)
        self._spk_proj = self.spk_proj_ecapa.weight
        mx.eval(self._table, self._spk_proj)
        self.set_dtype(dtype)

    def prefix_table(self) -> mx.array:
        """[(n_q - 1) * n_codes, D]: row level * n_codes + c is the prefix embedding of code c at slot level + 1."""
        if self._table is None:
            self.to_serving_dtype(self.level_emb.weight.dtype)
        return self._table

    def speaker_constants(self, ecapa=None) -> mx.array:
        """[layers, 1, D] cross-attention outputs for a voice: ECAPA (192,) or None (the learned null memory).
        The memory is computed in fp32 from the source-precision projection and cast, as the reference engine does.
        """
        dtype = self.level_emb.weight.dtype
        if ecapa is None:
            mem = self.null_mem[0].astype(dtype)
        else:
            proj = (
                self._spk_proj
                if self._spk_proj is not None
                else self.spk_proj_ecapa.weight
            )
            mem = (
                mx.array(ecapa, mx.float32)[None] @ proj.astype(mx.float32).T
            ).astype(dtype)
        return mx.stack([c.o(c.kv(mem)[:, self.dim :]) for c in self.cross])

    def frame(self, h, cb0, cross, key=None, temp=0.7, top_k=64, force=None):
        """h [B, d_backbone], cb0 [B] -> (codes [B, n_q], logits per level). key=None samples greedily; force [B, n_q]
        teacher-forces the prefix (for parity checks)."""
        table = self.prefix_table()
        dtype = table.dtype
        prefix = [
            (self.in_proj(h.astype(dtype)) + self.level_emb.weight[0])[:, None],
            table[cb0][:, None],
        ]
        codes, logits_all, kv, previous = [cb0], [], None, None
        for level in range(1, self.n_q):
            group, n = GROUP_OF_LEVEL[level], level + 1
            incremental = group == previous
            x = prefix[level] if incremental else mx.concatenate(prefix[:n], 1)
            new_kv: List = []
            for li, block in enumerate(self.groups[group]):
                last = (not incremental) and li == self.n_layers - 1
                x, kvl = block(
                    x,
                    kv[li] if incremental else None,
                    causal=not incremental and not last,
                    last_only=last,
                )
                new_kv.append(kvl)
                x = x + cross[li]
            kv, previous = new_kv, group
            logits = (
                self.heads[level - 1](self.norm(x[:, -1])).astype(mx.float32) / temp
            )
            logits_all.append(logits)
            if force is not None:
                code = force[:, level]
            elif key is None:
                code = mx.argmax(logits, -1)
            else:
                kth = mx.topk(logits, top_k, -1).min(-1, keepdims=True)
                probs = mx.softmax(mx.where(logits < kth, -mx.inf, logits), -1)
                key, sub = mx.random.split(key)
                race = -mx.log(
                    mx.random.uniform(shape=probs.shape, key=sub)
                )  # Exp(1) race == categorical draw
                code = mx.argmax(probs / race, -1)
            codes.append(code)
            if n < self.n_q:
                prefix.append(table[code + level * self.n_codes][:, None])
        return mx.stack(codes, -1), logits_all
