# Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""T2S (Text2Semantic) GPT-2 core ported to MLX.

Port phan loi: semantic embedding + learned pos + 24 GPT-2 blocks + ln_f +
final_norm + semantic_head. Prefix (speaker/text encoder) van do torch tinh o
B1 de co lap phep toan GPT-2. Weight doc thang tu t2s_model.safetensors (F32,
GPT-2 Conv1D layout [in,out]).
"""

import math
import os

import mlx.core as mx
import numpy as np

_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights")

BOS, EOS = 8192, 8193
N_LAYERS = 24
N_HEADS = 20
D_MODEL = 1280
HEAD_DIM = D_MODEL // N_HEADS


def find_ckpt():
    # In normal use the Model passes an explicit path; this default is only for
    # standalone use with a sibling weights/ dir.
    local = os.path.join(_WEIGHTS, "t2s_model.safetensors")
    if os.path.exists(local):
        return local
    raise FileNotFoundError(
        f"{local} not found; pass an explicit checkpoint path (see convert.py)."
    )


def gelu_new(x):
    # GPT-2 'gelu_new' (tanh approximation), matches HF
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + eps) * w + b


class T2SMLX:
    def __init__(self, ckpt=None, group_size=64, bits=8):
        self.W = mx.load(ckpt or find_ckpt())  # dict[str, mx.array]
        self.qgs, self.qbits = group_size, bits  # for quantized_matmul (int8/int4)

    # ---- embeddings ----
    def semantic_embed(self, ids):
        """ids: mx int array (B,T) -> (B,T,D) with learned semantic position added."""
        se = self.W["semantic_embedding.weight"][ids]  # (B,T,D)
        T = ids.shape[-1]
        pos = self.W["semantic_position_embedding.embedding.weight"][:T]  # (T,D)
        return se + pos[None]

    # GPT-2 Conv1D matmul x@W (W stored [in,out]); uses 8-bit quantized_matmul
    # when the weight has been quantized (sibling .scales/.biases present).
    def _cw(self, x, k):
        W = self.W
        sk = k[:-7] + ".scales"
        if sk in W:
            return mx.quantized_matmul(
                x,
                W[k],
                W[sk],
                W[k[:-7] + ".biases"],
                transpose=True,
                group_size=self.qgs,
                bits=self.qbits,
            )
        return x @ W[k]

    # ---- one GPT-2 block (optional KV cache) ----
    def _block(self, x, i, mask, cache=None):
        W = self.W
        p = f"transformer.h.{i}."
        h = layernorm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = (
            self._cw(h, p + "attn.c_attn.weight") + W[p + "attn.c_attn.bias"]
        )  # (B,T,3D)
        B, T, _ = qkv.shape
        q, k, v = mx.split(qkv, 3, axis=-1)

        def heads(t):
            return t.reshape(B, T, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        q, k, v = heads(q), heads(k), heads(v)
        if cache is not None and cache[0] is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(HEAD_DIM))
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        o = attn @ v
        o = o.transpose(0, 2, 1, 3).reshape(B, T, D_MODEL)
        o = self._cw(o, p + "attn.c_proj.weight") + W[p + "attn.c_proj.bias"]
        x = x + o
        h2 = layernorm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        h2 = gelu_new(self._cw(h2, p + "mlp.c_fc.weight") + W[p + "mlp.c_fc.bias"])
        h2 = self._cw(h2, p + "mlp.c_proj.weight") + W[p + "mlp.c_proj.bias"]
        return x + h2, new_cache

    def transformer(self, inputs_embeds, caches=None):
        """inputs_embeds: (B,T,D) -> (last_hidden_state after ln_f, caches).
        caches=None means full causal pass (no cache); else incremental."""
        B, T, _ = inputs_embeds.shape
        if caches is None and T > 1:
            mask = mx.triu(mx.full((T, T), -1e9, dtype=inputs_embeds.dtype), k=1)[
                None, None
            ]
        elif caches is None:
            mask = None
        else:
            # incremental step (T usually 1): new token sees all cached -> no mask
            mask = None
        x = inputs_embeds
        out_caches = []
        for i in range(N_LAYERS):
            c = caches[i] if caches is not None else None
            x, nc = self._block(x, i, mask, c)
            out_caches.append(nc)
        h = layernorm(
            x, self.W["transformer.ln_f.weight"], self.W["transformer.ln_f.bias"]
        )
        return h, out_caches

    def _prefill_mask(self, T, dtype):
        return mx.triu(mx.full((T, T), -1e9, dtype=dtype), k=1)[None, None]

    def logits_from_embeds(self, inputs_embeds):
        h, _ = self.transformer(inputs_embeds)
        return self._head(h)

    def _head(self, h_lnf):
        W = self.W
        h = layernorm(h_lnf, W["final_norm.weight"], W["final_norm.bias"])
        if "semantic_head.scales" in W:
            return (
                mx.quantized_matmul(
                    h,
                    W["semantic_head.weight"],
                    W["semantic_head.scales"],
                    W["semantic_head.biases"],
                    transpose=True,
                    group_size=self.qgs,
                    bits=self.qbits,
                )
                + W["semantic_head.bias"]
            )
        return h @ W["semantic_head.weight"].T + W["semantic_head.bias"]

    def _sem_token_embed(self, tok, pos):
        """single semantic token id at absolute semantic position -> (1,1,D)."""
        e = self.W["semantic_embedding.weight"][tok]
        e = e + self.W["semantic_position_embedding.embedding.weight"][pos]
        return e[None, None]

    def _sample(self, logits, gen, temperature, top_k, top_p, rep_pen, rng):
        logits = np.array(logits, dtype=np.float64)
        if gen and rep_pen != 1.0:
            g = np.array(list(set(gen)))
            logits[g] = np.where(
                logits[g] > 0, logits[g] / rep_pen, logits[g] * rep_pen
            )
        logits = logits / temperature
        if top_k and top_k < logits.shape[0]:
            kth = np.partition(logits, -top_k)[-top_k]
            logits[logits < kth] = -np.inf
        order = np.argsort(logits)[::-1]
        sp = logits[order]
        probs = np.exp(sp - sp.max())
        probs /= probs.sum()
        keep = np.cumsum(probs) <= top_p
        keep[0] = True
        sp[~keep] = -np.inf
        full = np.full_like(logits, -np.inf)
        full[order] = sp
        p = np.exp(full - np.nanmax(full))
        p /= p.sum()
        return int(rng.choice(len(p), p=p))

    def generate(
        self,
        cond_emb,
        text_emb,
        max_new=512,
        temperature=0.8,
        top_k=30,
        top_p=0.8,
        rep_pen=10.0,
        seed=0,
    ):
        """KV-cached autoregressive sampling. cond_emb (1,1,D), text_emb (1,Tt,D).
        Returns (semantic_codes int[T], lm_latent float[1,T,D]) for S2A."""
        rng = np.random.default_rng(seed)
        prefix = mx.concatenate([cond_emb, text_emb], axis=1)
        Tt = text_emb.shape[1]
        # prefill over [cond, text, BOS]
        bos_emb = self._sem_token_embed(BOS, 0)
        x = mx.concatenate([prefix, bos_emb], axis=1)
        h, caches = self._prefill(x)
        logits = self._head(h[:, -1:])[0, -1]
        mx.eval(logits)
        cur = [BOS]
        pos = 1
        for _ in range(max_new):
            tok = self._sample(logits, cur[1:], temperature, top_k, top_p, rep_pen, rng)
            cur.append(tok)
            if tok == EOS:
                break
            e = self._sem_token_embed(tok, pos)
            pos += 1
            h, caches = self.transformer(e, caches=caches)
            logits = self._head(h)[0, -1]
            mx.eval(logits)
        gen_raw = cur[1:]  # includes EOS if emitted
        scodes = [BOS] + gen_raw
        hful, _ = self.transformer(
            mx.concatenate([prefix, self.semantic_embed(mx.array([scodes]))], axis=1)
        )  # post ln_f
        latent = hful[:, 1 + Tt : -2]
        mx.eval(latent)
        return np.array(scodes[1:-1], dtype=np.int64), np.array(latent)

    def _prefill(self, x):
        """full causal pass that also returns per-layer caches."""
        B, T, _ = x.shape
        mask = self._prefill_mask(T, x.dtype)
        out_caches = []
        for i in range(N_LAYERS):
            x, nc = self._block(x, i, mask, (None, None))
            out_caches.append(nc)
        h = layernorm(
            x, self.W["transformer.ln_f.weight"], self.W["transformer.ln_f.bias"]
        )
        return h, out_caches
