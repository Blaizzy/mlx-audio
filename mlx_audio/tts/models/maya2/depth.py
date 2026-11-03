import mlx.core as mx
from mlx import nn

from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.sample_utils import make_sampler

GROUP_OF_LEVEL = (None, *range(8), *([8] * 8), *([9] * 8), *([10] * 7))


class SelfBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.f1 = nn.Linear(d, cfg.intermediate_size, bias=False)
        self.f2 = nn.Linear(cfg.intermediate_size, d, bias=False)

    def __call__(self, x, cache=None):
        b, t, d = x.shape
        q, k, v = [
            a.reshape(b, t, self.n_heads, -1).transpose(0, 2, 1, 3)
            for a in mx.split(self.qkv(self.n1(x)), 3, axis=-1)
        ]
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        a = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=q.shape[-1] ** -0.5, mask="causal" if t > 1 else None
        )
        x = x + self.o(a.transpose(0, 2, 1, 3).reshape(b, t, d))
        return x + self.f2(nn.gelu(self.f1(self.n2(x))))


class CrossBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.n = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def __call__(self, x, mem, mask=None):
        b, t, d = x.shape
        s = mem.shape[1]
        # A single memory token has attention weight one for every query.
        if s == 1 and mask is None:
            return x + self.o(mx.split(self.kv(mem), 2, axis=-1)[1])
        q = self.q(self.n(x)).reshape(b, t, self.n_heads, -1).transpose(0, 2, 1, 3)
        k, v = [
            a.reshape(b, s, self.n_heads, -1).transpose(0, 2, 1, 3)
            for a in mx.split(self.kv(mem), 2, axis=-1)
        ]
        a = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=q.shape[-1] ** -0.5,
            mask=mask[:, None, None, :] if mask is not None else None,
        )
        return x + self.o(a.transpose(0, 2, 1, 3).reshape(b, t, d))


class DepthDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dc = cfg.depth_config
        d = dc.hidden_size
        self.n_q, self.n_codes = cfg.num_codebooks, cfg.codebook_size
        self.in_proj = nn.Linear(cfg.qwen_config().hidden_size, d, bias=False)
        self.code_lo = nn.Embedding((self.n_q - 1) * self.n_codes, dc.embedding_rank)
        self.code_up = nn.Linear(dc.embedding_rank, d, bias=False)
        self.level_emb = nn.Embedding(self.n_q, d)
        groups = max(GROUP_OF_LEVEL[1 : self.n_q]) + 1
        self.groups = [
            [SelfBlock(dc) for _ in range(dc.num_hidden_layers)] for _ in range(groups)
        ]
        self.cross = [CrossBlock(dc) for _ in range(dc.num_hidden_layers)]
        self.norm = nn.LayerNorm(d)
        self.heads = [
            nn.Linear(d, self.n_codes, bias=False) for _ in range(self.n_q - 1)
        ]
        self.spk_proj_campp = nn.Linear(1024, d, bias=False)
        self.spk_proj_ecapa = nn.Linear(cfg.speaker_dim, d, bias=False)
        self.null_mem = mx.zeros((1, 1, d))

    def speaker_memory(self, refs=None):
        if refs is None:
            return self.null_mem
        return self.spk_proj_ecapa(refs.astype(self.null_mem.dtype))

    def prefix(self, hidden, codes, level):
        slots = mx.arange(level)
        h = self.in_proj(hidden.astype(self.null_mem.dtype))[:, None]
        e = self.code_up(self.code_lo(codes[:, :level] + slots * self.n_codes))
        return mx.concatenate(
            [h + self.level_emb.weight[:1], e + self.level_emb(slots + 1)], axis=1
        )

    def run(self, x, group, memory, mask=None, cache=None):
        for i, (block, cross) in enumerate(zip(self.groups[group], self.cross)):
            x = block(x, cache[i] if cache is not None else None)
            x = cross(x, memory, mask)
        return self.norm(x[:, -1])

    def __call__(self, hidden, codes, memory=None, mask=None):
        memory = self.speaker_memory() if memory is None else memory
        return [
            self.heads[level - 1](
                self.run(
                    self.prefix(hidden, codes, level),
                    GROUP_OF_LEVEL[level],
                    memory,
                    mask,
                )
            )
            for level in range(1, codes.shape[1])
        ]

    def generate(self, hidden, cb0, memory=None, temperature=0.7, top_k=64):
        memory = self.speaker_memory() if memory is None else memory
        sampler = make_sampler(temp=temperature, top_k=min(top_k, self.n_codes))
        codes = [cb0]
        previous = None
        for level in range(1, self.n_q):
            group = GROUP_OF_LEVEL[level]
            if group != previous:
                cache = [KVCache() for _ in self.cross]
                x = self.prefix(hidden, mx.stack(codes, axis=1), level)
            else:
                code = codes[-1] + (level - 1) * self.n_codes
                x = (
                    self.code_up(self.code_lo(code))[:, None]
                    + self.level_emb.weight[level]
                )
            h = self.run(x, group, memory, cache=cache)
            codes.append(sampler(self.heads[level - 1](h).astype(mx.float32)))
            previous = group
        return mx.stack(codes, axis=1)
