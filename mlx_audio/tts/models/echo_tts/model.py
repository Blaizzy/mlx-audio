from __future__ import annotations

import math
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

RotaryCache = Tuple[mx.array, mx.array]
KVCache = Tuple[mx.array, mx.array]


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> RotaryCache:
    freqs = 1.0 / (
        theta ** (mx.arange(0, dim, 2, dtype=mx.float32)[: (dim // 2)] / float(dim))
    )
    t = mx.arange(end, dtype=mx.float32)
    freqs = mx.outer(t, freqs)
    return mx.cos(freqs), mx.sin(freqs)


def apply_rotary_emb(x: mx.array, freqs_cis: RotaryCache) -> mx.array:
    cos, sin = freqs_cis
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_odd * cos + x_even * sin
    return mx.stack([x_rot_even, x_rot_odd], axis=-1).reshape(x.shape)


def get_timestep_embedding(timestep: mx.array, embed_size: int) -> mx.array:
    if embed_size % 2 != 0:
        raise ValueError("embed_size must be even")

    half = embed_size // 2
    base = mx.log(mx.array(10000.0, dtype=mx.float32))
    freqs = 1000.0 * mx.exp(
        -base * mx.arange(start=0, stop=half, dtype=mx.float32) / float(half)
    )
    args = timestep[..., None] * freqs[None, :]
    embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    return embedding.astype(timestep.dtype)


def _bool_to_additive_mask(mask: mx.array) -> mx.array:
    zero = mx.zeros(mask.shape, dtype=mx.float32)
    neg_inf = mx.full(mask.shape, -1e9, dtype=mx.float32)
    return mx.where(mask, zero, neg_inf)[:, None, :, :]


def _make_causal_mask(seq_len: int) -> mx.array:
    row = mx.arange(seq_len)[:, None]
    col = mx.arange(seq_len)[None, :]
    return row >= col


class LowRankAdaLN(nn.Module):
    def __init__(self, model_size: int, rank: int, eps: float):
        super().__init__()
        self.eps = eps

        self.shift_down = nn.Linear(model_size, rank, bias=False)
        self.scale_down = nn.Linear(model_size, rank, bias=False)
        self.gate_down = nn.Linear(model_size, rank, bias=False)

        self.shift_up = nn.Linear(rank, model_size, bias=True)
        self.scale_up = nn.Linear(rank, model_size, bias=True)
        self.gate_up = nn.Linear(rank, model_size, bias=True)

    def __call__(self, x: mx.array, cond_embed: mx.array) -> Tuple[mx.array, mx.array]:
        shift, scale, gate = mx.split(cond_embed, 3, axis=-1)

        shift = self.shift_up(self.shift_down(nn.silu(shift))) + shift
        scale = self.scale_up(self.scale_down(nn.silu(scale))) + scale
        gate = self.gate_up(self.gate_down(nn.silu(gate))) + gate

        x_dtype = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        x = x * (scale + 1.0) + shift
        gate = mx.tanh(gate)
        return x.astype(x_dtype), gate


class RMSNorm(nn.Module):
    def __init__(self, model_size: int | Tuple[int, int], eps: float):
        super().__init__()
        self.eps = eps
        if isinstance(model_size, int):
            model_size = (model_size,)
        self.weight = mx.ones(model_size)

    def __call__(self, x: mx.array) -> mx.array:
        x_dtype = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        x = x * self.weight
        return x.astype(x_dtype)


class SelfAttention(nn.Module):
    def __init__(
        self,
        model_size: int,
        num_heads: int,
        is_causal: bool,
        norm_eps: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.is_causal = is_causal

        self.wq = nn.Linear(model_size, model_size, bias=False)
        self.wk = nn.Linear(model_size, model_size, bias=False)
        self.wv = nn.Linear(model_size, model_size, bias=False)
        self.wo = nn.Linear(model_size, model_size, bias=False)
        self.gate = nn.Linear(model_size, model_size, bias=False)

        if model_size % num_heads != 0:
            raise ValueError("model_size must be divisible by num_heads")
        self.head_dim = model_size // num_heads

        self.q_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None,
        freqs_cis: RotaryCache,
    ) -> mx.array:
        batch_size, seq_len = x.shape[:2]

        xq = self.wq(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = self.wk(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = self.wv(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        gate = self.gate(x)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = apply_rotary_emb(xq, (freqs_cis[0][:seq_len], freqs_cis[1][:seq_len]))
        xk = apply_rotary_emb(xk, (freqs_cis[0][:seq_len], freqs_cis[1][:seq_len]))

        attn_mask_bool = None
        if mask is not None:
            key_mask = mx.broadcast_to(mask[:, None, :], (batch_size, seq_len, seq_len))
            attn_mask_bool = key_mask
        if self.is_causal:
            causal = mx.broadcast_to(
                _make_causal_mask(seq_len)[None, :, :], (batch_size, seq_len, seq_len)
            )
            attn_mask_bool = (
                causal
                if attn_mask_bool is None
                else mx.logical_and(attn_mask_bool, causal)
            )

        attn_mask = None
        if attn_mask_bool is not None:
            attn_mask = _bool_to_additive_mask(attn_mask_bool)

        output = mx.fast.scaled_dot_product_attention(
            q=mx.transpose(xq, (0, 2, 1, 3)),
            k=mx.transpose(xk, (0, 2, 1, 3)),
            v=mx.transpose(xv, (0, 2, 1, 3)),
            scale=1.0 / math.sqrt(self.head_dim),
            mask=attn_mask,
        )
        output = mx.transpose(output, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
        output = output * mx.sigmoid(gate)
        return self.wo(output)


class JointAttention(nn.Module):
    def __init__(
        self,
        model_size: int,
        num_heads: int,
        text_model_size: int,
        speaker_model_size: int,
        speaker_patch_size: int,
        norm_eps: float,
        use_latent_kv: bool = True,
    ):
        super().__init__()
        self.speaker_patch_size = speaker_patch_size
        self.num_heads = num_heads
        self.use_latent_kv = use_latent_kv

        self.wq = nn.Linear(model_size, model_size, bias=False)
        self.wk = nn.Linear(model_size, model_size, bias=False)
        self.wv = nn.Linear(model_size, model_size, bias=False)

        self.wk_text = nn.Linear(text_model_size, model_size, bias=False)
        self.wv_text = nn.Linear(text_model_size, model_size, bias=False)

        self.wk_speaker = nn.Linear(speaker_model_size, model_size, bias=False)
        self.wv_speaker = nn.Linear(speaker_model_size, model_size, bias=False)

        if use_latent_kv:
            self.wk_latent = nn.Linear(speaker_model_size, model_size, bias=False)
            self.wv_latent = nn.Linear(speaker_model_size, model_size, bias=False)
        else:
            self.wk_latent = None
            self.wv_latent = None

        if model_size % num_heads != 0:
            raise ValueError("model_size must be divisible by num_heads")
        self.head_dim = model_size // num_heads

        self.q_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)

        self.gate = nn.Linear(model_size, model_size, bias=False)
        self.wo = nn.Linear(model_size, model_size, bias=False)

    def _apply_rotary_half(self, y: mx.array, freqs_cis: RotaryCache) -> mx.array:
        half = y.shape[-2] // 2
        y1 = y[..., :half, :]
        y2 = y[..., half:, :]
        y1 = apply_rotary_emb(y1, freqs_cis)
        return mx.concatenate([y1, y2], axis=-2)

    def __call__(
        self,
        x: mx.array,
        text_mask: mx.array,
        speaker_mask: mx.array,
        freqs_cis: RotaryCache,
        kv_cache_text: KVCache,
        kv_cache_speaker: KVCache,
        start_pos: int | None,
        kv_cache_latent: KVCache | None,
    ) -> mx.array:
        batch_size, seq_len = x.shape[:2]

        xq = self.wq(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        xk_self = self.wk(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        xv_self = self.wv(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk_self = self.k_norm(xk_self)
        gate = self.gate(x)

        if start_pos is None:
            start_pos = 0

        q_cos = freqs_cis[0][start_pos : start_pos + seq_len]
        q_sin = freqs_cis[1][start_pos : start_pos + seq_len]
        xq = self._apply_rotary_half(xq, (q_cos, q_sin))
        xk_self = self._apply_rotary_half(xk_self, (q_cos, q_sin))

        xk_text, xv_text = kv_cache_text
        xk_speaker, xv_speaker = kv_cache_speaker

        if kv_cache_latent is None or kv_cache_latent[0].shape[1] == 0:
            xk_latent = mx.zeros(
                (batch_size, 0, self.num_heads, self.head_dim), dtype=x.dtype
            )
            xv_latent = mx.zeros(
                (batch_size, 0, self.num_heads, self.head_dim), dtype=x.dtype
            )
            latent_mask = mx.zeros((batch_size, 0), dtype=mx.bool_)
        else:
            xk_latent, xv_latent = kv_cache_latent
            latent_positions = (
                mx.arange(xk_latent.shape[1], dtype=mx.int32) * self.speaker_patch_size
            )
            latent_mask = mx.broadcast_to(
                latent_positions[None, :] < start_pos,
                (batch_size, xk_latent.shape[1]),
            )

        xk = mx.concatenate([xk_self, xk_latent, xk_text, xk_speaker], axis=1)
        xv = mx.concatenate([xv_self, xv_latent, xv_text, xv_speaker], axis=1)

        self_mask = mx.ones((batch_size, seq_len), dtype=mx.bool_)
        mask = mx.concatenate([self_mask, latent_mask, text_mask, speaker_mask], axis=1)
        mask = mx.broadcast_to(mask[:, None, :], (batch_size, seq_len, mask.shape[1]))
        attn_mask = _bool_to_additive_mask(mask)

        output = mx.fast.scaled_dot_product_attention(
            q=mx.transpose(xq, (0, 2, 1, 3)),
            k=mx.transpose(xk, (0, 2, 1, 3)),
            v=mx.transpose(xv, (0, 2, 1, 3)),
            scale=1.0 / math.sqrt(self.head_dim),
            mask=attn_mask,
        )
        output = mx.transpose(output, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
        output = output * mx.sigmoid(gate)
        return self.wo(output)

    def get_kv_cache_text(self, text_state: mx.array) -> KVCache:
        batch_size = text_state.shape[0]
        xk = self.wk_text(text_state).reshape(
            batch_size, text_state.shape[1], self.num_heads, self.head_dim
        )
        xv = self.wv_text(text_state).reshape(
            batch_size, text_state.shape[1], self.num_heads, self.head_dim
        )
        xk = self.k_norm(xk)
        return xk, xv

    def get_kv_cache_speaker(self, speaker_state: mx.array) -> KVCache:
        batch_size = speaker_state.shape[0]
        xk = self.wk_speaker(speaker_state).reshape(
            batch_size, speaker_state.shape[1], self.num_heads, self.head_dim
        )
        xv = self.wv_speaker(speaker_state).reshape(
            batch_size, speaker_state.shape[1], self.num_heads, self.head_dim
        )
        xk = self.k_norm(xk)
        return xk, xv

    def get_kv_cache_latent(
        self, latent_state: mx.array, freqs_cis: RotaryCache
    ) -> KVCache:
        if not self.use_latent_kv or self.wk_latent is None or self.wv_latent is None:
            raise ValueError(
                "Latent KV cache modules are disabled. Use a model config with "
                "`delete_blockwise_modules=False` to enable blockwise generation."
            )
        batch_size = latent_state.shape[0]
        seq_len = latent_state.shape[1]
        xk = self.wk_latent(latent_state).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        xv = self.wv_latent(latent_state).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        xk = self.k_norm(xk)
        xk = self._apply_rotary_half(xk, freqs_cis)
        return xk, xv


class MLP(nn.Module):
    def __init__(self, model_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, model_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class EncoderTransformerBlock(nn.Module):
    def __init__(
        self,
        model_size: int,
        num_heads: int,
        intermediate_size: int,
        is_causal: bool,
        norm_eps: float,
    ):
        super().__init__()
        self.attention = SelfAttention(
            model_size=model_size,
            num_heads=num_heads,
            is_causal=is_causal,
            norm_eps=norm_eps,
        )
        self.mlp = MLP(model_size=model_size, intermediate_size=intermediate_size)
        self.attention_norm = RMSNorm(model_size, norm_eps)
        self.mlp_norm = RMSNorm(model_size, norm_eps)

    def __call__(
        self, x: mx.array, mask: mx.array | None, freqs_cis: RotaryCache
    ) -> mx.array:
        x = x + self.attention(self.attention_norm(x), mask, freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_size: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
        text_model_size: int,
        speaker_model_size: int,
        speaker_patch_size: int,
        adaln_rank: int,
        use_latent_kv: bool = True,
    ):
        super().__init__()
        self.attention = JointAttention(
            model_size=model_size,
            num_heads=num_heads,
            text_model_size=text_model_size,
            speaker_model_size=speaker_model_size,
            speaker_patch_size=speaker_patch_size,
            norm_eps=norm_eps,
            use_latent_kv=use_latent_kv,
        )
        self.mlp = MLP(model_size=model_size, intermediate_size=intermediate_size)
        self.attention_adaln = LowRankAdaLN(
            model_size=model_size, rank=adaln_rank, eps=norm_eps
        )
        self.mlp_adaln = LowRankAdaLN(
            model_size=model_size, rank=adaln_rank, eps=norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        cond_embed: mx.array,
        text_mask: mx.array,
        speaker_mask: mx.array,
        freqs_cis: RotaryCache,
        kv_cache_text: KVCache,
        kv_cache_speaker: KVCache,
        start_pos: int | None,
        kv_cache_latent: KVCache | None,
    ) -> mx.array:
        x_norm, attention_gate = self.attention_adaln(x, cond_embed)
        x = x + attention_gate * self.attention(
            x_norm,
            text_mask,
            speaker_mask,
            freqs_cis,
            kv_cache_text,
            kv_cache_speaker,
            start_pos,
            kv_cache_latent,
        )

        x_norm, mlp_gate = self.mlp_adaln(x, cond_embed)
        x = x + mlp_gate * self.mlp(x_norm)
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, model_size)
        self.blocks = [
            EncoderTransformerBlock(
                model_size=model_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                is_causal=False,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.head_dim = model_size // num_heads

    def __call__(self, input_ids: mx.array, mask: mx.array | None = None) -> mx.array:
        x = self.text_embedding(input_ids)
        freqs_cis = precompute_freqs_cis(self.head_dim, input_ids.shape[1])
        for block in self.blocks:
            x = block(x, mask, freqs_cis)
        return x


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        latent_size: int,
        patch_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_proj = nn.Linear(latent_size * patch_size, model_size, bias=True)
        self.blocks = [
            EncoderTransformerBlock(
                model_size=model_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                is_causal=True,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.head_dim = model_size // num_heads

    def __call__(self, latent: mx.array) -> mx.array:
        seq_len = latent.shape[1]
        seq_len_patched = (seq_len // self.patch_size) * self.patch_size
        latent = latent[:, :seq_len_patched]
        x = latent.reshape(
            latent.shape[0],
            seq_len_patched // self.patch_size,
            latent.shape[-1] * self.patch_size,
        )

        x = self.in_proj(x) / 6.0
        freqs_cis = precompute_freqs_cis(self.head_dim, x.shape[1])
        for block in self.blocks:
            x = block(x, None, freqs_cis)
        return x


class EchoDiT(nn.Module):
    def __init__(
        self,
        latent_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
        text_vocab_size: int,
        text_model_size: int,
        text_num_layers: int,
        text_num_heads: int,
        text_intermediate_size: int,
        speaker_patch_size: int,
        speaker_model_size: int,
        speaker_num_layers: int,
        speaker_num_heads: int,
        speaker_intermediate_size: int,
        timestep_embed_size: int,
        adaln_rank: int,
        enable_blockwise_modules: bool = True,
    ):
        super().__init__()
        self.speaker_patch_size = speaker_patch_size
        self.timestep_embed_size = timestep_embed_size
        self.enable_blockwise_modules = enable_blockwise_modules

        self.text_encoder = TextEncoder(
            vocab_size=text_vocab_size,
            model_size=text_model_size,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            intermediate_size=text_intermediate_size,
            norm_eps=norm_eps,
        )
        self.speaker_encoder = SpeakerEncoder(
            latent_size=latent_size,
            patch_size=speaker_patch_size,
            model_size=speaker_model_size,
            num_layers=speaker_num_layers,
            num_heads=speaker_num_heads,
            intermediate_size=speaker_intermediate_size,
            norm_eps=norm_eps,
        )
        if enable_blockwise_modules:
            self.latent_encoder = SpeakerEncoder(
                latent_size=latent_size,
                patch_size=speaker_patch_size,
                model_size=speaker_model_size,
                num_layers=speaker_num_layers,
                num_heads=speaker_num_heads,
                intermediate_size=speaker_intermediate_size,
                norm_eps=norm_eps,
            )
            self.latent_norm = RMSNorm(speaker_model_size, norm_eps)
        else:
            self.latent_encoder = None
            self.latent_norm = None
        self.text_norm = RMSNorm(text_model_size, norm_eps)
        self.speaker_norm = RMSNorm(speaker_model_size, norm_eps)

        self.cond_module = nn.Sequential(
            nn.Linear(timestep_embed_size, model_size, bias=False),
            nn.SiLU(),
            nn.Linear(model_size, model_size, bias=False),
            nn.SiLU(),
            nn.Linear(model_size, model_size * 3, bias=False),
        )

        self.in_proj = nn.Linear(latent_size, model_size, bias=True)
        self.blocks = [
            TransformerBlock(
                model_size=model_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                norm_eps=norm_eps,
                text_model_size=text_model_size,
                speaker_model_size=speaker_model_size,
                speaker_patch_size=speaker_patch_size,
                adaln_rank=adaln_rank,
                use_latent_kv=enable_blockwise_modules,
            )
            for _ in range(num_layers)
        ]
        self.out_norm = RMSNorm(model_size, norm_eps)
        self.out_proj = nn.Linear(model_size, latent_size, bias=True)
        self.head_dim = model_size // num_heads

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        text_mask: mx.array,
        speaker_mask: mx.array,
        kv_cache_text: List[KVCache],
        kv_cache_speaker: List[KVCache],
        start_pos: int | None = None,
        kv_cache_latent: List[KVCache] | None = None,
    ) -> mx.array:
        if start_pos is None:
            start_pos = 0

        max_pos = start_pos + x.shape[1]
        freqs_cis = precompute_freqs_cis(self.head_dim, max_pos)
        speaker_mask = speaker_mask[..., :: self.speaker_patch_size]

        cond_embed = self.cond_module(
            get_timestep_embedding(t, self.timestep_embed_size)
        )
        cond_embed = cond_embed[:, None, :]

        x = self.in_proj(x)
        for i, block in enumerate(self.blocks):
            x = block(
                x=x,
                cond_embed=cond_embed,
                text_mask=text_mask,
                speaker_mask=speaker_mask,
                freqs_cis=freqs_cis,
                kv_cache_text=kv_cache_text[i],
                kv_cache_speaker=kv_cache_speaker[i],
                start_pos=start_pos,
                kv_cache_latent=(
                    kv_cache_latent[i] if kv_cache_latent is not None else None
                ),
            )

        x = self.out_norm(x)
        x = self.out_proj(x)
        return x.astype(mx.float32)

    def get_kv_cache_text(
        self,
        text_input_ids: mx.array,
        text_mask: mx.array | None,
    ) -> List[KVCache]:
        text_state = self.text_encoder(text_input_ids, text_mask)
        text_state = self.text_norm(text_state)
        return [block.attention.get_kv_cache_text(text_state) for block in self.blocks]

    def get_kv_cache_speaker(self, speaker_latent: mx.array) -> List[KVCache]:
        speaker_state = self.speaker_encoder(speaker_latent)
        speaker_state = self.speaker_norm(speaker_state)
        return [
            block.attention.get_kv_cache_speaker(speaker_state) for block in self.blocks
        ]

    def get_kv_cache_latent(self, prefix_latent: mx.array) -> List[KVCache]:
        if (
            not self.enable_blockwise_modules
            or self.latent_encoder is None
            or self.latent_norm is None
        ):
            raise ValueError(
                "Latent prefix modules are disabled. Use a model config with "
                "`delete_blockwise_modules=False` to enable blockwise generation."
            )
        if prefix_latent.shape[1] == 0:
            batch_size = prefix_latent.shape[0]
            return [
                (
                    mx.zeros(
                        (
                            batch_size,
                            0,
                            block.attention.num_heads,
                            block.attention.head_dim,
                        ),
                        dtype=prefix_latent.dtype,
                    ),
                    mx.zeros(
                        (
                            batch_size,
                            0,
                            block.attention.num_heads,
                            block.attention.head_dim,
                        ),
                        dtype=prefix_latent.dtype,
                    ),
                )
                for block in self.blocks
            ]

        latent_state = self.latent_encoder(prefix_latent)
        latent_state = self.latent_norm(latent_state)

        seq_len = latent_state.shape[1]
        max_pos = seq_len * self.speaker_patch_size
        freqs_cis = precompute_freqs_cis(self.head_dim, max_pos)
        positions = mx.arange(seq_len, dtype=mx.int32) * self.speaker_patch_size
        freqs_latent = freqs_cis[0][positions], freqs_cis[1][positions]
        return [
            block.attention.get_kv_cache_latent(latent_state, freqs_latent)
            for block in self.blocks
        ]
