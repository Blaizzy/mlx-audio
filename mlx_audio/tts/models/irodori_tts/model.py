from __future__ import annotations

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import IrodoriDiTConfig

RotaryCache = Tuple[mx.array, mx.array]
KVCache = Tuple[mx.array, mx.array]


# ---------------------------------------------------------------------------
# Positional encoding helpers
# ---------------------------------------------------------------------------


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
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(timestep.dtype)


def _bool_to_additive_mask(mask: mx.array) -> mx.array:
    """Convert boolean mask (B, Sq, Sk) to additive float mask (B, 1, Sq, Sk)."""
    zero = mx.zeros(mask.shape, dtype=mx.float32)
    neg_inf = mx.full(mask.shape, -1e9, dtype=mx.float32)
    return mx.where(mask, zero, neg_inf)[:, None, :, :]


def patch_sequence_with_mask(
    seq: mx.array,
    mask: mx.array,
    patch_size: int,
) -> Tuple[mx.array, mx.array]:
    """
    Patch along the sequence axis.
      seq  : (B, S, D)       -> (B, S//patch, D*patch)
      mask : (B, S) bool     -> (B, S//patch) bool  (True iff all tokens in patch are valid)
    """
    if patch_size <= 1:
        return seq, mask
    bsz, seq_len, dim = seq.shape
    usable = (seq_len // patch_size) * patch_size
    seq = seq[:, :usable].reshape(bsz, usable // patch_size, dim * patch_size)
    mask = mask[:, :usable].reshape(bsz, usable // patch_size, patch_size)
    mask = mx.all(mask, axis=-1)
    return seq, mask


def _safe_attention_mask(
    x: mx.array,
    mask: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Ensure at least one position is valid per batch for attention pooling."""
    if mask.ndim != 2 or mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[1]:
        raise ValueError(
            f"mask must have shape (B, S) matching x, got x={tuple(x.shape)} "
            f"mask={tuple(mask.shape)}"
        )
    mask = mask.astype(mx.bool_)
    has_any = mx.any(mask, axis=1)
    if mx.all(has_any):
        return x, mask
    if x.shape[1] <= 0:
        raise ValueError("Cannot attention-pool an empty sequence.")
    x = mx.copy(x)
    mask = mx.copy(mask)
    x = mx.where(has_any[:, None, None], x, mx.zeros_like(x))
    # Set first position valid for batches with no valid positions
    fallback = ~has_any
    mask = mx.where(
        fallback[:, None],
        mx.concatenate([mx.ones((x.shape[0], 1), dtype=mx.bool_), mask[:, 1:]], axis=1),
        mask,
    )
    return x, mask


# ---------------------------------------------------------------------------
# Normalisation layers
# ---------------------------------------------------------------------------


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
        return (x * self.weight).astype(x_dtype)


class LowRankAdaLN(nn.Module):
    """
    Low-rank adaptive layer normalisation with shift/scale/gate from timestep embedding.
    Matches Irodori's LowRankAdaLN exactly (including residual connection on each branch).
    """

    def __init__(self, model_dim: int, rank: int, eps: float):
        super().__init__()
        self.eps = eps
        rank = max(1, min(int(rank), int(model_dim)))
        self.shift_down = nn.Linear(model_dim, rank, bias=False)
        self.scale_down = nn.Linear(model_dim, rank, bias=False)
        self.gate_down = nn.Linear(model_dim, rank, bias=False)
        self.shift_up = nn.Linear(rank, model_dim, bias=True)
        self.scale_up = nn.Linear(rank, model_dim, bias=True)
        self.gate_up = nn.Linear(rank, model_dim, bias=True)

    def __call__(self, x: mx.array, cond_embed: mx.array) -> Tuple[mx.array, mx.array]:
        shift, scale, gate = mx.split(cond_embed, 3, axis=-1)
        shift = self.shift_up(self.shift_down(nn.silu(shift))) + shift
        scale = self.scale_up(self.scale_down(nn.silu(scale))) + scale
        gate = self.gate_up(self.gate_down(nn.silu(gate))) + gate

        x_dtype = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        x = x * (1.0 + scale) + shift
        gate = mx.tanh(gate)
        return x.astype(x_dtype), gate


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU MLP: w2(silu(w1(x)) * w3(x))."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """Non-causal self-attention with RoPE and output gate (used in encoders)."""

    def __init__(self, dim: int, heads: int, norm_eps: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm((heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((heads, self.head_dim), eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        key_mask: Optional[mx.array],
        freqs_cis: RotaryCache,
    ) -> mx.array:
        bsz, seq_len = x.shape[:2]
        q = self.wq(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        k = self.wk(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        v = self.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        gate = self.gate(x)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary_emb(q, (freqs_cis[0][:seq_len], freqs_cis[1][:seq_len]))
        k = apply_rotary_emb(k, (freqs_cis[0][:seq_len], freqs_cis[1][:seq_len]))

        attn_mask = None
        if key_mask is not None:
            m = mx.broadcast_to(key_mask[:, None, :], (bsz, seq_len, seq_len))
            attn_mask = _bool_to_additive_mask(m)

        out = mx.fast.scaled_dot_product_attention(
            q=mx.transpose(q, (0, 2, 1, 3)),
            k=mx.transpose(k, (0, 2, 1, 3)),
            v=mx.transpose(v, (0, 2, 1, 3)),
            scale=1.0 / math.sqrt(self.head_dim),
            mask=attn_mask,
        )
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(bsz, seq_len, -1)
        return self.wo(out * mx.sigmoid(gate))


class JointAttention(nn.Module):
    """
    Joint attention over latent self-tokens, text context, and speaker/caption context.
    Uses half-RoPE: RoPE applied to the first half of head dimensions.

    speaker_ctx_dim and/or caption_ctx_dim may be provided independently.
    For dual-mode (v3 VoiceDesign), both are set and keys for both branches
    (wk_speaker/wv_speaker and wk_caption/wv_caption) are created.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        text_ctx_dim: int,
        speaker_ctx_dim: Optional[int],
        norm_eps: float,
        caption_ctx_dim: Optional[int] = None,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wk_text = nn.Linear(text_ctx_dim, dim, bias=False)
        self.wv_text = nn.Linear(text_ctx_dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm((heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((heads, self.head_dim), eps=norm_eps)

        self.has_speaker_condition = speaker_ctx_dim is not None
        self.has_caption_condition = caption_ctx_dim is not None
        if not self.has_speaker_condition and not self.has_caption_condition:
            raise ValueError(
                "At least one of speaker_ctx_dim or caption_ctx_dim must be set"
            )
        if speaker_ctx_dim is not None:
            self.wk_speaker = nn.Linear(speaker_ctx_dim, dim, bias=False)
            self.wv_speaker = nn.Linear(speaker_ctx_dim, dim, bias=False)
        if caption_ctx_dim is not None:
            self.wk_caption = nn.Linear(caption_ctx_dim, dim, bias=False)
            self.wv_caption = nn.Linear(caption_ctx_dim, dim, bias=False)

    def _apply_rotary_half(self, y: mx.array, freqs_cis: RotaryCache) -> mx.array:
        """Apply RoPE to the first half of head dimensions only."""
        half = y.shape[-2] // 2
        y1 = apply_rotary_emb(y[..., :half, :], freqs_cis)
        return mx.concatenate([y1, y[..., half:, :]], axis=-2)

    def get_kv_cache_text(self, text_state: mx.array) -> KVCache:
        bsz = text_state.shape[0]
        k = self.wk_text(text_state).reshape(
            bsz, text_state.shape[1], self.heads, self.head_dim
        )
        v = self.wv_text(text_state).reshape(
            bsz, text_state.shape[1], self.heads, self.head_dim
        )
        k = self.k_norm(k)
        return k, v

    def get_kv_cache_speaker(self, speaker_state: mx.array) -> KVCache:
        bsz = speaker_state.shape[0]
        k = self.wk_speaker(speaker_state).reshape(
            bsz, speaker_state.shape[1], self.heads, self.head_dim
        )
        v = self.wv_speaker(speaker_state).reshape(
            bsz, speaker_state.shape[1], self.heads, self.head_dim
        )
        k = self.k_norm(k)
        return k, v

    def get_kv_cache_caption(self, caption_state: mx.array) -> KVCache:
        bsz = caption_state.shape[0]
        k = self.wk_caption(caption_state).reshape(
            bsz, caption_state.shape[1], self.heads, self.head_dim
        )
        v = self.wv_caption(caption_state).reshape(
            bsz, caption_state.shape[1], self.heads, self.head_dim
        )
        k = self.k_norm(k)
        return k, v

    def __call__(
        self,
        x: mx.array,
        text_mask: mx.array,
        freqs_cis: RotaryCache,
        kv_cache_text: KVCache,
        kv_cache_speaker: Optional[KVCache] = None,
        speaker_mask: Optional[mx.array] = None,
        kv_cache_caption: Optional[KVCache] = None,
        caption_mask: Optional[mx.array] = None,
        start_pos: int = 0,
    ) -> mx.array:
        bsz, seq_len = x.shape[:2]
        q = self.wq(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        k_self = self.wk(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        v_self = self.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        gate = self.gate(x)

        q = self.q_norm(q)
        k_self = self.k_norm(k_self)

        q_cos = freqs_cis[0][start_pos : start_pos + seq_len]
        q_sin = freqs_cis[1][start_pos : start_pos + seq_len]
        q = self._apply_rotary_half(q, (q_cos, q_sin))
        k_self = self._apply_rotary_half(k_self, (q_cos, q_sin))

        k_text, v_text = kv_cache_text
        self_mask = mx.ones((bsz, seq_len), dtype=mx.bool_)

        k_parts = [k_self, k_text]
        v_parts = [v_self, v_text]
        mask_parts = [self_mask, text_mask]

        if kv_cache_speaker is not None:
            k_spk, v_spk = kv_cache_speaker
            k_parts.append(k_spk)
            v_parts.append(v_spk)
            mask_parts.append(speaker_mask)

        if kv_cache_caption is not None:
            k_cap, v_cap = kv_cache_caption
            k_parts.append(k_cap)
            v_parts.append(v_cap)
            mask_parts.append(caption_mask)

        k = mx.concatenate(k_parts, axis=1)
        v = mx.concatenate(v_parts, axis=1)
        full_mask = mx.concatenate(mask_parts, axis=1)
        full_mask = mx.broadcast_to(
            full_mask[:, None, :], (bsz, seq_len, full_mask.shape[1])
        )
        attn_mask = _bool_to_additive_mask(full_mask)

        out = mx.fast.scaled_dot_product_attention(
            q=mx.transpose(q, (0, 2, 1, 3)),
            k=mx.transpose(k, (0, 2, 1, 3)),
            v=mx.transpose(v, (0, 2, 1, 3)),
            scale=1.0 / math.sqrt(self.head_dim),
            mask=attn_mask,
        )
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(bsz, seq_len, -1)
        return self.wo(out * mx.sigmoid(gate))


# ---------------------------------------------------------------------------
# Encoder blocks
# ---------------------------------------------------------------------------


class TextBlock(nn.Module):
    """Transformer block used in both TextEncoder and ReferenceLatentEncoder."""

    def __init__(self, dim: int, heads: int, mlp_hidden_dim: int, norm_eps: float):
        super().__init__()
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = SelfAttention(dim, heads, norm_eps=norm_eps)
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim, mlp_hidden_dim)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array], freqs_cis: RotaryCache
    ) -> mx.array:
        x = x + self.attention(
            self.attention_norm(x), key_mask=mask, freqs_cis=freqs_cis
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TextEncoder(nn.Module):
    """
    Text encoder: embedding + non-causal Transformer blocks.
    Applies mask zeroing after each block so fully-masked positions stay zero.
    Used for both text and caption encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        heads: int,
        num_layers: int,
        mlp_ratio: float,
        norm_eps: float,
    ):
        super().__init__()
        self.head_dim = dim // heads
        self.text_embedding = nn.Embedding(vocab_size, dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.blocks = [
            TextBlock(dim, heads, mlp_hidden, norm_eps) for _ in range(num_layers)
        ]

    def __call__(
        self, input_ids: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:
        x = self.text_embedding(input_ids)
        freqs_cis = precompute_freqs_cis(self.head_dim, input_ids.shape[1])
        if mask is not None:
            mask_f = mask[..., None].astype(x.dtype)
            x = x * mask_f
            for block in self.blocks:
                x = block(x, mask=mask, freqs_cis=freqs_cis)
                x = x * mask_f
            return x * mask_f
        else:
            for block in self.blocks:
                x = block(x, mask=None, freqs_cis=freqs_cis)
            return x


class ReferenceLatentEncoder(nn.Module):
    """
    Encoder for reference (speaker) audio latents.
    Receives already-patched DACVAE latents: (B, S, latent_dim * speaker_patch_size).
    Uses non-causal attention (unlike Echo TTS which uses causal).
    """

    def __init__(
        self,
        in_dim: int,
        dim: int,
        heads: int,
        num_layers: int,
        mlp_ratio: float,
        norm_eps: float,
    ):
        super().__init__()
        self.head_dim = dim // heads
        self.in_proj = nn.Linear(in_dim, dim, bias=True)
        mlp_hidden = int(dim * mlp_ratio)
        self.blocks = [
            TextBlock(dim, heads, mlp_hidden, norm_eps) for _ in range(num_layers)
        ]

    def __call__(self, latent: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = self.in_proj(latent) / 6.0
        freqs_cis = precompute_freqs_cis(self.head_dim, x.shape[1])
        if mask is not None:
            mask_f = mask[..., None].astype(x.dtype)
            x = x * mask_f
            for block in self.blocks:
                x = block(x, mask=mask, freqs_cis=freqs_cis)
                x = x * mask_f
            return x * mask_f
        else:
            for block in self.blocks:
                x = block(x, mask=None, freqs_cis=freqs_cis)
            return x


# ---------------------------------------------------------------------------
# Diffusion block
# ---------------------------------------------------------------------------


class DiffusionBlock(nn.Module):
    """
    Single DiT block: JointAttention + SwiGLU, both conditioned via LowRankAdaLN.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_hidden_dim: int,
        text_ctx_dim: int,
        speaker_ctx_dim: Optional[int],
        adaln_rank: int,
        norm_eps: float,
        caption_ctx_dim: Optional[int] = None,
    ):
        super().__init__()
        self.attention = JointAttention(
            dim,
            heads,
            text_ctx_dim,
            speaker_ctx_dim,
            norm_eps,
            caption_ctx_dim=caption_ctx_dim,
        )
        self.mlp = SwiGLU(dim, mlp_hidden_dim)
        self.attention_adaln = LowRankAdaLN(dim, adaln_rank, norm_eps)
        self.mlp_adaln = LowRankAdaLN(dim, adaln_rank, norm_eps)

    def __call__(
        self,
        x: mx.array,
        cond_embed: mx.array,
        text_mask: mx.array,
        freqs_cis: RotaryCache,
        kv_cache_text: KVCache,
        kv_cache_speaker: Optional[KVCache] = None,
        speaker_mask: Optional[mx.array] = None,
        kv_cache_caption: Optional[KVCache] = None,
        caption_mask: Optional[mx.array] = None,
        start_pos: int = 0,
    ) -> mx.array:
        x_norm, attn_gate = self.attention_adaln(x, cond_embed)
        x = x + attn_gate * self.attention(
            x_norm,
            text_mask,
            freqs_cis,
            kv_cache_text,
            kv_cache_speaker,
            speaker_mask,
            kv_cache_caption,
            caption_mask,
            start_pos,
        )
        x_norm, mlp_gate = self.mlp_adaln(x, cond_embed)
        x = x + mlp_gate * self.mlp(x_norm)
        return x


# ---------------------------------------------------------------------------
# Duration predictor components (v3)
# ---------------------------------------------------------------------------


class DurationSwiGLUBlock(nn.Module):
    """SwiGLU block with optional AdaRN-Zero modulation for duration predictor.

    Supports dual modulation (speaker + caption) that are added together,
    matching the token_sum_dual_adarn_zero_no_aux architecture.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        norm_eps: float,
        cond_dim: Optional[int] = None,
        caption_cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim, hidden_dim)
        self.cond_dim = cond_dim
        self.caption_cond_dim = caption_cond_dim
        self.modulation = None
        self.caption_modulation = None
        if cond_dim is not None:
            self.modulation = nn.Linear(cond_dim, dim * 3)
            self.modulation.weight = self.modulation.weight * 0.0
            self.modulation.bias = self.modulation.bias * 0.0
        if caption_cond_dim is not None:
            self.caption_modulation = nn.Linear(caption_cond_dim, dim * 3)
            self.caption_modulation.weight = self.caption_modulation.weight * 0.0
            self.caption_modulation.bias = self.caption_modulation.bias * 0.0

    def __call__(
        self,
        x: mx.array,
        cond: Optional[mx.array] = None,
        caption_cond: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.norm(x)
        if self.modulation is not None or self.caption_modulation is not None:
            shift = mx.zeros_like(h)
            scale = mx.zeros_like(h)
            gate = mx.zeros_like(h)
            if self.modulation is not None:
                if cond is None:
                    raise ValueError("cond is required for AdaRN-Zero duration blocks.")
                ds, dsc, dg = mx.split(self.modulation(nn.silu(cond)), 3, axis=-1)
                if h.ndim == 3 and ds.ndim == 2:
                    ds, dsc, dg = ds[:, None, :], dsc[:, None, :], dg[:, None, :]
                shift = shift + ds
                scale = scale + dsc
                gate = gate + dg
            if self.caption_modulation is not None:
                if caption_cond is None:
                    raise ValueError(
                        "caption_cond is required for caption AdaRN-Zero blocks."
                    )
                cs, csc, cg = mx.split(
                    self.caption_modulation(nn.silu(caption_cond)), 3, axis=-1
                )
                if h.ndim == 3 and cs.ndim == 2:
                    cs, csc, cg = cs[:, None, :], csc[:, None, :], cg[:, None, :]
                shift = shift + cs
                scale = scale + csc
                gate = gate + cg
            h = h * (1.0 + scale) + shift
            return x + mx.tanh(gate) * self.mlp(h)
        return x + self.mlp(h)


class AttentionPooling(nn.Module):
    """Attention pooling to reduce sequence to single vector."""

    def __init__(self, dim: int, heads: int, norm_eps: float):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.query = mx.zeros((1, 1, dim))
        self.q_norm = RMSNorm(dim, eps=norm_eps)
        self.k_norm = RMSNorm(dim, eps=norm_eps)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"x must have shape (B, S, {self.dim}), got {tuple(x.shape)}"
            )
        x, mask = _safe_attention_mask(x, mask)
        bsz, seq_len, _ = x.shape
        q = mx.broadcast_to(self.query.astype(x.dtype), (bsz, 1, self.dim))
        q = self.wq(self.q_norm(q)).reshape(bsz, 1, self.heads, self.head_dim)
        k = self.wk(self.k_norm(x)).reshape(bsz, seq_len, self.heads, self.head_dim)
        v = self.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        attn_mask = _bool_to_additive_mask(mask)
        y = mx.fast.scaled_dot_product_attention(
            q=mx.transpose(q, (0, 2, 1, 3)),
            k=mx.transpose(k, (0, 2, 1, 3)),
            v=mx.transpose(v, (0, 2, 1, 3)),
            scale=1.0 / math.sqrt(self.head_dim),
            mask=attn_mask,
        )
        y = mx.transpose(y, (0, 2, 1, 3)).reshape(bsz, 1, self.dim)
        return self.wo(y).squeeze(1)


class CrossAttentionPooling(nn.Module):
    """Cross-attention pooling: query attends to context sequence."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        output_dim: int,
        heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.heads = heads
        self.head_dim = output_dim // heads
        self.q_norm = RMSNorm(query_dim, eps=norm_eps)
        self.k_norm = RMSNorm(context_dim, eps=norm_eps)
        self.wq = nn.Linear(query_dim, output_dim, bias=False)
        self.wk = nn.Linear(context_dim, output_dim, bias=False)
        self.wv = nn.Linear(context_dim, output_dim, bias=False)
        self.wo = nn.Linear(output_dim, output_dim, bias=False)

    def __call__(
        self,
        query: mx.array,
        context: mx.array,
        context_mask: mx.array,
    ) -> mx.array:
        if query.ndim != 2 or query.shape[-1] != self.query_dim:
            raise ValueError(
                f"query must have shape (B, {self.query_dim}), got {tuple(query.shape)}"
            )
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must have shape (B, S, {self.context_dim}), got {tuple(context.shape)}"
            )
        context, context_mask = _safe_attention_mask(context, context_mask)
        bsz, seq_len, _ = context.shape
        q = query[:, None, :]
        q = self.wq(self.q_norm(q)).reshape(bsz, 1, self.heads, self.head_dim)
        k = self.wk(self.k_norm(context)).reshape(
            bsz, seq_len, self.heads, self.head_dim
        )
        v = self.wv(context).reshape(bsz, seq_len, self.heads, self.head_dim)
        attn_mask = _bool_to_additive_mask(context_mask)
        y = mx.fast.scaled_dot_product_attention(
            q=mx.transpose(q, (0, 2, 1, 3)),
            k=mx.transpose(k, (0, 2, 1, 3)),
            v=mx.transpose(v, (0, 2, 1, 3)),
            scale=1.0 / math.sqrt(self.head_dim),
            mask=attn_mask,
        )
        y = mx.transpose(y, (0, 2, 1, 3)).reshape(bsz, 1, self.output_dim)
        return self.wo(y).squeeze(1)


class DurationPredictor(nn.Module):
    """
    Duration predictor that regresses log1p(num_frames) from text state + aux features.

    Ported from Irodori-TTS v3 (Aratako/Irodori-TTS).
    Supports multiple speaker fusion strategies:
      - concat: simple concatenation
      - adarn: AdaRN modulation
      - adarn_zero: AdaRN-Zero with zero-init modulation (v3 default)
      - speaker_cross_attn: cross-attention pooling over speaker sequence
      - text_cross_attn: cross-attention pooling over text sequence

    The default v3 architecture is 'token_sum_adarn_zero_no_aux', which
    processes each text token through DurationSwiGLUBlocks with speaker
    AdaRN-Zero modulation, then sums frame predictions.
    """

    def __init__(
        self,
        *,
        text_dim: int,
        aux_dim: int,
        hidden_dim: int,
        layers: int,
        norm_eps: float,
        speaker_dim: Optional[int] = None,
        speaker_fusion: str = "concat",
        caption_dim: Optional[int] = None,
        caption_fusion: str = "adarn_zero",
        caption_pooling: str = "masked_mean",
        attention_heads: int = 8,
        architecture: str = "token_sum_adarn_zero_no_aux",
        token_init_frames: float = 9.0,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim
        self.speaker_fusion = speaker_fusion
        self.caption_dim = caption_dim
        self.caption_fusion = caption_fusion
        self.caption_pooling = caption_pooling
        self.duration_architecture = architecture

        self.null_speaker = None
        if speaker_dim is not None:
            self.null_speaker = mx.zeros((speaker_dim,))

        self.null_caption = None
        if caption_dim is not None:
            self.null_caption = mx.zeros((caption_dim,))

        # Token-sum architecture (v3 default)
        self.token_input_proj = None
        self.token_blocks = None
        self.token_out_norm = None
        self.token_out_proj = None

        # Pooled architecture components
        self.text_pool = None
        self.text_adarn_norm = None
        self.text_adarn = None
        self.speaker_cross_attn = None
        self.text_cross_attn = None
        self.input_proj = None
        self.blocks = None
        self.out_norm = None
        self.out_proj = None

        if architecture == "token_sum_adarn_zero_no_aux":
            self.token_input_proj = nn.Linear(text_dim, hidden_dim)
            self.token_blocks = [
                DurationSwiGLUBlock(
                    dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps,
                    cond_dim=speaker_dim,
                )
                for _ in range(layers)
            ]
            self.token_out_norm = RMSNorm(hidden_dim, eps=norm_eps)
            self.token_out_proj = nn.Linear(hidden_dim, 1)
            self.token_out_proj.weight = self.token_out_proj.weight * 0.0
            bias_init = math.log(math.expm1(token_init_frames))
            self.token_out_proj.bias = mx.full(
                self.token_out_proj.bias.shape, bias_init
            )
            return

        if architecture == "token_sum_dual_adarn_zero_no_aux":
            self.token_input_proj = nn.Linear(text_dim, hidden_dim)
            self.token_blocks = [
                DurationSwiGLUBlock(
                    dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps,
                    cond_dim=speaker_dim,
                    caption_cond_dim=caption_dim,
                )
                for _ in range(layers)
            ]
            self.token_out_norm = RMSNorm(hidden_dim, eps=norm_eps)
            self.token_out_proj = nn.Linear(hidden_dim, 1)
            self.token_out_proj.weight = self.token_out_proj.weight * 0.0
            bias_init = math.log(math.expm1(token_init_frames))
            self.token_out_proj.bias = mx.full(
                self.token_out_proj.bias.shape, bias_init
            )
            return

        # Pooled architecture
        self.text_pool = AttentionPooling(
            dim=text_dim,
            heads=attention_heads,
            norm_eps=norm_eps,
        )

        if speaker_dim is not None:
            if speaker_fusion == "concat":
                input_dim = text_dim + speaker_dim + aux_dim
            elif speaker_fusion == "adarn":
                input_dim = text_dim + aux_dim
                self.text_adarn_norm = RMSNorm(text_dim, eps=norm_eps)
                self.text_adarn = nn.Linear(speaker_dim, text_dim * 2)
                self.text_adarn.weight = self.text_adarn.weight * 0.0
                self.text_adarn.bias = self.text_adarn.bias * 0.0
            elif speaker_fusion == "adarn_zero":
                input_dim = text_dim + aux_dim
            elif speaker_fusion == "speaker_cross_attn":
                input_dim = text_dim * 2 + aux_dim
                self.speaker_cross_attn = CrossAttentionPooling(
                    query_dim=text_dim,
                    context_dim=speaker_dim,
                    output_dim=text_dim,
                    heads=attention_heads,
                    norm_eps=norm_eps,
                )
            elif speaker_fusion == "text_cross_attn":
                input_dim = text_dim + speaker_dim + aux_dim
                self.text_cross_attn = CrossAttentionPooling(
                    query_dim=speaker_dim,
                    context_dim=text_dim,
                    output_dim=text_dim,
                    heads=attention_heads,
                    norm_eps=norm_eps,
                )
            else:
                raise ValueError(
                    f"Unsupported duration speaker fusion: {speaker_fusion!r}"
                )
        else:
            input_dim = text_dim + aux_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        block_cond_dim = speaker_dim if speaker_fusion == "adarn_zero" else None
        self.blocks = [
            DurationSwiGLUBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                norm_eps=norm_eps,
                cond_dim=block_cond_dim,
            )
            for _ in range(layers)
        ]
        self.out_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def _speaker_vec(
        self,
        batch_size: int,
        dtype,
        speaker_state: Optional[mx.array],
        has_speaker: mx.array,
    ) -> mx.array:
        if self.null_speaker is None or self.speaker_dim is None:
            raise RuntimeError("Duration speaker modules are missing.")
        null_vec = mx.broadcast_to(
            self.null_speaker.astype(dtype)[None, :], (batch_size, self.speaker_dim)
        )
        if speaker_state is None:
            return null_vec
        speaker_vec = speaker_state[:, 0].astype(dtype)
        return mx.where(has_speaker[:, None], speaker_vec, null_vec)

    def _caption_vec(
        self,
        batch_size: int,
        dtype,
        caption_state: Optional[mx.array],
        caption_mask: Optional[mx.array],
        has_caption: mx.array,
    ) -> mx.array:
        if self.null_caption is None or self.caption_dim is None:
            raise RuntimeError("Duration caption modules are missing.")
        null_vec = mx.broadcast_to(
            self.null_caption.astype(dtype)[None, :], (batch_size, self.caption_dim)
        )
        if caption_state is None:
            return null_vec
        caption_state = caption_state.astype(dtype)
        if caption_mask is not None:
            mask_f = caption_mask[..., None].astype(dtype)
            denom = mx.maximum(mx.sum(mask_f, axis=1), 1.0)
            caption_vec = mx.sum(caption_state * mask_f, axis=1) / denom
        else:
            caption_vec = caption_state.mean(axis=1)
        return mx.where(has_caption[:, None], caption_vec, null_vec)

    def __call__(
        self,
        text_state: mx.array,
        text_mask: mx.array,
        aux_features: mx.array,
        speaker_state: Optional[mx.array] = None,
        speaker_mask: Optional[mx.array] = None,
        has_speaker: Optional[mx.array] = None,
        caption_state: Optional[mx.array] = None,
        caption_mask: Optional[mx.array] = None,
        has_caption: Optional[mx.array] = None,
    ) -> mx.array:
        if text_state.ndim != 3 or text_state.shape[-1] != self.text_dim:
            raise ValueError(
                f"text_state must have shape (B, S, {self.text_dim}), got {tuple(text_state.shape)}"
            )
        if aux_features.ndim != 2 or aux_features.shape[1] != self.aux_dim:
            raise ValueError(
                f"aux_features must have shape (B, {self.aux_dim}), got {tuple(aux_features.shape)}"
            )
        text_state, text_mask = _safe_attention_mask(text_state, text_mask)
        aux_features = aux_features.astype(text_state.dtype)

        # Token-sum architecture (speaker-only)
        if self.duration_architecture == "token_sum_adarn_zero_no_aux":
            if self.speaker_dim is None:
                raise RuntimeError(
                    "Token-sum duration architecture requires speaker modules."
                )
            if has_speaker is None:
                raise ValueError(
                    "has_speaker is required for speaker-conditioned duration prediction."
                )
            has_speaker = has_speaker.astype(mx.bool_)
            speaker_vec = self._speaker_vec(
                batch_size=text_state.shape[0],
                dtype=text_state.dtype,
                speaker_state=speaker_state,
                has_speaker=has_speaker,
            )
            h = self.token_input_proj(text_state)
            for block in self.token_blocks:
                h = block(h, cond=speaker_vec)
            token_logits = self.token_out_proj(self.token_out_norm(h)).squeeze(-1)
            # NOTE: MLX lacks mx.softplus; equivalent to log(1 + exp(x))
            token_frames = mx.log(1.0 + mx.exp(token_logits.astype(mx.float32)))
            total_frames = mx.sum(
                token_frames * text_mask.astype(token_frames.dtype), axis=1
            )
            return mx.log1p(mx.maximum(total_frames, 0.0))

        # Token-sum dual architecture (speaker + caption, v3 VoiceDesign)
        if self.duration_architecture == "token_sum_dual_adarn_zero_no_aux":
            if self.speaker_dim is None or self.caption_dim is None:
                raise RuntimeError(
                    "Dual token-sum architecture requires both speaker and caption modules."
                )
            if has_speaker is None:
                raise ValueError(
                    "has_speaker is required for dual duration prediction."
                )
            if has_caption is None:
                raise ValueError(
                    "has_caption is required for dual duration prediction."
                )
            has_speaker = has_speaker.astype(mx.bool_)
            has_caption = has_caption.astype(mx.bool_)
            speaker_vec = self._speaker_vec(
                batch_size=text_state.shape[0],
                dtype=text_state.dtype,
                speaker_state=speaker_state,
                has_speaker=has_speaker,
            )
            caption_vec = self._caption_vec(
                batch_size=text_state.shape[0],
                dtype=text_state.dtype,
                caption_state=caption_state,
                caption_mask=caption_mask,
                has_caption=has_caption,
            )
            h = self.token_input_proj(text_state)
            for block in self.token_blocks:
                h = block(h, cond=speaker_vec, caption_cond=caption_vec)
            token_logits = self.token_out_proj(self.token_out_norm(h)).squeeze(-1)
            token_frames = mx.log(1.0 + mx.exp(token_logits.astype(mx.float32)))
            total_frames = mx.sum(
                token_frames * text_mask.astype(token_frames.dtype), axis=1
            )
            return mx.log1p(mx.maximum(total_frames, 0.0))

        # Pooled architecture
        if self.text_pool is None:
            raise RuntimeError("Pooled duration modules are missing.")
        text_vec = self.text_pool(text_state, text_mask)

        if self.speaker_dim is None:
            x = mx.concatenate([text_vec, aux_features], axis=-1)
            h = self.input_proj(x)
            for block in self.blocks:
                h = block(h)
            return self.out_proj(self.out_norm(h)).squeeze(-1)

        if has_speaker is None:
            raise ValueError(
                "has_speaker is required for speaker-conditioned duration prediction."
            )
        has_speaker = has_speaker.astype(mx.bool_)
        speaker_vec = self._speaker_vec(
            batch_size=text_vec.shape[0],
            dtype=text_vec.dtype,
            speaker_state=speaker_state,
            has_speaker=has_speaker,
        )

        if self.speaker_fusion == "concat":
            x = mx.concatenate([text_vec, speaker_vec, aux_features], axis=-1)
            cond = None
        elif self.speaker_fusion == "adarn":
            if self.text_adarn_norm is None or self.text_adarn is None:
                raise RuntimeError("AdaRN duration speaker modules are missing.")
            scale, shift = mx.split(self.text_adarn(speaker_vec), 2, axis=-1)
            text_vec = self.text_adarn_norm(text_vec) * (1.0 + scale) + shift
            x = mx.concatenate([text_vec, aux_features], axis=-1)
            cond = None
        elif self.speaker_fusion == "adarn_zero":
            x = mx.concatenate([text_vec, aux_features], axis=-1)
            cond = speaker_vec
        elif self.speaker_fusion == "speaker_cross_attn":
            if self.speaker_cross_attn is None:
                raise RuntimeError("speaker_cross_attn duration module is missing.")
            speaker_context, speaker_context_mask = self._speaker_sequence(
                batch_size=text_vec.shape[0],
                dtype=text_vec.dtype,
                speaker_state=speaker_state,
                speaker_mask=speaker_mask,
                has_speaker=has_speaker,
            )
            context_vec = self.speaker_cross_attn(
                query=text_vec,
                context=speaker_context,
                context_mask=speaker_context_mask,
            )
            x = mx.concatenate([text_vec, context_vec, aux_features], axis=-1)
            cond = None
        elif self.speaker_fusion == "text_cross_attn":
            if self.text_cross_attn is None:
                raise RuntimeError("text_cross_attn duration module is missing.")
            context_vec = self.text_cross_attn(
                query=speaker_vec,
                context=text_state,
                context_mask=text_mask,
            )
            x = mx.concatenate([context_vec, speaker_vec, aux_features], axis=-1)
            cond = None
        else:
            raise RuntimeError(
                f"Unsupported duration speaker fusion: {self.speaker_fusion!r}"
            )

        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, cond=cond)
        return self.out_proj(self.out_norm(h)).squeeze(-1)

    def _speaker_sequence(
        self,
        batch_size: int,
        dtype,
        speaker_state: Optional[mx.array],
        speaker_mask: Optional[mx.array],
        has_speaker: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        if self.null_speaker is None or self.speaker_dim is None:
            raise RuntimeError("Duration speaker modules are missing.")
        null_token = mx.broadcast_to(
            self.null_speaker.astype(dtype)[None, None, :],
            (batch_size, 1, self.speaker_dim),
        )
        if speaker_state is None:
            return null_token, mx.ones((batch_size, 1), dtype=mx.bool_)
        if speaker_state.ndim != 3 or speaker_state.shape[0] != batch_size:
            raise ValueError(
                f"speaker_state must have shape (B, S, D), got {tuple(speaker_state.shape)}"
            )
        if speaker_state.shape[-1] != self.speaker_dim:
            raise ValueError(
                f"speaker_state last dim must be {self.speaker_dim}, got {speaker_state.shape[-1]}"
            )
        speaker_state = speaker_state.astype(dtype)
        if speaker_mask is None:
            speaker_mask = mx.ones((batch_size, speaker_state.shape[1]), dtype=mx.bool_)
        elif (
            speaker_mask.ndim != 2 or speaker_mask.shape[:2] != speaker_state.shape[:2]
        ):
            raise ValueError(
                "speaker_mask must have shape matching speaker_state (B, S), "
                f"got speaker_state={tuple(speaker_state.shape)} mask={tuple(speaker_mask.shape)}"
            )
        speaker_mask = speaker_mask.astype(mx.bool_)
        real_mask = speaker_mask & has_speaker[:, None]
        fallback_mask = ~mx.any(real_mask, axis=1, keepdims=True)
        context = mx.concatenate([speaker_state, null_token], axis=1)
        context_mask = mx.concatenate([real_mask, fallback_mask], axis=1)
        return context, context_mask


# ---------------------------------------------------------------------------
# Main DiT model
# ---------------------------------------------------------------------------


class IrodoriDiT(nn.Module):
    """
    Irodori-TTS DiT model (MLX port of TextToLatentRFDiT).

    Supports three conditioning modes:
      - Speaker mode (use_speaker_condition_resolved=True, use_caption_condition=False)
      - Caption mode (use_caption_condition=True, use_speaker_condition_resolved=False)
      - Dual mode (both True, v3 VoiceDesign): speaker + caption simultaneously

    Input x_t : (B, S, latent_dim * latent_patch_size)
    Output v_t : same shape — velocity prediction for Rectified Flow ODE.
    """

    def __init__(self, cfg: IrodoriDiTConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.model_dim // cfg.num_heads

        self.text_encoder = TextEncoder(
            vocab_size=cfg.text_vocab_size,
            dim=cfg.text_dim,
            heads=cfg.text_heads,
            num_layers=cfg.text_layers,
            mlp_ratio=cfg.text_mlp_ratio_resolved,
            norm_eps=cfg.norm_eps,
        )
        self.text_norm = RMSNorm(cfg.text_dim, eps=cfg.norm_eps)

        speaker_ctx_dim: Optional[int] = None
        caption_ctx_dim: Optional[int] = None

        if cfg.use_speaker_condition_resolved:
            self.speaker_encoder = ReferenceLatentEncoder(
                in_dim=cfg.speaker_patched_latent_dim,
                dim=cfg.speaker_dim,
                heads=cfg.speaker_heads,
                num_layers=cfg.speaker_layers,
                mlp_ratio=cfg.speaker_mlp_ratio_resolved,
                norm_eps=cfg.norm_eps,
            )
            self.speaker_norm = RMSNorm(cfg.speaker_dim, eps=cfg.norm_eps)
            speaker_ctx_dim = cfg.speaker_dim

        if cfg.use_caption_condition:
            self.caption_encoder = TextEncoder(
                vocab_size=cfg.caption_vocab_size_resolved,
                dim=cfg.caption_dim_resolved,
                heads=cfg.caption_heads_resolved,
                num_layers=cfg.caption_layers_resolved,
                mlp_ratio=cfg.caption_mlp_ratio_resolved,
                norm_eps=cfg.norm_eps,
            )
            self.caption_norm = RMSNorm(cfg.caption_dim_resolved, eps=cfg.norm_eps)
            caption_ctx_dim = cfg.caption_dim_resolved

        # Duration predictor (v3)
        self.duration_predictor = None
        if cfg.use_duration_predictor:
            duration_speaker_dim = (
                cfg.speaker_dim if cfg.use_speaker_condition_resolved else None
            )
            duration_caption_dim = (
                cfg.caption_dim_resolved if cfg.use_caption_condition else None
            )
            self.duration_predictor = DurationPredictor(
                text_dim=cfg.text_dim,
                aux_dim=cfg.duration_aux_dim,
                hidden_dim=cfg.duration_hidden_dim,
                layers=cfg.duration_layers,
                norm_eps=cfg.norm_eps,
                speaker_dim=duration_speaker_dim,
                speaker_fusion=cfg.duration_speaker_fusion,
                caption_dim=duration_caption_dim,
                caption_fusion=cfg.duration_caption_fusion,
                caption_pooling=cfg.duration_caption_pooling,
                attention_heads=cfg.duration_attention_heads,
                architecture=cfg.duration_architecture,
                token_init_frames=cfg.duration_token_init_frames,
            )

        # Timestep → conditioning embedding (3 × model_dim for shift/scale/gate)
        self.cond_module = nn.Sequential(
            nn.Linear(cfg.timestep_embed_dim, cfg.model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.model_dim, cfg.model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.model_dim, cfg.model_dim * 3, bias=False),
        )

        self.in_proj = nn.Linear(cfg.patched_latent_dim, cfg.model_dim, bias=True)
        mlp_hidden = int(cfg.model_dim * cfg.mlp_ratio)
        self.blocks = [
            DiffusionBlock(
                dim=cfg.model_dim,
                heads=cfg.num_heads,
                mlp_hidden_dim=mlp_hidden,
                text_ctx_dim=cfg.text_dim,
                speaker_ctx_dim=speaker_ctx_dim,
                adaln_rank=cfg.adaln_rank,
                norm_eps=cfg.norm_eps,
                caption_ctx_dim=caption_ctx_dim,
            )
            for _ in range(cfg.num_layers)
        ]
        self.out_norm = RMSNorm(cfg.model_dim, eps=cfg.norm_eps)
        self.out_proj = nn.Linear(cfg.model_dim, cfg.patched_latent_dim, bias=True)

    # ------------------------------------------------------------------
    # Condition encoding (text + speaker/caption) — cached across steps
    # ------------------------------------------------------------------

    def encode_conditions(
        self,
        text_input_ids: mx.array,
        text_mask: mx.array,
        ref_latent: Optional[mx.array] = None,
        ref_mask: Optional[mx.array] = None,
        caption_input_ids: Optional[mx.array] = None,
        caption_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Encode text and context (speaker latent or caption) into conditioning states.
        Returns (text_state, text_mask, context_state, context_mask).
        """
        text_state = self.text_norm(self.text_encoder(text_input_ids, text_mask))

        if self.cfg.use_speaker_condition_resolved:
            assert ref_latent is not None and ref_mask is not None
            ref_latent_p, ref_mask_p = patch_sequence_with_mask(
                ref_latent, ref_mask, self.cfg.speaker_patch_size
            )
            context_state = self.speaker_norm(
                self.speaker_encoder(ref_latent_p, ref_mask_p)
            )
            context_mask = ref_mask_p
        else:
            assert caption_input_ids is not None and caption_mask is not None
            context_state = self.caption_norm(
                self.caption_encoder(caption_input_ids, caption_mask)
            )
            context_mask = caption_mask

        return text_state, text_mask, context_state, context_mask

    def encode_conditions_full(
        self,
        text_input_ids: mx.array,
        text_mask: mx.array,
        ref_latent: Optional[mx.array] = None,
        ref_mask: Optional[mx.array] = None,
        caption_input_ids: Optional[mx.array] = None,
        caption_mask: Optional[mx.array] = None,
    ) -> Tuple[
        mx.array,
        mx.array,
        Optional[mx.array],
        Optional[mx.array],
        Optional[mx.array],
        Optional[mx.array],
    ]:
        """
        Encode all conditions including both speaker and caption states.
        Returns (text_state, text_mask, speaker_state, speaker_mask, caption_state, caption_mask).
        """
        text_state = self.text_norm(self.text_encoder(text_input_ids, text_mask))

        speaker_state = None
        speaker_mask = None
        if self.cfg.use_speaker_condition_resolved:
            if ref_latent is not None and ref_mask is not None:
                ref_latent_p, ref_mask_p = patch_sequence_with_mask(
                    ref_latent, ref_mask, self.cfg.speaker_patch_size
                )
                speaker_state = self.speaker_norm(
                    self.speaker_encoder(ref_latent_p, ref_mask_p)
                )
                speaker_mask = ref_mask_p
            else:
                # Zero speaker state for duration prediction with no reference
                speaker_state = mx.zeros(
                    (text_input_ids.shape[0], 1, self.cfg.speaker_dim),
                    dtype=text_state.dtype,
                )
                speaker_mask = mx.zeros(
                    (text_input_ids.shape[0], 1),
                    dtype=mx.bool_,
                )

        out_caption_state = None
        out_caption_mask = None
        if self.cfg.use_caption_condition:
            if caption_input_ids is not None and caption_mask is not None:
                out_caption_state = self.caption_norm(
                    self.caption_encoder(caption_input_ids, caption_mask)
                )
                out_caption_mask = caption_mask

        return (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            out_caption_state,
            out_caption_mask,
        )

    def build_kv_cache(
        self,
        text_state: mx.array,
        speaker_state: Optional[mx.array] = None,
        caption_state: Optional[mx.array] = None,
    ) -> Tuple[List[KVCache], Optional[List[KVCache]], Optional[List[KVCache]]]:
        """Pre-compute per-layer text/speaker/caption KV projections for fast sampling."""
        kv_text = [
            block.attention.get_kv_cache_text(text_state) for block in self.blocks
        ]
        kv_speaker = None
        if speaker_state is not None and self.cfg.use_speaker_condition_resolved:
            kv_speaker = [
                block.attention.get_kv_cache_speaker(speaker_state)
                for block in self.blocks
            ]
        kv_caption = None
        if caption_state is not None and self.cfg.use_caption_condition:
            kv_caption = [
                block.attention.get_kv_cache_caption(caption_state)
                for block in self.blocks
            ]
        return kv_text, kv_speaker, kv_caption

    @staticmethod
    def masked_mean(state: mx.array, mask: mx.array) -> mx.array:
        """Compute masked mean over sequence dimension."""
        mask_f = mask[..., None].astype(state.dtype)
        denom = mx.maximum(mx.sum(mask_f, axis=1), 1.0)
        return mx.sum(state * mask_f, axis=1) / denom

    def predict_duration_log_frames(
        self,
        text_state: mx.array,
        text_mask: mx.array,
        speaker_state: Optional[mx.array],
        speaker_mask: Optional[mx.array],
        duration_features: mx.array,
        has_speaker: Optional[mx.array],
        caption_state: Optional[mx.array] = None,
        caption_mask: Optional[mx.array] = None,
        has_caption: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Predict log1p(num_frames) from text state and duration features.
        Returns (B,) array of log frame predictions.
        """
        if self.duration_predictor is None:
            raise RuntimeError("Duration predictor is disabled for this model.")
        if duration_features.ndim != 2:
            raise ValueError(
                f"duration_features must have shape (B, D), got {tuple(duration_features.shape)}"
            )
        if duration_features.shape[1] != self.cfg.duration_aux_dim:
            raise ValueError(
                f"duration_features dim mismatch: "
                f"expected {self.cfg.duration_aux_dim}, got {duration_features.shape[1]}"
            )
        pred = self.duration_predictor(
            text_state,
            text_mask=text_mask,
            aux_features=duration_features,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            has_speaker=has_speaker,
            caption_state=caption_state,
            caption_mask=caption_mask,
            has_caption=has_caption,
        )
        return pred.astype(mx.float32)

    # ------------------------------------------------------------------
    # Forward (with pre-encoded conditions)
    # ------------------------------------------------------------------

    def forward_with_conditions(
        self,
        x_t: mx.array,
        t: mx.array,
        text_state: mx.array,
        text_mask: mx.array,
        speaker_state: mx.array,
        speaker_mask: mx.array,
        kv_text: Optional[List[KVCache]] = None,
        kv_speaker: Optional[List[KVCache]] = None,
        start_pos: int = 0,
        caption_state: Optional[mx.array] = None,
        caption_mask: Optional[mx.array] = None,
        kv_caption: Optional[List[KVCache]] = None,
    ) -> mx.array:
        """
        Forward pass with pre-encoded conditions.

        For speaker-only models: speaker_state/mask carry the speaker context.
        For caption-only models: speaker_state/mask carry the caption context
            (backward-compat; internally routed to caption branch).
        For dual models (v3 VoiceDesign): speaker_state=speaker, caption_state=caption.
        """
        t_embed = get_timestep_embedding(t, self.cfg.timestep_embed_dim).astype(
            x_t.dtype
        )
        cond_embed = self.cond_module(t_embed)[:, None, :]  # (B, 1, 3*model_dim)

        x = self.in_proj(x_t)
        freqs_cis = precompute_freqs_cis(self.head_dim, start_pos + x.shape[1])

        use_spk = self.cfg.use_speaker_condition_resolved
        use_cap = self.cfg.use_caption_condition

        # For caption-only models: speaker_state arg contains caption context (backward compat)
        if not use_spk and use_cap:
            actual_caption_state = (
                caption_state if caption_state is not None else speaker_state
            )
            actual_caption_mask = (
                caption_mask if caption_mask is not None else speaker_mask
            )
            actual_kv_caption = kv_caption if kv_caption is not None else kv_speaker
            actual_speaker_state = None
            actual_speaker_mask = None
            actual_kv_speaker = None
        else:
            actual_speaker_state = speaker_state
            actual_speaker_mask = speaker_mask
            actual_kv_speaker = kv_speaker
            actual_caption_state = caption_state
            actual_caption_mask = caption_mask
            actual_kv_caption = kv_caption

        for i, block in enumerate(self.blocks):
            kv_t = (
                kv_text[i]
                if kv_text is not None
                else block.attention.get_kv_cache_text(text_state)
            )
            kv_s = None
            if use_spk and actual_speaker_state is not None:
                kv_s = (
                    actual_kv_speaker[i]
                    if actual_kv_speaker is not None
                    else block.attention.get_kv_cache_speaker(actual_speaker_state)
                )
            kv_c = None
            if use_cap and actual_caption_state is not None:
                kv_c = (
                    actual_kv_caption[i]
                    if actual_kv_caption is not None
                    else block.attention.get_kv_cache_caption(actual_caption_state)
                )
            x = block(
                x,
                cond_embed,
                text_mask,
                freqs_cis,
                kv_t,
                kv_s,
                actual_speaker_mask,
                kv_c,
                actual_caption_mask,
                start_pos,
            )

        x = self.out_norm(x)
        return self.out_proj(x).astype(mx.float32)

    # ------------------------------------------------------------------
    # Full forward (encode conditions + denoise)
    # ------------------------------------------------------------------

    def __call__(
        self,
        x_t: mx.array,
        t: mx.array,
        text_input_ids: mx.array,
        text_mask: mx.array,
        ref_latent: mx.array,
        ref_mask: mx.array,
    ) -> mx.array:
        text_state, text_mask, speaker_state, speaker_mask = self.encode_conditions(
            text_input_ids, text_mask, ref_latent, ref_mask
        )
        return self.forward_with_conditions(
            x_t, t, text_state, text_mask, speaker_state, speaker_mask
        )
