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

    Exactly one of speaker_ctx_dim or caption_ctx_dim must be provided.
    The corresponding weight keys (wk_speaker/wv_speaker or wk_caption/wv_caption)
    are created accordingly so that loaded PyTorch weights map directly.
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

        if speaker_ctx_dim is not None:
            self.wk_speaker = nn.Linear(speaker_ctx_dim, dim, bias=False)
            self.wv_speaker = nn.Linear(speaker_ctx_dim, dim, bias=False)
            self._context_mode = "speaker"
        elif caption_ctx_dim is not None:
            self.wk_caption = nn.Linear(caption_ctx_dim, dim, bias=False)
            self.wv_caption = nn.Linear(caption_ctx_dim, dim, bias=False)
            self._context_mode = "caption"
        else:
            raise ValueError("Either speaker_ctx_dim or caption_ctx_dim must be set")

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

    def get_kv_cache_context(self, context_state: mx.array) -> KVCache:
        """Dispatch to speaker or caption KV cache based on model mode."""
        if self._context_mode == "caption":
            return self.get_kv_cache_caption(context_state)
        return self.get_kv_cache_speaker(context_state)

    def __call__(
        self,
        x: mx.array,
        text_mask: mx.array,
        context_mask: mx.array,
        freqs_cis: RotaryCache,
        kv_cache_text: KVCache,
        kv_cache_context: KVCache,
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
        k_ctx, v_ctx = kv_cache_context

        k = mx.concatenate([k_self, k_text, k_ctx], axis=1)
        v = mx.concatenate([v_self, v_text, v_ctx], axis=1)

        self_mask = mx.ones((bsz, seq_len), dtype=mx.bool_)
        full_mask = mx.concatenate([self_mask, text_mask, context_mask], axis=1)
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
        context_mask: mx.array,
        freqs_cis: RotaryCache,
        kv_cache_text: KVCache,
        kv_cache_context: KVCache,
        start_pos: int = 0,
    ) -> mx.array:
        x_norm, attn_gate = self.attention_adaln(x, cond_embed)
        x = x + attn_gate * self.attention(
            x_norm,
            text_mask,
            context_mask,
            freqs_cis,
            kv_cache_text,
            kv_cache_context,
            start_pos,
        )
        x_norm, mlp_gate = self.mlp_adaln(x, cond_embed)
        x = x + mlp_gate * self.mlp(x_norm)
        return x


# ---------------------------------------------------------------------------
# Main DiT model
# ---------------------------------------------------------------------------


class IrodoriDiT(nn.Module):
    """
    Irodori-TTS DiT model (MLX port of TextToLatentRFDiT).

    Supports two conditioning modes (mutually exclusive):
      - Speaker mode (use_speaker_condition=True): ref audio latent → speaker embedding
      - Caption mode (use_caption_condition=True): style text → caption embedding

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

        if cfg.use_speaker_condition:
            self.speaker_encoder = ReferenceLatentEncoder(
                in_dim=cfg.speaker_patched_latent_dim,
                dim=cfg.speaker_dim,
                heads=cfg.speaker_heads,
                num_layers=cfg.speaker_layers,
                mlp_ratio=cfg.speaker_mlp_ratio_resolved,
                norm_eps=cfg.norm_eps,
            )
            self.speaker_norm = RMSNorm(cfg.speaker_dim, eps=cfg.norm_eps)
            context_dim = cfg.speaker_dim
            speaker_ctx_dim: Optional[int] = cfg.speaker_dim
            caption_ctx_dim: Optional[int] = None
        else:
            self.caption_encoder = TextEncoder(
                vocab_size=cfg.caption_vocab_size_resolved,
                dim=cfg.caption_dim_resolved,
                heads=cfg.caption_heads_resolved,
                num_layers=cfg.caption_layers_resolved,
                mlp_ratio=cfg.caption_mlp_ratio_resolved,
                norm_eps=cfg.norm_eps,
            )
            self.caption_norm = RMSNorm(cfg.caption_dim_resolved, eps=cfg.norm_eps)
            context_dim = cfg.caption_dim_resolved
            speaker_ctx_dim = None
            caption_ctx_dim = cfg.caption_dim_resolved

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

        self._context_dim = context_dim

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

        if self.cfg.use_speaker_condition:
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

    def build_kv_cache(
        self,
        text_state: mx.array,
        context_state: mx.array,
    ) -> Tuple[List[KVCache], List[KVCache]]:
        """Pre-compute per-layer text/context KV projections for fast sampling."""
        kv_text = [
            block.attention.get_kv_cache_text(text_state) for block in self.blocks
        ]
        kv_context = [
            block.attention.get_kv_cache_context(context_state) for block in self.blocks
        ]
        return kv_text, kv_context

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
    ) -> mx.array:
        # NOTE: speaker_state/speaker_mask/kv_speaker are the generic "context"
        # (either speaker or caption depending on cfg.use_caption_condition).
        t_embed = get_timestep_embedding(t, self.cfg.timestep_embed_dim).astype(
            x_t.dtype
        )
        cond_embed = self.cond_module(t_embed)[:, None, :]  # (B, 1, 3*model_dim)

        x = self.in_proj(x_t)
        freqs_cis = precompute_freqs_cis(self.head_dim, start_pos + x.shape[1])

        for i, block in enumerate(self.blocks):
            kv_t = (
                kv_text[i]
                if kv_text is not None
                else block.attention.get_kv_cache_text(text_state)
            )
            kv_s = (
                kv_speaker[i]
                if kv_speaker is not None
                else block.attention.get_kv_cache_context(speaker_state)
            )
            x = block(
                x,
                cond_embed,
                text_mask,
                speaker_mask,
                freqs_cis,
                kv_t,
                kv_s,
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
