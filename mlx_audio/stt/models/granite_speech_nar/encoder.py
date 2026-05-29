"""MLX implementation of the Granite Speech NAR Conformer encoder.

Mirrors modeling_granite_speech_nar.py in the upstream HF repo, but in MLX layout:
  - All tensors are [B, T, C] throughout (PyTorch encoder uses [B, C, T] in the conv module).
  - Conv1d weights are stored [C_out, K, C_in] (PyTorch is [C_out, C_in, K] — converter handles).
  - BatchNorm runs in eval-only mode using stored running stats (manual implementation).
  - Attention is block-local on windows of context_size=200 frames, with Shaw relative
    positional embeddings via a learned table per layer.

All sub-modules accept a `dropout` arg that is a no-op at inference (dropout=0.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


class ConformerFFN(nn.Module):
    """Macaron FFN half: LN → up_proj → SiLU → down_proj."""

    def __init__(self, hidden_dim: int, mult: int, dropout: float = 0.0):
        super().__init__()
        inner = hidden_dim * mult
        self.pre_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.up_proj = nn.Linear(hidden_dim, inner, bias=True)
        self.down_proj = nn.Linear(inner, hidden_dim, bias=True)
        # dropout = 0.0 at inference; we keep the param for parity with the config
        # but never apply it.

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        h = self.pre_norm(x)
        h = self.up_proj(h)
        h = nn.silu(h)
        h = self.down_proj(h)
        return h.astype(dtype)


class ConformerAttention(nn.Module):
    """Block-local multi-head self-attention with Shaw relative positional embeddings.

    The input sequence [B, T, hidden_dim] is reshaped into blocks of context_size frames:
    [B, T // ctx, ctx, hidden_dim] (final partial block zero-padded). Attention is computed
    independently within each block. Shaw rel-pos is added to the pre-softmax logits as a
    per-(query, key)-position learned bias.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_head: int,
        max_pos_emb: int,
        context_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.context_size = context_size
        self.max_pos_emb = max_pos_emb
        self.inner = num_heads * dim_head
        self.pre_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.to_q = nn.Linear(hidden_dim, self.inner, bias=False)
        self.to_kv = nn.Linear(hidden_dim, 2 * self.inner, bias=False)
        self.to_out = nn.Linear(self.inner, hidden_dim, bias=True)
        # Shaw rel-pos table: rows are relative-distance indices in [-max_pos_emb, +max_pos_emb].
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        # Precompute the [ctx, ctx] distance index matrix once.
        idx = mx.arange(context_size)
        dist = idx[:, None] - idx[None, :]
        dist = mx.clip(dist, -context_size, context_size) + max_pos_emb
        self._dist = dist  # shape [ctx, ctx], int

    def __call__(self, x: mx.array) -> mx.array:
        B, T, H = x.shape
        ctx = self.context_size
        dtype = x.dtype
        h = self.pre_norm(x)

        # Pad to multiple of ctx
        pad = (ctx - (T % ctx)) % ctx
        if pad:
            h = mx.concatenate([h, mx.zeros((B, pad, H), dtype=h.dtype)], axis=1)
        T_pad = T + pad
        n_blocks = T_pad // ctx

        q = self.to_q(h)  # [B, T_pad, inner]
        kv = self.to_kv(h)  # [B, T_pad, 2*inner]
        k, v = mx.split(kv, 2, axis=-1)

        # Reshape into blocks and heads: [B, n_blocks, ctx, num_heads, dim_head]
        def _reshape(z):
            return z.reshape(B, n_blocks, ctx, self.num_heads, self.dim_head)

        q, k, v = _reshape(q), _reshape(k), _reshape(v)

        # Move heads to axis 2: [B, n_blocks, num_heads, ctx, dim_head]
        q = mx.transpose(q, (0, 1, 3, 2, 4))
        k = mx.transpose(k, (0, 1, 3, 2, 4))
        v = mx.transpose(v, (0, 1, 3, 2, 4))

        scale = self.dim_head**-0.5

        # Standard QK^T attention logits: [B, n_blocks, num_heads, ctx, ctx]
        attn_logits = (q @ mx.transpose(k, (0, 1, 2, 4, 3))) * scale

        # Shaw rel-pos: lookup → [ctx, ctx, dim_head], then einsum with q.
        rel = self.rel_pos_emb(self._dist.astype(mx.int32))  # [ctx, ctx, dim_head]
        # pos_attn[b, m, h, i, j] = sum_d q[b, m, h, i, d] * rel[i, j, d]  * scale
        pos_attn = mx.einsum("bmhcd,crd->bmhcr", q, rel) * scale

        attn_logits = attn_logits + pos_attn
        attn = mx.softmax(attn_logits.astype(mx.float32), axis=-1).astype(q.dtype)

        out = attn @ v  # [B, n_blocks, num_heads, ctx, dim_head]
        out = mx.transpose(
            out, (0, 1, 3, 2, 4)
        )  # [B, n_blocks, ctx, num_heads, dim_head]
        out = out.reshape(B, T_pad, self.inner)
        out = out[:, :T, :]  # trim padding
        return self.to_out(out).astype(dtype)


class EvalBatchNorm(nn.Module):
    """Inference-only BatchNorm over the last dim. Uses stored running stats.

    Formula: (x - running_mean) / sqrt(running_var + eps) * weight + bias.
    Loaded from a PyTorch BatchNorm1d checkpoint where running_mean / running_var
    are stored as 1-D tensors indexed by channel.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)
        self.running_mean = mx.zeros(num_features)
        self.running_var = mx.ones(num_features)

    def __call__(self, x: mx.array) -> mx.array:
        # Broadcasting: x [B, T, C], stats [C]
        inv = mx.rsqrt(self.running_var + self.eps)
        return (x - self.running_mean) * inv * self.weight + self.bias


class ConformerConvModule(nn.Module):
    """Conformer conv module.

    Layout (all MLX-style [B, T, C]):
      LN → pointwise up_conv (C → C*expansion*2) → GLU along channel dim (→ C*expansion)
        → depthwise Conv1d (C*expansion → C*expansion, k=15, groups=channels)
        → SiLU → EvalBatchNorm → pointwise down_conv (C*expansion → C)
    """

    def __init__(
        self, hidden_dim: int, expansion: int, kernel_size: int, dropout: float = 0.0
    ):
        super().__init__()
        inner = hidden_dim * expansion
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        # MLX Conv1d uses [B, T, C] inputs. Weights stored as [C_out, K, C_in/groups].
        self.up_conv = nn.Conv1d(hidden_dim, 2 * inner, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv1d(
            inner,
            inner,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner,
            bias=False,
        )
        self.bn = EvalBatchNorm(inner)
        self.down_conv = nn.Conv1d(inner, hidden_dim, kernel_size=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        h = self.norm(x)
        h = self.up_conv(h)  # [B, T, 2*inner]
        # GLU along channel dim: split into (a, gate), output = a * sigmoid(gate)
        a, gate = mx.split(h, 2, axis=-1)
        h = a * mx.sigmoid(gate)  # [B, T, inner]
        h = self.depth_conv(h)  # [B, T, inner]
        h = self.bn(h)
        h = nn.silu(h)  # upstream order: silu(batch_norm(x))
        h = self.down_conv(h)  # [B, T, hidden_dim]
        return h.astype(dtype)


class ConformerBlock(nn.Module):
    """Full Conformer block: ff1 + attn + conv + ff2, each with residual; then post_norm."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_head: int,
        feedforward_mult: int,
        conv_expansion: int,
        conv_kernel_size: int,
        max_pos_emb: int,
        context_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ff1 = ConformerFFN(hidden_dim, feedforward_mult, dropout)
        self.attn = ConformerAttention(
            hidden_dim, num_heads, dim_head, max_pos_emb, context_size, dropout
        )
        self.conv = ConformerConvModule(
            hidden_dim, conv_expansion, conv_kernel_size, dropout
        )
        self.ff2 = ConformerFFN(hidden_dim, feedforward_mult, dropout)
        self.post_norm = nn.LayerNorm(hidden_dim, eps=1e-5)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = 0.5 * self.ff1(x) + x
        x = self.attn(x) + x
        x = self.conv(x) + x
        x = 0.5 * self.ff2(x) + x
        x = self.post_norm(x)
        return x.astype(dtype)


@dataclass
class EncoderOutput:
    char_logits: mx.array
    bpe_logits: mx.array
    hidden_states_for_projector: list[mx.array]


class ConformerEncoder(nn.Module):
    """Full encoder: input_linear → 16 Conformer blocks → self-cond → BPE pool → CTC heads."""

    def __init__(self, cfg, encoder_layer_indices: list[int]):
        super().__init__()
        self.cfg = cfg
        self.encoder_layer_indices = encoder_layer_indices
        self.input_linear = nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=True)
        self.layers = [
            ConformerBlock(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                dim_head=cfg.dim_head,
                feedforward_mult=cfg.feedforward_mult,
                conv_expansion=cfg.conv_expansion_factor,
                conv_kernel_size=cfg.conv_kernel_size,
                max_pos_emb=cfg.max_pos_emb,
                context_size=cfg.context_size,
                dropout=0.0,
            )
            for _ in range(cfg.num_layers)
        ]
        self.out = nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=True)
        self.out_mid = nn.Linear(cfg.output_dim, cfg.hidden_dim, bias=True)
        self.out_bpe = nn.Linear(cfg.hidden_dim, cfg.bpe_output_dim, bias=True)
        self.self_conditioning_layer = cfg.self_conditioning_layer
        self.bpe_pooling_window = cfg.bpe_pooling_window

    def __call__(self, features: mx.array) -> EncoderOutput:
        # features: [B, T, input_dim]
        h = self.input_linear(features)  # [B, T, hidden]
        all_hidden_states: list[mx.array] = [h]
        char_logits = None
        blank_probs = None
        for i, layer in enumerate(self.layers, start=1):
            h = layer(h)
            if i == self.self_conditioning_layer:
                char_logits = self.out(h)  # [B, T, output_dim]
                probs = mx.softmax(char_logits.astype(mx.float32), axis=-1)
                blank_probs = probs[..., 0]  # [B, T]
                h = h + self.out_mid(probs.astype(h.dtype))
            all_hidden_states.append(h)

        # Hidden states for projector at indices [4, 8, 12, -1]
        hs_for_proj = [all_hidden_states[idx] for idx in self.encoder_layer_indices]

        # Posterior-weighted BPE pooling (windows of bpe_pooling_window=4 frames).
        pooled = _posterior_weighted_pool(h, blank_probs, self.bpe_pooling_window)
        bpe_logits = self.out_bpe(pooled).astype(features.dtype)
        if char_logits is None:
            raise RuntimeError(
                "self_conditioning_layer was not reached — config mismatch"
            )
        return EncoderOutput(
            char_logits=char_logits,
            bpe_logits=bpe_logits,
            hidden_states_for_projector=hs_for_proj,
        )


def _posterior_weighted_pool(
    h: mx.array, blank_probs: mx.array, window: int
) -> mx.array:
    """Pool h [B, T, C] in windows of `window` frames, weighting each frame by (1 - blank_probs).

    If T is not a multiple of `window`, the trailing partial window is included with available
    frames (matching upstream behavior).
    """
    B, T, C = h.shape
    n_full = T // window
    tail = T - n_full * window
    if tail == 0:
        importance = (1.0 - blank_probs).reshape(B, n_full, window)  # [B, n_full, w]
        h_w = h.reshape(B, n_full, window, C)
        weights = importance / mx.maximum(importance.sum(axis=-1, keepdims=True), 1e-6)
        pooled = (h_w * weights[..., None]).sum(axis=2)
        return pooled
    # Pad to next multiple, then mask
    pad = window - tail
    h_p = mx.concatenate([h, mx.zeros((B, pad, C), dtype=h.dtype)], axis=1)
    imp = mx.concatenate(
        [
            1.0 - blank_probs,
            mx.zeros((B, pad), dtype=blank_probs.dtype),
        ],
        axis=1,
    )
    n = (T + pad) // window
    h_w = h_p.reshape(B, n, window, C)
    imp_w = imp.reshape(B, n, window)
    weights = imp_w / mx.maximum(imp_w.sum(axis=-1, keepdims=True), 1e-6)
    pooled = (h_w * weights[..., None]).sum(axis=2)
    return pooled
