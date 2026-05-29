"""MLX implementation of the Granite Speech NAR projector.

The projector consumes the 4 encoder hidden states selected by
`encoder_layer_indices=[4, 8, 12, -1]` (concatenated by the caller to
[B, T, 4096]) and produces [B, T_out=nblocks*3, 2048] acoustic embeddings ready
to feed the bidirectional Granite editor.

Pipeline (top-level):
  per-layer LayerNorm × 4 → fuse linear + GELU → block-windowed reshape
    → mean-pool query init → 2× Q-Former layer (cross-attn + MLP) → out_norm + out_linear

No self-attention over queries (despite `attn_norm` naming — that LN is the cross-
attention pre-norm). No relative position bias in cross-attention.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class QFormerMLP(nn.Module):
    """SiLU MLP: fc1 → SiLU → fc2. Both linears have bias."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        h = self.fc1(x)
        h = nn.silu(h)
        h = self.fc2(h)
        return h.astype(dtype)


class QFormerCrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to encoder K/V (no causal mask).

    Standard scaled dot product attention. 32 heads × 64 dim_head = 2048.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, query: mx.array, kv: mx.array) -> mx.array:
        # query: [B, Q, hidden], kv: [B, K, hidden]
        B, Q, H = query.shape
        K = kv.shape[1]
        dtype = query.dtype

        q = (
            self.q_proj(query)
            .reshape(B, Q, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(kv)
            .reshape(B, K, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(kv)
            .reshape(B, K, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        # Shapes now: q [B, H, Q, D], k [B, H, K, D], v [B, H, K, D]

        scale = self.head_dim**-0.5
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        # attn: [B, H, Q, D]

        attn = attn.transpose(0, 2, 1, 3).reshape(B, Q, H)
        return self.o_proj(attn).astype(dtype)


class QFormerLayer(nn.Module):
    """Single Q-Former layer: pre-norm cross-attention + pre-norm MLP, each with residual.

    Note there is NO self-attention over the query tokens — `attn_norm` is the cross-attention
    pre-norm, not a self-attention norm.
    """

    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int, eps: float
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.cross_attention = QFormerCrossAttention(hidden_size, num_heads)
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = QFormerMLP(hidden_size, intermediate_size)

    def __call__(self, query: mx.array, kv: mx.array) -> mx.array:
        dtype = query.dtype
        # Cross-attention with pre-norm and residual on the QUERY stream only.
        # K, V come from `kv` (with window_positions already added by the caller).
        attn_out = self.cross_attention(self.attn_norm(query), kv)
        query = query + attn_out
        # MLP with pre-norm and residual.
        mlp_out = self.mlp(self.mlp_norm(query))
        query = query + mlp_out
        return query.astype(dtype)


class QFormer(nn.Module):
    """Stack of QFormerLayers. Threads `query` through each layer; `kv` is constant."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        eps: float,
    ):
        super().__init__()
        self.layers = [
            QFormerLayer(hidden_size, intermediate_size, num_heads, eps)
            for _ in range(num_layers)
        ]

    def __call__(self, query: mx.array, kv: mx.array) -> mx.array:
        for layer in self.layers:
            query = layer(query, kv)
        return query


class GraniteSpeechNarProjector(nn.Module):
    """Top-level projector.

    Takes the concatenation of 4 encoder hidden states [B, T, 4*encoder_dim] and
    produces [B, T_out, llm_dim] where T_out = ceil(T / block_size) * (block_size / downsample_rate).

    The caller is responsible for concatenating the encoder hidden states in the order
    given by `encoder_layer_indices=[4,8,12,-1]` and for applying the output scale
    (`/embedding_multiplier`) before feeding the result to the LLM.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Per-encoder-layer LayerNorms (4 of them, dim=encoder_dim=1024)
        self.layer_norms = [
            nn.LayerNorm(cfg.encoder_dim, eps=cfg.layernorm_eps)
            for _ in range(cfg.num_encoder_layers)
        ]
        # Fuse 4 normalized encoder layers → hidden_size
        self.layer_projector = nn.Linear(
            cfg.num_encoder_layers * cfg.encoder_dim,
            cfg.hidden_size,
            bias=cfg.mlp_bias,
        )
        # Learned tokens with leading batch-broadcast dim
        query_length = cfg.block_size // cfg.downsample_rate  # 15 // 5 = 3
        self.query = mx.zeros((1, query_length, cfg.hidden_size))
        self.window_positions = mx.zeros((1, cfg.block_size, cfg.hidden_size))
        # Q-Former stack
        self.qformer = QFormer(
            num_layers=cfg.num_layers,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.hidden_size * cfg.mlp_ratio,
            num_heads=cfg.num_heads,
            eps=cfg.layernorm_eps,
        )
        # Output head
        self.out_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_eps)
        self.out_linear = nn.Linear(cfg.hidden_size, cfg.llm_dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """hidden_states: [B, T, num_encoder_layers * encoder_dim] (= [B, T, 4096])"""
        cfg = self.cfg
        B, T, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Per-encoder-layer LayerNorm + re-concat + project to hidden_size + GELU
        sliced = mx.split(hidden_states, cfg.num_encoder_layers, axis=-1)
        normed = [self.layer_norms[i](sliced[i]) for i in range(cfg.num_encoder_layers)]
        h = mx.concatenate(normed, axis=-1)  # [B, T, 4 * encoder_dim]
        h = self.layer_projector(h)  # [B, T, hidden_size]
        h = nn.gelu(h)

        # 2. Right-pad to next multiple of block_size and reshape into windows
        block = cfg.block_size  # 15
        pad = (block - (T % block)) % block
        if pad:
            h = mx.concatenate(
                [h, mx.zeros((B, pad, cfg.hidden_size), dtype=h.dtype)], axis=1
            )
        T_pad = T + pad
        nblocks = T_pad // block
        h = h.reshape(B * nblocks, block, cfg.hidden_size)  # [B*nblocks, 15, hidden]

        # 3. Mean-pool over downsample_rate=5 groups → query init
        query_length = block // cfg.downsample_rate  # 3
        mean_pool = h.reshape(
            B * nblocks,
            query_length,
            cfg.downsample_rate,
            cfg.hidden_size,
        ).mean(
            axis=-2
        )  # [B*nblocks, 3, hidden]

        # 4. Broadcast learned query and window_positions
        # self.query: [1, 3, hidden] → broadcast to [B*nblocks, 3, hidden]
        query = self.query.astype(h.dtype) + mean_pool
        kv = h + self.window_positions.astype(h.dtype)

        # 5. Run through Q-Former
        out = self.qformer(query, kv)  # [B*nblocks, 3, hidden]

        # 6. Output head
        out = self.out_norm(out)
        out = self.out_linear(out)  # [B*nblocks, 3, llm_dim]

        # 7. Reshape back: [B, nblocks*query_length, llm_dim]
        out = out.reshape(B, nblocks * query_length, cfg.llm_dim)
        return out.astype(dtype)
