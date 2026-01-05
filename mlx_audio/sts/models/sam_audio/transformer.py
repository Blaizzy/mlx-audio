# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import math
from functools import partial
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import TransformerConfig
from .patcher import Patcher
from .rope import RotaryEmbedding


def get_nonlinearity(kind: str) -> Callable:
    """Get activation function by name."""
    nonlinearities = {
        "relu": nn.relu,
        "gelu": nn.gelu,
        "swiglu": None,  # Handled specially
        "approx_gelu": partial(nn.gelu, approx="fast"),
        "silu": nn.silu,
    }
    return nonlinearities.get(kind)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def _norm(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        output = self._norm(x.astype(mx.float32))
        return (output * self.weight).astype(x.dtype)


class ProjectionLayer(nn.Module):
    """Projection layer with optional SwiGLU activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        non_linearity: str,
        dropout: float = 0.0,
        fc_bias: bool = False,
    ):
        super().__init__()
        self.swiglu = non_linearity == "swiglu"
        self.dropout_rate = dropout

        self.w1 = nn.Linear(in_dim, out_dim, bias=fc_bias)
        self.w2 = nn.Linear(out_dim, out_dim, bias=fc_bias)

        if self.swiglu:
            self.w3 = nn.Linear(in_dim, out_dim, bias=fc_bias)

        self.non_linearity = get_nonlinearity(non_linearity)

    def __call__(self, x: mx.array) -> mx.array:
        hidden1 = self.w1(x)
        if self.swiglu:
            hidden3 = self.w3(x)
            hidden = nn.silu(hidden1) * hidden3
        else:
            hidden = self.non_linearity(hidden1)
        return self.w2(hidden)


class Attention(nn.Module):
    """Multi-head attention with optional QK normalization and RoPE support.

    NOTE: This uses the same non-standard head reshape order as PyTorch SAM-Audio:
    - Reshape: (B, T, C) -> (B, T, head_dim, n_heads) [NOT (B, T, n_heads, head_dim)]
    - Permute: (B, T, head_dim, n_heads) -> (B, n_heads, T, head_dim)

    This is critical for weight compatibility with the pretrained SAM-Audio model.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float = 1e-5,
        use_qk_norm: bool = False,
        fc_bias: bool = False,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0

        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.scale = head_dim**-0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=fc_bias)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=fc_bias)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=fc_bias)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=fc_bias)

        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=norm_eps)

    def _reshape_heads(self, x: mx.array, n_heads: int) -> mx.array:
        """Reshape using SAM-Audio's non-standard order.

        SAM-Audio: (B, T, C) -> (B, T, C/H, H) -> (B, H, T, C/H)
        where C/H = head_dim and H = n_heads

        This interleaves the head dimension differently than standard implementations.
        """
        B, T, C = x.shape
        # B x T x C -> B x T x C/H x H (head_dim, n_heads order)
        x = x.reshape(B, T, C // n_heads, n_heads)
        # B x T x C/H x H -> B x H x T x C/H
        return mx.transpose(x, (0, 3, 1, 2))

    def __call__(
        self,
        x: mx.array,
        cross_x: Optional[mx.array] = None,
        key_padding_mask: Optional[mx.array] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        xq = self.wq(x)
        if cross_x is not None:
            xk = self.wk(cross_x)
            xv = self.wv(cross_x)
            kv_seq_len = cross_x.shape[1]
        else:
            xk = self.wk(x)
            xv = self.wv(x)
            kv_seq_len = seq_len

        # Reshape using SAM-Audio's non-standard order: (B, T, head_dim, n_heads) -> (B, n_heads, T, head_dim)
        xq = self._reshape_heads(xq, self.n_heads)
        xk = self._reshape_heads(xk, self.n_kv_heads)
        xv = self._reshape_heads(xv, self.n_kv_heads)

        # Apply QK normalization (operates on last dim = head_dim)
        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # Apply RoPE (only for self-attention)
        # RoPE expects bhle=True since we're in (B, H, T, E) format
        if rope is not None and cross_x is None:
            xq = rope(xq, bhle=True)
            xk = rope(xk, bhle=True)

        # Handle GQA (grouped query attention)
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            xk = mx.repeat(xk, n_rep, axis=1)
            xv = mx.repeat(xv, n_rep, axis=1)

        # Scaled dot-product attention
        scores = (xq @ mx.transpose(xk, (0, 1, 3, 2))) * self.scale

        # Apply attention mask
        if key_padding_mask is not None:
            # key_padding_mask: (batch, kv_seq_len) where True = attend, False = mask out
            # mx.where(cond, x, y): returns x where cond is True, y where False
            mask = key_padding_mask[:, None, None, :]
            scores = mx.where(mask, scores, mx.array(float("-inf")))

        weights = mx.softmax(scores, axis=-1)
        output = weights @ xv

        # Reshape back: (B, H, T, D) -> (B, T, H*D)
        # Using einops notation: "b h n d -> b n (h d)"
        output = mx.transpose(output, (0, 2, 1, 3))  # (B, T, H, D)
        output = output.reshape(batch_size, seq_len, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU or other activations."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        ffn_dim_multiplier: float = 1.0,
        multiple_of: int = 64,
        dropout: float = 0.0,
        non_linearity: str = "swiglu",
        fc_bias: bool = False,
    ):
        super().__init__()
        self.dropout_rate = dropout
        self.swiglu = non_linearity == "swiglu"

        # SwiGLU hidden dim adjustment
        if self.swiglu:
            hidden_dim = int(2 * hidden_dim / 3)

        # Apply multiplier and round
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=fc_bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=fc_bias)

        if self.swiglu:
            self.w3 = nn.Linear(dim, hidden_dim, bias=fc_bias)

        self.non_linearity = get_nonlinearity(non_linearity)

    def __call__(self, x: mx.array) -> mx.array:
        hidden1 = self.w1(x)
        if self.swiglu:
            hidden3 = self.w3(x)
            hidden = nn.silu(hidden1) * hidden3
        else:
            hidden = self.non_linearity(hidden1)
        return self.w2(hidden)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embeddings with projection."""

    def __init__(
        self,
        dim: int,
        frequency_embedding_dim: int,
        non_linearity: str,
        dropout: float = 0.0,
        fc_bias: bool = False,
        max_period: int = 10000,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_dim

        self.projection = ProjectionLayer(
            in_dim=frequency_embedding_dim,
            out_dim=dim,
            non_linearity=non_linearity,
            dropout=dropout,
            fc_bias=fc_bias,
        )

        # Precompute frequencies
        half = frequency_embedding_dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        self._freqs = freqs

    def _timestep_embedding(self, t: mx.array, dim: int) -> mx.array:
        """Create sinusoidal timestep embeddings."""
        args = t[:, None].astype(mx.float32) * self._freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding.astype(t.dtype)

    def __call__(self, t: mx.array) -> mx.array:
        x = self._timestep_embedding(t, self.frequency_embedding_size)
        return self.projection(x)


class ContextEmbedder(nn.Module):
    """Context embedder for cross-attention memory."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        non_linearity: str,
        dropout: float = 0.0,
        fc_bias: bool = False,
        norm_eps: float = 1e-5,
        context_norm: bool = False,
    ):
        super().__init__()
        self.context_norm = context_norm

        if context_norm:
            self.norm = RMSNorm(in_dim, norm_eps)

        self.projection = ProjectionLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            non_linearity=non_linearity,
            dropout=dropout,
            fc_bias=fc_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.context_norm:
            x = self.norm(x)
        return self.projection(x)


class DiTBlock(nn.Module):
    """Diffusion Transformer Block with adaptive layer norm modulation."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        fc_bias: bool = False,
        ffn_exp: int = 1,
        ffn_dim_multiplier: float = 4.0,
        multiple_of: int = 64,
        non_linearity: str = "silu",
        no_cross_attention: bool = False,
    ):
        super().__init__()
        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.dim = dim
        self.head_dim = dim // n_heads

        # Self-attention
        self.attention = Attention(
            dim=dim,
            head_dim=self.head_dim,
            n_heads=n_heads,
            n_kv_heads=self.n_kv_heads,
            norm_eps=norm_eps,
            use_qk_norm=qk_norm,
            fc_bias=fc_bias,
        )

        # Feed-forward
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=int(ffn_exp * dim),
            ffn_dim_multiplier=ffn_dim_multiplier,
            multiple_of=multiple_of,
            dropout=dropout,
            non_linearity=non_linearity,
            fc_bias=fc_bias,
        )

        # Layer norms
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

        # Cross-attention (optional)
        self.cross_attention = None
        if not no_cross_attention:
            self.cross_attention = Attention(
                dim=dim,
                head_dim=self.head_dim,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                norm_eps=norm_eps,
                use_qk_norm=qk_norm,
                fc_bias=fc_bias,
            )

        # Adaptive LayerNorm modulation parameters
        self.scale_shift_table = mx.random.normal((6, dim)) / (dim**0.5)

    def __call__(
        self,
        x: mx.array,
        cross_x: Optional[mx.array],
        t: mx.array,
        padding_mask: Optional[mx.array],
        memory_padding_mask: Optional[mx.array],
        rope: Optional[RotaryEmbedding] = None,
    ) -> mx.array:
        # Get modulation parameters from timestep embedding
        biases = self.scale_shift_table[None] + t.reshape(x.shape[0], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            biases, 6, axis=1
        )

        # Self-attention with modulation
        h_normed = self.attention_norm(x)
        h_modulated = h_normed * (1 + scale_msa) + shift_msa
        h_attn = self.attention(
            h_modulated,
            key_padding_mask=padding_mask,
            rope=rope,
        )
        h = x + h_attn * gate_msa

        # Cross-attention (if enabled)
        if self.cross_attention is not None and cross_x is not None:
            h_cross = self.cross_attention(
                x=h,
                cross_x=cross_x,
                key_padding_mask=memory_padding_mask,
            )
            h = h + h_cross

        # Feed-forward with modulation
        h_normed = self.ffn_norm(h)
        h_modulated = h_normed * (1 + scale_mlp) + shift_mlp
        h_ff = self.feed_forward(h_modulated)
        out = h + h_ff * gate_mlp

        return out


class DiT(nn.Module):
    """Diffusion Transformer for audio generation."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dropout_rate = config.dropout

        # Input projection (optional)
        if config.in_channels is not None:
            self.data_proj = nn.Linear(config.in_channels, config.dim)
        else:
            self.data_proj = None

        # Rotary embeddings (optional)
        self.rope_embeddings = None
        if config.use_rope:
            self.rope_embeddings = RotaryEmbedding(
                theta=max(10000, 2 * config.max_positions),
                head_dim=config.dim // config.n_heads,
                max_seqlen=config.max_positions,
            )

        # Transformer blocks
        self.layers = [
            DiTBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                dropout=config.dropout,
                norm_eps=config.norm_eps,
                qk_norm=config.qk_norm,
                fc_bias=config.fc_bias,
                ffn_exp=config.ffn_exp,
                ffn_dim_multiplier=config.ffn_dim_multiplier,
                multiple_of=config.multiple_of,
                non_linearity=config.non_linearity,
            )
            for _ in range(config.n_layers)
        ]

        # Final norm
        self.norm = RMSNorm(config.dim, config.norm_eps)

        # Output projection
        self.output = nn.Linear(config.dim, config.out_channels, bias=config.fc_bias)

        # Input patcher
        self.x_embedder = Patcher(
            in_channels=config.dim,
            out_channels=config.dim,
            patch_size=1,
        )

        # Context embedder
        self.y_embedder = ContextEmbedder(
            in_dim=config.context_dim,
            out_dim=config.dim,
            non_linearity=config.context_non_linearity,
            dropout=config.context_embedder_dropout,
            fc_bias=config.fc_bias,
            norm_eps=config.norm_eps,
            context_norm=config.context_norm,
        )

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(
            config.dim,
            config.frequency_embedding_dim,
            non_linearity=config.timestep_non_linearity,
            dropout=config.dropout,
            fc_bias=config.fc_bias,
        )

        # Timestep block
        self.t_block_non_linearity = get_nonlinearity(config.t_block_non_linearity)
        self.t_block = nn.Linear(config.dim, config.dim * 6, bias=config.t_block_bias)

        # Final layer modulation
        self.final_layer_scale_shift_table = mx.random.normal((2, config.dim)) / (
            config.dim**0.5
        )

    def __call__(
        self,
        x: mx.array,
        time: mx.array,
        padding_mask: Optional[mx.array] = None,
        memory: Optional[mx.array] = None,
        memory_padding_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass of the DiT.

        Args:
            x: Input features (batch, seq_len, dim)
            time: Timestep tensor (batch,)
            padding_mask: Padding mask for input (batch, seq_len)
            memory: Cross-attention memory (batch, mem_len, context_dim)
            memory_padding_mask: Padding mask for memory (batch, mem_len)

        Returns:
            Output features (batch, seq_len, out_channels)
        """
        # Rearrange to (batch, dim, seq_len) for patcher
        x = mx.transpose(x, (0, 2, 1))
        h = self.x_embedder(x)
        # Rearrange back to (batch, seq_len, dim)
        h = mx.transpose(h, (0, 2, 1))
        original_N = h.shape[1]

        # Timestep embedding
        t = self.t_embedder(time)
        t0 = self.t_block_non_linearity(t)
        t0 = self.t_block(t0)

        # Context embedding
        y = self.y_embedder(memory) if memory is not None else None

        # Pass through transformer blocks
        for layer in self.layers:
            h = layer(
                x=h,
                cross_x=y,
                t=t0,
                padding_mask=padding_mask,
                memory_padding_mask=memory_padding_mask,
                rope=self.rope_embeddings,
            )

        # Final modulation
        shift, scale = mx.split(
            self.final_layer_scale_shift_table[None] + t[:, None], 2, axis=1
        )

        # Final norm and output
        h = self.norm(h)
        h = h * (1 + scale) + shift
        output = self.output(h)

        # Handle sequence length mismatch
        if original_N != output.shape[1]:
            output = output[:, -original_N:]

        return output
