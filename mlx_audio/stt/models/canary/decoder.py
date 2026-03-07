from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import DecoderConfig


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention layer.

    Attends to encoder output (key/value from encoder, query from decoder).
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def __call__(
        self,
        x: mx.array,
        encoder_output: mx.array,
        encoder_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        B, T, _ = x.shape

        q = self.q_proj(x)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache
        else:
            k = self.k_proj(encoder_output)
            v = self.v_proj(encoder_output)
            S = encoder_output.shape[1]
            k = k.reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        mask = None
        if encoder_mask is not None:
            mask = encoder_mask[:, None, None, :].astype(mx.float32)
            mask = mx.where(mask == 0, -1e9, 0.0)

        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)

        return self.out_proj(o), (k, v)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head causal self-attention layer with KV-cache support."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            prev_k, prev_v = cache
            k = mx.concatenate([prev_k, k], axis=2)
            v = mx.concatenate([prev_v, v], axis=2)

        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)

        return self.out_proj(o), (k, v)


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block with self-attention + cross-attention.

    NeMo uses pre-norm (LayerNorm before attention/FFN).
    """

    def __init__(self, d_model: int, n_heads: int, inner_size: int):
        super().__init__()

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads)

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads)

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, inner_size, bias=True)
        self.ff2 = nn.Linear(inner_size, d_model, bias=True)

    def __call__(
        self,
        x: mx.array,
        encoder_output: mx.array,
        encoder_mask: Optional[mx.array] = None,
        self_attn_mask: Optional[mx.array] = None,
        self_attn_cache: Optional[Tuple[mx.array, mx.array]] = None,
        cross_attn_cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array], Tuple[mx.array, mx.array]]:
        residual = x
        x_norm = self.self_attn_norm(x)
        x_sa, new_self_cache = self.self_attn(x_norm, mask=self_attn_mask, cache=self_attn_cache)
        x = residual + x_sa

        residual = x
        x_norm = self.cross_attn_norm(x)
        x_ca, new_cross_cache = self.cross_attn(
            x_norm, encoder_output, encoder_mask=encoder_mask, cache=cross_attn_cache
        )
        x = residual + x_ca

        residual = x
        x_norm = self.ff_norm(x)
        x = residual + self.ff2(nn.relu(self.ff1(x_norm)))

        return x, new_self_cache, new_cross_cache


class FixedPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (matches NeMo's implementation)."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.pos_enc = mx.zeros((max_len, d_model))

    def __call__(self, position_ids: mx.array) -> mx.array:
        """Get position embeddings for the given position IDs."""
        return self.pos_enc[position_ids]


class CanaryDecoder(nn.Module):
    """Canary transformer decoder.

    Takes encoder output and generates text tokens autoregressively.
    Includes token embedding, positional encoding, and layer norm
    (matching NeMo's TransformerEmbedding).
    """

    def __init__(self, config: DecoderConfig, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.num_layers = config.num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = FixedPositionalEncoding(d_model)
        self.embedding_layer_norm = nn.LayerNorm(d_model)

        self.blocks = [
            TransformerDecoderBlock(
                d_model=d_model,
                n_heads=config.num_attention_heads,
                inner_size=config.inner_size,
            )
            for _ in range(config.num_layers)
        ]
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=True)

    def __call__(
        self,
        tokens: mx.array,
        encoder_output: mx.array,
        encoder_mask: Optional[mx.array] = None,
        cache: Optional[List[dict]] = None,
        start_pos: int = 0,
    ) -> Tuple[mx.array, List[dict]]:
        """Forward pass through decoder.

        Args:
            tokens: Token IDs, shape (B, T)
            encoder_output: Encoder states, shape (B, S, D)
            encoder_mask: Encoder mask, shape (B, S)
            cache: KV-cache from previous steps
            start_pos: Starting position for positional encoding

        Returns:
            logits: Shape (B, T, vocab_size)
            new_cache: Updated KV-cache
        """
        B, T = tokens.shape

        x = self.embedding(tokens)

        position_ids = mx.arange(start_pos, start_pos + T)
        position_ids = mx.broadcast_to(position_ids[None, :], (B, T))
        x = x + self.position_embedding(position_ids)

        x = self.embedding_layer_norm(x)

        if cache is None:
            cache = [{"self_attn": None, "cross_attn": None} for _ in range(self.num_layers)]

        if T > 1:
            causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            if cache[0]["self_attn"] is not None:
                cached_len = cache[0]["self_attn"][0].shape[2]
                prefix = mx.zeros((T, cached_len))
                causal_mask = mx.concatenate([prefix, causal_mask], axis=1)
        else:
            causal_mask = None

        new_cache = []
        for i, block in enumerate(self.blocks):
            x, new_self, new_cross = block(
                x,
                encoder_output,
                encoder_mask=encoder_mask,
                self_attn_mask=causal_mask,
                self_attn_cache=cache[i]["self_attn"],
                cross_attn_cache=cache[i]["cross_attn"],
            )
            new_cache.append({"self_attn": new_self, "cross_attn": new_cross})

        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits, new_cache
