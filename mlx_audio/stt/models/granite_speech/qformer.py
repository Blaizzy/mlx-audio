"""BLIP-2 Q-Former based projector for Granite Speech.

Maps encoder output to LLM-compatible embeddings using window-based cross-attention
with learnable query tokens.

Weight key structure:
  projector.query                                         (1, num_queries, hidden_size)
  projector.qformer.encoder.layer.{i}.attention.attention.{query,key,value}.{weight,bias}
  projector.qformer.encoder.layer.{i}.attention.output.dense.{weight,bias}
  projector.qformer.encoder.layer.{i}.attention.output.LayerNorm.{weight,bias}
  projector.qformer.encoder.layer.{i}.crossattention.attention.{query,key,value}.{weight,bias}
  projector.qformer.encoder.layer.{i}.crossattention.output.dense.{weight,bias}
  projector.qformer.encoder.layer.{i}.crossattention.output.LayerNorm.{weight,bias}
  projector.qformer.encoder.layer.{i}.intermediate_query.dense.{weight,bias}
  projector.qformer.encoder.layer.{i}.output_query.dense.{weight,bias}
  projector.qformer.encoder.layer.{i}.output_query.LayerNorm.{weight,bias}
  projector.qformer.layernorm.{weight,bias}
  projector.linear.{weight,bias}
"""

import mlx.core as mx
import mlx.nn as nn

from .config import ProjectorConfig


class QFormerMultiHeadAttention(nn.Module):
    """Multi-head attention for Q-Former (self-attention or cross-attention).

    Keys: {attention,crossattention}.attention.{query,key,value}
    """

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array = None
    ) -> mx.array:
        B, S, _ = hidden_states.shape

        q = self.query(hidden_states)
        if encoder_hidden_states is not None:
            k = self.key(encoder_hidden_states)
            v = self.value(encoder_hidden_states)
        else:
            k = self.key(hidden_states)
            v = self.value(hidden_states)

        KS = k.shape[1]

        q = q.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, KS, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, KS, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = self.head_dim**-0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return out


class QFormerSelfOutput(nn.Module):
    """Output projection + LayerNorm for Q-Former attention.

    Keys: {attention,crossattention}.output.{dense,LayerNorm}
    Note: HF uses 'LayerNorm' (capital) in weight keys.
    """

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, hidden_states: mx.array, residual: mx.array) -> mx.array:
        h = self.dense(hidden_states)
        return self.LayerNorm(h + residual)


class QFormerAttention(nn.Module):
    """Q-Former attention block (self-attn or cross-attn with output projection).

    Keys: {attention,crossattention}.{attention,output}.*
    """

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.attention = QFormerMultiHeadAttention(config)
        self.output = QFormerSelfOutput(config)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array = None
    ) -> mx.array:
        attn_out = self.attention(hidden_states, encoder_hidden_states)
        return self.output(attn_out, hidden_states)


class QFormerIntermediate(nn.Module):
    """Q-Former FFN intermediate layer.

    Keys: intermediate_query.dense
    """

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return nn.gelu(self.dense(hidden_states))


class QFormerOutput(nn.Module):
    """Q-Former FFN output layer.

    Keys: output_query.{dense,LayerNorm}
    """

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, hidden_states: mx.array, residual: mx.array) -> mx.array:
        h = self.dense(hidden_states)
        return self.LayerNorm(h + residual)


class QFormerLayer(nn.Module):
    """Single Q-Former layer: self-attention + cross-attention + FFN.

    Keys: encoder.layer.{i}.{attention,crossattention,intermediate_query,output_query}
    """

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.attention = QFormerAttention(config)
        self.crossattention = QFormerAttention(config)
        self.intermediate_query = QFormerIntermediate(config)
        self.output_query = QFormerOutput(config)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array
    ) -> mx.array:
        # Self-attention
        h = self.attention(hidden_states)

        # Cross-attention with encoder output
        h = self.crossattention(h, encoder_hidden_states)

        # FFN
        intermediate = self.intermediate_query(h)
        h = self.output_query(intermediate, h)

        return h


class QFormerEncoder(nn.Module):
    """Stack of Q-Former layers.

    Keys: encoder.layer.{i}.*
    """

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.layer = [QFormerLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array
    ) -> mx.array:
        for layer in self.layer:
            hidden_states = layer(hidden_states, encoder_hidden_states)
        return hidden_states


class QFormerModel(nn.Module):
    """BLIP-2 Q-Former model.

    Keys: qformer.{encoder,layernorm}.*
    """

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.encoder = QFormerEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self, query_embeds: mx.array, encoder_hidden_states: mx.array
    ) -> mx.array:
        # HF BLIP-2: LayerNorm applied to query BEFORE encoder, not after
        h = self.layernorm(query_embeds)
        return self.encoder(h, encoder_hidden_states)


class EncoderProjector(nn.Module):
    """Window Q-Former projector: splits encoder output into windows,
    applies Q-Former cross-attention, then projects to LLM dim.

    Keys: projector.{query,qformer,linear}.*
    """

    def __init__(
        self,
        config: ProjectorConfig,
        window_size: int,
        downsample_rate: int,
        num_queries: int,
        output_dim: int,
    ):
        super().__init__()
        self.window_size = window_size
        self.downsample_rate = downsample_rate
        self.num_queries = num_queries

        self.query = mx.zeros((1, num_queries, config.hidden_size))
        self.qformer = QFormerModel(config)
        self.linear = nn.Linear(config.hidden_size, output_dim, bias=True)

    def __call__(self, encoder_output: mx.array) -> mx.array:
        """Project encoder output to LLM embedding space.

        Splits encoder output into non-overlapping windows of window_size,
        applies Q-Former cross-attention per window, then projects to LLM dim.

        Args:
            encoder_output: (batch, seq_len, hidden_dim)

        Returns:
            (batch, nblocks * num_queries, llm_dim)
        """
        import math

        B, T, C = encoder_output.shape

        # Pad to multiple of window_size, then split into non-overlapping windows
        nblocks = math.ceil(T / self.window_size)
        pad_len = nblocks * self.window_size - T
        if pad_len > 0:
            encoder_output = mx.pad(encoder_output, [(0, 0), (0, pad_len), (0, 0)])

        # Reshape into (B * nblocks, window_size, C)
        encoder_output = encoder_output.reshape(B * nblocks, self.window_size, C)

        # Q-Former: process all windows at once
        query = mx.broadcast_to(self.query, (B * nblocks, self.num_queries, C))
        qformer_out = self.qformer(query, encoder_output)  # (B*nblocks, num_queries, C)

        # Reshape back to (B, nblocks * num_queries, C) and project
        projected = qformer_out.reshape(B, nblocks * self.num_queries, C)
        return self.linear(projected)
