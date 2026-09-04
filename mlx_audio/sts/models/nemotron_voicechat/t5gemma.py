from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1.0e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, inputs: mx.array) -> mx.array:
        return mx.fast.rms_norm(inputs, 1.0 + self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.down_proj(
            nn.gelu_approx(self.gate_proj(inputs)) * self.up_proj(inputs)
        )


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.repeats = num_attention_heads // num_key_value_heads
        self.scale = 256**-0.5
        self.softcap = 50.0
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.rope = nn.RoPE(head_dim, traditional=False, base=10_000.0)

    def __call__(
        self, inputs: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        batch, length, _ = inputs.shape
        queries = self.q_proj(inputs).reshape(
            batch, length, self.num_attention_heads, self.head_dim
        )
        keys = self.k_proj(inputs).reshape(
            batch, length, self.num_key_value_heads, self.head_dim
        )
        values = self.v_proj(inputs).reshape(
            batch, length, self.num_key_value_heads, self.head_dim
        )
        queries = self.rope(queries.transpose(0, 2, 1, 3))
        keys = self.rope(keys.transpose(0, 2, 1, 3))
        values = values.transpose(0, 2, 1, 3)

        if self.repeats > 1:
            keys = mx.repeat(keys, self.repeats, axis=1)
            values = mx.repeat(values, self.repeats, axis=1)

        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        scores = mx.tanh(scores / self.softcap) * self.softcap
        if attention_mask is not None:
            scores = mx.where(attention_mask[:, None, None, :], scores, -mx.inf)
        probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(
            queries.dtype
        )
        outputs = probabilities @ values
        outputs = outputs.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(outputs)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.self_attn = SelfAttention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
        )
        self.pre_self_attn_layernorm = RMSNorm(hidden_size)
        self.post_self_attn_layernorm = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size)
        self.post_feedforward_layernorm = RMSNorm(hidden_size)

    def __call__(
        self, inputs: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        hidden = self.pre_self_attn_layernorm(inputs)
        hidden = self.self_attn(hidden, attention_mask)
        inputs = inputs + self.post_self_attn_layernorm(hidden)
        hidden = self.pre_feedforward_layernorm(inputs)
        hidden = self.mlp(hidden)
        return inputs + self.post_feedforward_layernorm(hidden)


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        num_hidden_layers: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = [
            EncoderLayer(
                hidden_size,
                intermediate_size,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = RMSNorm(hidden_size)

    def __call__(
        self, inputs: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        hidden = inputs * mx.array(self.hidden_size**0.5, dtype=inputs.dtype)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return self.norm(hidden)


class Model(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        num_hidden_layers: int = 1,
    ):
        super().__init__()
        self.encoder = Encoder(
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            num_hidden_layers,
        )

    def __call__(
        self, inputs: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        return self.encoder(inputs, attention_mask)
