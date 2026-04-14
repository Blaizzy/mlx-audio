"""BERT encoder for MeloTTS prosodic feature extraction."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..base import BaseModelArgs


@dataclass
class BertConfig(BaseModelArgs):
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)

        position_ids = mx.arange(input_ids.shape[1])[None, :]

        embeddings = (
            self.word_embeddings(input_ids)
            + self.token_type_embeddings(token_type_ids)
            + self.position_embeddings(position_ids)
        )
        return self.norm(embeddings)


class TransformerEncoderLayer(nn.Module):
    """Single BERT transformer layer (post-norm)."""

    def __init__(self, dims: int, num_heads: int, mlp_dims: int):
        super().__init__()
        self.attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.ln1 = nn.LayerNorm(dims)
        self.ln2 = nn.LayerNorm(dims)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)
        self.act = nn.GELU()

    def __call__(self, x, mask=None):
        attention_out = self.attention(x, x, x, mask=mask)
        x = self.ln1(x + attention_out)
        ffn_out = self.linear2(self.act(self.linear1(x)))
        return self.ln2(x + ffn_out)


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, dims: int, num_heads: int, mlp_dims: int):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(dims, num_heads, mlp_dims)
            for _ in range(num_layers)
        ]

    def __call__(self, x, mask=None, output_hidden_states=False):
        all_hidden_states = [x] if output_hidden_states else None

        for layer in self.layers:
            x = layer(x, mask=mask)
            if output_hidden_states:
                all_hidden_states.append(x)

        return x, all_hidden_states


class BertModel(nn.Module):
    """MLX BERT model compatible with mlx-examples/bert weights."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.hidden_size,
            num_heads=config.num_attention_heads,
            mlp_dims=config.intermediate_size,
        )
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_hidden_states=False,
    ):
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            attention_mask = mx.where(
                attention_mask[:, None, None, :] == 0, -mx.inf, 0.0
            )

        x, all_hidden_states = self.encoder(
            x, mask=attention_mask, output_hidden_states=output_hidden_states
        )

        pooled = mx.tanh(self.pooler(x[:, 0]))
        return x, pooled, all_hidden_states

    def extract_features(self, input_ids, token_type_ids=None, attention_mask=None):
        """Extract features from the 3rd-to-last hidden layer."""
        _, _, all_hidden_states = self(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return all_hidden_states[-3]

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            sanitized[k] = v
        return sanitized
