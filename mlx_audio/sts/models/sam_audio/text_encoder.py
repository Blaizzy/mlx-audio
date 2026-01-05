# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import T5EncoderConfig

# Suppress HTTPX and HuggingFace Hub logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@dataclass
class T5Config:
    """Configuration for T5 model."""

    vocab_size: int = 32128
    d_model: int = 768
    d_kv: int = 64
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    is_gated_act: bool = True  # T5 1.1 and later use gated activation
    dense_act_fn: str = "gelu_new"

    @classmethod
    def from_hf_config(cls, hf_config) -> "T5Config":
        """Create T5Config from HuggingFace T5Config."""
        return cls(
            vocab_size=hf_config.vocab_size,
            d_model=hf_config.d_model,
            d_kv=hf_config.d_kv,
            d_ff=hf_config.d_ff,
            num_layers=hf_config.num_layers,
            num_heads=hf_config.num_heads,
            relative_attention_num_buckets=hf_config.relative_attention_num_buckets,
            relative_attention_max_distance=hf_config.relative_attention_max_distance,
            dropout_rate=hf_config.dropout_rate,
            layer_norm_epsilon=hf_config.layer_norm_epsilon,
            is_gated_act=hf_config.is_gated_act,
            dense_act_fn=hf_config.dense_act_fn,
        )


class T5LayerNorm(nn.Module):
    """T5-style layer normalization (RMSNorm without mean subtraction)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # T5 uses RMSNorm: no mean subtraction, just variance normalization
        variance = mx.mean(
            hidden_states.astype(mx.float32) ** 2, axis=-1, keepdims=True
        )
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class T5DenseActDense(nn.Module):
    """T5 feed-forward with single activation (non-gated)."""

    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act_fn = config.dense_act_fn

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.wi(hidden_states)
        if self.act_fn == "relu":
            hidden_states = nn.relu(hidden_states)
        elif self.act_fn in ("gelu", "gelu_new"):
            hidden_states = nn.gelu(hidden_states)
        else:
            hidden_states = nn.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    """T5 feed-forward with gated activation (T5 1.1+)."""

    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act_fn = config.dense_act_fn

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # Gated activation: act(wi_0(x)) * wi_1(x)
        hidden_gelu = self.wi_0(hidden_states)
        if self.act_fn == "relu":
            hidden_gelu = nn.relu(hidden_gelu)
        elif self.act_fn in ("gelu", "gelu_new"):
            hidden_gelu = nn.gelu(hidden_gelu)
        else:
            hidden_gelu = nn.relu(hidden_gelu)

        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    """T5 feed-forward layer with pre-norm."""

    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    """T5 attention with relative position bias."""

    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias: bool = False,
    ):
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = nn.Dropout(config.dropout_rate)
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: mx.array,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> mx.array:
        """Compute bucketed relative position indices."""
        relative_buckets = mx.zeros(relative_position.shape, dtype=mx.int32)

        if bidirectional:
            num_buckets //= 2
            relative_buckets = (
                relative_buckets
                + (relative_position > 0).astype(mx.int32) * num_buckets
            )
            relative_position = mx.abs(relative_position)
        else:
            relative_position = -mx.minimum(
                relative_position, mx.zeros_like(relative_position)
            )

        # Half of buckets are for exact small positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Other half are for logarithmically bigger positions
        relative_position_if_large = max_exact + (
            mx.log(relative_position.astype(mx.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mx.int32)

        relative_position_if_large = mx.minimum(
            relative_position_if_large,
            mx.full(relative_position_if_large.shape, num_buckets - 1, dtype=mx.int32),
        )

        relative_buckets = relative_buckets + mx.where(
            is_small, relative_position.astype(mx.int32), relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int) -> mx.array:
        """Generate binned relative position bias values."""
        # Create position indices
        context_position = mx.arange(query_length)[:, None]
        memory_position = mx.arange(key_length)[None, :]

        # Compute relative position
        relative_position = memory_position - context_position

        # Bucket the positions
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,  # Encoder is always bidirectional
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # Look up bias values: (query_length, key_length, n_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # Reshape to (1, n_heads, query_length, key_length)
        values = mx.transpose(values, (2, 0, 1))[None, :, :, :]
        return values

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        position_bias: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Forward pass for T5 attention.

        Args:
            hidden_states: (batch, seq_len, d_model)
            mask: Optional attention mask (batch, 1, 1, seq_len) or similar
            position_bias: Optional pre-computed position bias

        Returns:
            Tuple of (output, position_bias)
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)

        # Reshape to (batch, n_heads, seq_len, d_kv)
        query_states = query_states.reshape(
            batch_size, seq_length, self.n_heads, self.key_value_proj_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            batch_size, seq_length, self.n_heads, self.key_value_proj_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            batch_size, seq_length, self.n_heads, self.key_value_proj_dim
        ).transpose(0, 2, 1, 3)

        # Attention scores: (batch, n_heads, seq_len, seq_len)
        scores = query_states @ key_states.transpose(0, 1, 3, 2)

        # Add position bias
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(seq_length, seq_length)
            else:
                position_bias = mx.zeros((1, self.n_heads, seq_length, seq_length))

        # Add position bias to scores
        scores = scores + position_bias

        # Apply attention mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax and dropout
        attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(
            scores.dtype
        )
        attn_weights = self.dropout(attn_weights)

        # Compute output
        attn_output = attn_weights @ value_states

        # Reshape back: (batch, seq_len, inner_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.inner_dim
        )

        # Output projection
        attn_output = self.o(attn_output)

        return attn_output, position_bias


class T5LayerSelfAttention(nn.Module):
    """T5 self-attention layer with pre-norm."""

    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_bias: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states, position_bias


class T5Block(nn.Module):
    """T5 transformer block (self-attention + feed-forward)."""

    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = [
            T5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            ),
            T5LayerFF(config),
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_bias: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        # Self-attention
        hidden_states, position_bias = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        # Feed-forward
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias


class T5Stack(nn.Module):
    """T5 encoder stack."""

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        # Note: embed_tokens is set externally after construction to share embeddings
        self._embed_tokens = None
        self.block = [
            T5Block(config, has_relative_attention_bias=(i == 0))
            for i in range(config.num_layers)
        ]
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def set_input_embeddings(self, embeddings: nn.Embedding):
        """Set the input embeddings (shared with main model)."""
        self._embed_tokens = embeddings

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass for T5 encoder stack.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) mask where 1=attend, 0=pad
            inputs_embeds: Optional pre-computed embeddings

        Returns:
            hidden_states: (batch, seq_len, d_model)
        """
        if inputs_embeds is None:
            if self._embed_tokens is None:
                raise ValueError("Must provide inputs_embeds or set_input_embeddings")
            inputs_embeds = self._embed_tokens(input_ids)

        # Convert attention mask to additive mask
        # HuggingFace: 1 = attend, 0 = mask
        # We need: 0 = attend, -inf = mask
        extended_attention_mask = None
        if attention_mask is not None:
            # (batch, seq_len) -> (batch, 1, 1, seq_len)
            extended_attention_mask = attention_mask[:, None, None, :]
            # Convert 0 -> -inf, 1 -> 0
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        hidden_states = self.dropout(inputs_embeds)
        position_bias = None

        for layer_module in self.block:
            hidden_states, position_bias = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class T5Encoder(nn.Module):
    """Pure MLX T5 Encoder model."""

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config)
        # Share embeddings between shared and encoder
        self.encoder.set_input_embeddings(self.shared)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Encode input tokens.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) mask where 1=attend, 0=pad

        Returns:
            hidden_states: (batch, seq_len, d_model)
        """
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array], prefix: str = "") -> Dict[str, mx.array]:
        """
        Sanitize T5 weights for MLX loading.

        Args:
            weights: Dictionary of weights
            prefix: Prefix to strip from keys (e.g., "text_encoder." for SAM-Audio)

        Returns:
            Sanitized weights ready for loading
        """
        sanitized = {}

        for key, value in weights.items():
            # Strip prefix if provided
            if prefix:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                else:
                    # Skip keys that don't match the prefix
                    continue

            # Skip decoder weights
            if "decoder" in key:
                continue

            # Map HuggingFace names to MLX names
            new_key = key

            # shared -> shared (embedding)
            # encoder.embed_tokens -> use shared instead
            if key == "encoder.embed_tokens.weight":
                new_key = "shared.weight"

            sanitized[new_key] = value

        return sanitized

    @classmethod
    def from_pretrained(cls, model_name: str = "t5-base") -> "T5Encoder":
        """Load pretrained T5 encoder from HuggingFace."""
        import json

        from huggingface_hub import hf_hub_download

        # Download config
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")

        with open(config_path, encoding="utf-8") as f:
            hf_config = json.load(f)

        # Create config
        config = T5Config(
            vocab_size=hf_config.get("vocab_size", 32128),
            d_model=hf_config.get("d_model", 768),
            d_kv=hf_config.get("d_kv", 64),
            d_ff=hf_config.get("d_ff", 3072),
            num_layers=hf_config.get("num_layers", 12),
            num_heads=hf_config.get("num_heads", 12),
            relative_attention_num_buckets=hf_config.get(
                "relative_attention_num_buckets", 32
            ),
            relative_attention_max_distance=hf_config.get(
                "relative_attention_max_distance", 128
            ),
            dropout_rate=hf_config.get("dropout_rate", 0.1),
            layer_norm_epsilon=hf_config.get("layer_norm_epsilon", 1e-6),
            is_gated_act=hf_config.get("is_gated_act", False),
            dense_act_fn=hf_config.get("dense_act_fn", "relu"),
        )

        # Create model
        model = cls(config)

        # Load weights
        try:
            weights_path = hf_hub_download(
                repo_id=model_name, filename="model.safetensors"
            )
            weights = mx.load(weights_path)
        except Exception:
            # Fall back to PyTorch weights
            import torch

            weights_path = hf_hub_download(
                repo_id=model_name, filename="pytorch_model.bin"
            )
            pt_weights = torch.load(weights_path, map_location="cpu")
            weights = {k: mx.array(v.numpy()) for k, v in pt_weights.items()}

        # Sanitize and load
        sanitized = cls.sanitize(weights)
        model.load_weights(list(sanitized.items()))
        mx.eval(model.parameters())

        # Set to eval mode (disables dropout)
        model.eval()

        return model


class T5TextEncoder:
    """
    Pure MLX T5 text encoder for SAM-Audio.

    Uses native MLX T5 implementation for better performance on Apple Silicon.
    Weights are loaded from HuggingFace (SAM-Audio checkpoint doesn't include T5 weights).
    """

    def __init__(self, config: T5EncoderConfig):
        self.config = config
        self.model: Optional[T5Encoder] = None
        self.tokenizer = None

    def _lazy_load(self):
        """Lazily load the T5 model and tokenizer."""
        if self.model is None:
            import transformers

            # Load tokenizer from HuggingFace
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.name
            )

            # Load MLX T5 model from HuggingFace
            self.model = T5Encoder.from_pretrained(self.config.name)

    def __call__(self, texts: List[str]) -> Tuple[mx.array, mx.array]:
        """
        Encode text descriptions.

        Args:
            texts: List of text descriptions

        Returns:
            Tuple of (features, attention_mask) as MLX arrays
            - features: (batch, seq_len, dim)
            - attention_mask: (batch, seq_len) boolean mask where True = attend, False = mask out
        """
        self._lazy_load()

        # Tokenize
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_length,
            padding=self.config.pad_mode,
            return_tensors="np",
        )

        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        # Encode with MLX model
        features = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Convert attention mask to boolean
        # HuggingFace: attention_mask=1 means "attend", =0 means "padding"
        # We keep same convention as PyTorch SAM-Audio: True = attend, False = mask out
        bool_mask = attention_mask.astype(mx.bool_)

        return features, bool_mask
