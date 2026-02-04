# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Text encoder for ACE-Step using MLX Qwen3."""

import glob
import json
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3 import ModelArgs as Qwen3ModelArgs


def _load_weights(model_path: Path) -> dict:
    """Load weights from safetensors files."""
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    return weights


def _sanitize_weights(weights: dict) -> dict:
    """Add 'model.' prefix to weights for mlx_lm Qwen3 compatibility.

    The Qwen3-Embedding weights have keys like 'embed_tokens.weight' but
    mlx_lm expects 'model.embed_tokens.weight'.
    """
    sanitized = {}
    for key, value in weights.items():
        # Skip lm_head as it's tied to embeddings
        if key == "lm_head.weight":
            continue
        # Add model. prefix if not present
        if not key.startswith("model."):
            new_key = f"model.{key}"
        else:
            new_key = key
        sanitized[new_key] = value
    return sanitized


class TextEncoder:
    """Text encoder using MLX Qwen3 model.

    This encoder uses mlx-lm's Qwen3 implementation for native MLX inference,
    avoiding the need for PyTorch/transformers dependencies at runtime.
    """

    def __init__(self, model_path: str):
        """Initialize the text encoder.

        Args:
            model_path: Path to the Qwen3-Embedding model directory
        """
        from transformers import AutoTokenizer

        self.model_path = Path(model_path)

        # Load config
        config_path = self.model_path / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Create model args
        args = Qwen3ModelArgs(
            model_type=config.get("model_type", "qwen3"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            intermediate_size=config["intermediate_size"],
            num_attention_heads=config["num_attention_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            vocab_size=config["vocab_size"],
            num_key_value_heads=config["num_key_value_heads"],
            max_position_embeddings=config.get("max_position_embeddings", 32768),
            rope_theta=config.get("rope_theta", 1000000.0),
            head_dim=config.get(
                "head_dim", config["hidden_size"] // config["num_attention_heads"]
            ),
            tie_word_embeddings=config.get("tie_word_embeddings", True),
            rope_scaling=config.get("rope_scaling", None),
        )

        # Create model
        self.model = Qwen3Model(args)

        # Load and sanitize weights
        weights = _load_weights(self.model_path)
        weights = _sanitize_weights(weights)
        self.model.load_weights(list(weights.items()))

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Get hidden size from model config
        self.hidden_size = args.hidden_size

    def embed_tokens(
        self,
        text: str,
        max_length: int = 1024,
    ) -> Tuple[mx.array, mx.array]:

        tokens = self.tokenizer.encode(text)

        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        input_ids = mx.array([tokens])

        embeddings = self.model.model.embed_tokens(input_ids)

        attention_mask = mx.ones((1, len(tokens)))

        return embeddings, attention_mask

    def encode(
        self,
        text: str,
        max_length: int = 256,
    ) -> Tuple[mx.array, mx.array]:

        tokens = self.tokenizer.encode(text)

        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        input_ids = mx.array([tokens])

        hidden_states = self.model.model(input_ids)

        attention_mask = mx.ones((1, len(tokens)))

        return hidden_states, attention_mask

    def encode_batch(
        self,
        texts: list,
        max_length: int = 512,
    ) -> Tuple[mx.array, mx.array]:

        all_tokens = [self.tokenizer.encode(text) for text in texts]

        # Truncate and find max length
        all_tokens = [tokens[:max_length] for tokens in all_tokens]
        batch_max_len = max(len(tokens) for tokens in all_tokens)

        # Pad to same length
        pad_token_id = self.tokenizer.pad_token_id or 0
        padded_tokens = []
        attention_masks = []

        for tokens in all_tokens:
            pad_len = batch_max_len - len(tokens)
            padded = tokens + [pad_token_id] * pad_len
            mask = [1] * len(tokens) + [0] * pad_len
            padded_tokens.append(padded)
            attention_masks.append(mask)

        input_ids = mx.array(padded_tokens)
        attention_mask = mx.array(attention_masks)

        hidden_states = self.model.model(input_ids)

        mx.eval(hidden_states, attention_mask)

        return hidden_states, attention_mask


def load_text_encoder(model_path: Optional[str] = None) -> TextEncoder:
    """Load the text encoder.

    Args:
        model_path: Path to ACE-Step model. If None, downloads from HuggingFace.

    Returns:
        TextEncoder instance
    """
    if model_path is None:
        from huggingface_hub import snapshot_download

        model_path = snapshot_download("ACE-Step/ACE-Step1.5")

    qwen_path = Path(model_path) / "Qwen3-Embedding-0.6B"
    return TextEncoder(str(qwen_path))
