"""Typed configuration mirroring the HF config.json schema.

Single source of JSON-shape knowledge in the package. Modules consume the
typed dataclasses; no other file should call json.loads on config.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EncoderConfig:
    num_layers: int
    hidden_dim: int
    num_heads: int
    dim_head: int
    input_dim: int
    output_dim: int
    bpe_output_dim: int
    bpe_pooling_window: int
    conv_kernel_size: int
    conv_expansion_factor: int
    feedforward_mult: int
    max_pos_emb: int
    context_size: int
    self_conditioning_layer: int
    blank_token_id: int
    dropout: float
    pred_dropout: float
    initializer_range: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EncoderConfig":
        return cls(
            num_layers=d["num_layers"],
            hidden_dim=d["hidden_dim"],
            num_heads=d["num_heads"],
            dim_head=d["dim_head"],
            input_dim=d["input_dim"],
            output_dim=d["output_dim"],
            bpe_output_dim=d["bpe_output_dim"],
            bpe_pooling_window=d["bpe_pooling_window"],
            conv_kernel_size=d["conv_kernel_size"],
            conv_expansion_factor=d["conv_expansion_factor"],
            feedforward_mult=d["feedforward_mult"],
            max_pos_emb=d["max_pos_emb"],
            context_size=d["context_size"],
            self_conditioning_layer=d["self_conditioning_layer"],
            blank_token_id=d["blank_token_id"],
            dropout=d["dropout"],
            pred_dropout=d["pred_dropout"],
            initializer_range=d["initializer_range"],
        )


@dataclass(frozen=True)
class ProjectorConfig:
    num_layers: int
    num_encoder_layers: int
    hidden_size: int
    num_heads: int
    block_size: int
    downsample_rate: int
    encoder_dim: int
    llm_dim: int
    mlp_ratio: int
    mlp_bias: bool
    attn_bias: bool
    dropout_prob: float
    layernorm_eps: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProjectorConfig":
        return cls(
            num_layers=d["num_layers"],
            num_encoder_layers=d["num_encoder_layers"],
            hidden_size=d["hidden_size"],
            num_heads=d["num_heads"],
            block_size=d["block_size"],
            downsample_rate=d["downsample_rate"],
            encoder_dim=d["encoder_dim"],
            llm_dim=d["llm_dim"],
            mlp_ratio=d["mlp_ratio"],
            mlp_bias=d["mlp_bias"],
            attn_bias=d["attn_bias"],
            dropout_prob=d["dropout_prob"],
            layernorm_eps=d["layernorm_eps"],
        )


@dataclass(frozen=True)
class TextConfig:
    """Granite editor LLM config. Preserves Granite-specific multipliers."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    # Granite-specific
    attention_multiplier: float
    embedding_multiplier: float
    logits_scaling: float
    residual_multiplier: float
    # tokens
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextConfig":
        return cls(
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            vocab_size=d["vocab_size"],
            max_position_embeddings=d["max_position_embeddings"],
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=d["rope_parameters"]["rope_theta"],
            tie_word_embeddings=d["tie_word_embeddings"],
            attention_multiplier=d["attention_multiplier"],
            embedding_multiplier=d["embedding_multiplier"],
            logits_scaling=d["logits_scaling"],
            residual_multiplier=d["residual_multiplier"],
            bos_token_id=d["bos_token_id"],
            eos_token_id=d["eos_token_id"],
            pad_token_id=d["pad_token_id"],
        )


@dataclass(frozen=True)
class ModelConfig:
    encoder: EncoderConfig
    projector: ProjectorConfig
    text: TextConfig
    encoder_layer_indices: list[int]
    blank_token_id: int
    scale_projected_embeddings: bool
    min_edit_sequence_length: int
    tie_word_embeddings: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(
            encoder=EncoderConfig.from_dict(d["encoder_config"]),
            projector=ProjectorConfig.from_dict(d["projector_config"]),
            text=TextConfig.from_dict(d["text_config"]),
            encoder_layer_indices=list(d["encoder_layer_indices"]),
            blank_token_id=d["blank_token_id"],
            scale_projected_embeddings=d["scale_projected_embeddings"],
            min_edit_sequence_length=d["min_edit_sequence_length"],
            tie_word_embeddings=d["tie_word_embeddings"],
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "ModelConfig":
        path = Path(path)
        raw = json.loads((path / "config.json").read_text())
        return cls.from_dict(raw)
