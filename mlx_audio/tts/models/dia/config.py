"""Configuration management module for the Dia model.

This module provides comprehensive configuration management for the Dia model,
utilizing dataclasses for validation. It defines configurations for data processing,
model architecture (encoder and decoder), and training settings.

Key components:
- DataConfig: Parameters for data loading and preprocessing.
- EncoderConfig: Architecture details for the encoder module.
- DecoderConfig: Architecture details for the decoder module.
- ModelConfig: Combined model architecture settings.
- TrainingConfig: Training hyperparameters and settings.
- DiaConfig: Master configuration combining all components.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading and preprocessing.

    Attributes:
        text_length: Maximum length of text sequences (must be multiple of 128).
        audio_length: Maximum length of audio sequences (must be multiple of 128).
        channels: Number of audio channels.
        text_pad_value: Value used for padding text sequences.
        audio_eos_value: Value representing the end of audio sequences.
        audio_bos_value: Value representing the beginning of audio sequences.
        audio_pad_value: Value used for padding audio sequences.
        delay_pattern: List of delay values for each audio channel.
    """

    text_length: int
    audio_length: int
    channels: int = 9
    text_pad_value: int = 0
    audio_eos_value: int = 1024
    audio_pad_value: int = 1025
    audio_bos_value: int = 1026
    delay_pattern: List[int] = field(
        default_factory=lambda: [0, 8, 9, 10, 11, 12, 13, 14, 15]
    )

    def __post_init__(self):
        # Ensure text_length and audio_length are multiples of 128
        object.__setattr__(self, "text_length", (self.text_length + 127) // 128 * 128)
        object.__setattr__(self, "audio_length", (self.audio_length + 127) // 128 * 128)

    def __hash__(self) -> int:
        """Generate a hash based on all fields of the config."""
        return hash(
            (
                self.text_length,
                self.audio_length,
                self.channels,
                self.text_pad_value,
                self.audio_pad_value,
                self.audio_bos_value,
                self.audio_eos_value,
                tuple(self.delay_pattern),
            )
        )


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the encoder component of the Dia model.

    Attributes:
        n_layer: Number of transformer layers.
        n_embd: Embedding dimension.
        n_hidden: Hidden dimension size in the MLP layers.
        n_head: Number of attention heads.
        head_dim: Dimension per attention head.
        mlp_activations: List of activation functions for the MLP layers.
        use_pre_norm: Whether to use pre-normalization (LayerNorm before attention/MLP).
    """

    n_layer: int
    n_embd: int
    n_hidden: int
    n_head: int
    head_dim: int
    mlp_activations: List[str] = field(default_factory=lambda: ["silu", "linear"])
    use_pre_norm: bool = False


@dataclass(frozen=True)
class DecoderConfig:
    """Configuration for the decoder component of the Dia model.

    Attributes:
        n_layer: Number of transformer layers.
        n_embd: Embedding dimension.
        n_hidden: Hidden dimension size in the MLP layers.
        gqa_query_heads: Number of query heads for grouped-query self-attention.
        kv_heads: Number of key/value heads for grouped-query self-attention.
        gqa_head_dim: Dimension per query head for grouped-query self-attention.
        cross_query_heads: Number of query heads for cross-attention.
        cross_head_dim: Dimension per cross-attention head.
        mlp_activations: List of activation functions for the MLP layers.
        use_pre_norm: Whether to use pre-normalization.
    """

    n_layer: int
    n_embd: int
    n_hidden: int
    gqa_query_heads: int
    kv_heads: int
    gqa_head_dim: int
    cross_query_heads: int
    cross_head_dim: int
    mlp_activations: List[str] = field(default_factory=lambda: ["silu", "linear"])
    use_pre_norm: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """Main configuration container for the Dia model architecture.

    Attributes:
        encoder: Configuration for the encoder component.
        decoder: Configuration for the decoder component.
        src_vocab_size: Size of the source (text) vocabulary.
        tgt_vocab_size: Size of the target (audio code) vocabulary.
        dropout: Dropout probability applied within the model.
        normalization_layer_epsilon: Epsilon value for normalization layers (e.g., LayerNorm).
        weight_dtype: Data type for model weights (e.g., "float32", "bfloat16").
        rope_min_timescale: Minimum timescale for Rotary Positional Embeddings (RoPE).
        rope_max_timescale: Maximum timescale for Rotary Positional Embeddings (RoPE).
    """

    encoder: EncoderConfig
    decoder: DecoderConfig
    src_vocab_size: int = 128
    tgt_vocab_size: int = 1028
    dropout: float = 0.0
    normalization_layer_epsilon: float = 1.0e-5
    weight_dtype: str = "float32"
    rope_min_timescale: int = 1
    rope_max_timescale: int = 10_000
    sample_rate: int = 44100


@dataclass(frozen=True)
class TrainingConfig:
    """Training process configuration and hyperparameters.

    Note: This configuration currently only includes precision settings.
    Other training parameters (like batch size, learning rate, optimizer settings)
    are assumed to be handled externally.

    Attributes:
        dtype: Data type for activations during training (e.g., "bfloat16", "float32").
        logits_dot_in_fp32: Whether to compute the final logits dot product in fp32 for stability.
    """

    dtype: str = "bfloat16"
    logits_dot_in_fp32: bool = False


@dataclass(frozen=True)
class DiaConfig:
    """Master configuration for the Dia model.

    Combines all sub-configurations into a single validated object.

    Attributes:
        version: Configuration version string.
        model: Model architecture configuration.
        training: Training process configuration (precision settings).
        data: Data loading and processing configuration.
    """

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    version: str = "1.0"

    def save(self, path: str) -> None:
        """Save the current configuration instance to a JSON file.

        Ensures the parent directory exists and the file has a .json extension.

        Args:
            path: The target file path to save the configuration.

        Raises:
            ValueError: If the path is not a file with a .json extension.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config_dict = {
            "version": self.version,
            "model": {
                "encoder": vars(self.model.encoder),
                "decoder": vars(self.model.decoder),
                "src_vocab_size": self.model.src_vocab_size,
                "tgt_vocab_size": self.model.tgt_vocab_size,
                "dropout": self.model.dropout,
                "normalization_layer_epsilon": self.model.normalization_layer_epsilon,
                "weight_dtype": self.model.weight_dtype,
                "rope_min_timescale": self.model.rope_min_timescale,
                "rope_max_timescale": self.model.rope_max_timescale,
                "sample_rate": self.model.sample_rate,
            },
            "training": vars(self.training),
            "data": vars(self.data),
        }
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_dict(cls, config: dict) -> Optional["DiaConfig"]:
        try:
            model_config = ModelConfig(
                encoder=EncoderConfig(**config["model"]["encoder"]),
                decoder=DecoderConfig(**config["model"]["decoder"]),
                **{
                    k: v
                    for k, v in config["model"].items()
                    if k not in ["encoder", "decoder"]
                },
            )
            return cls(
                version=config.get("version", "1.0"),
                model=model_config,
                training=TrainingConfig(**config["training"]),
                data=DataConfig(**config["data"]),
            )
        except (KeyError, TypeError):
            return None

    @classmethod
    def load(cls, path: str) -> Optional["DiaConfig"]:
        """Load and validate a Dia configuration from a JSON file.

        Args:
            path: The path to the configuration file.

        Returns:
            A validated DiaConfig instance if the file exists and is valid,
            otherwise None if the file is not found.

        Raises:
            ValueError: If the JSON content fails validation against the DiaConfig schema.
        """
        try:
            with open(path, "r") as f:
                config = json.load(f)
            return cls.load_dict(config)
        except FileNotFoundError:
            return None
