"""Configuration management module for the Dia model.

This module provides comprehensive configuration management for the Dia model,
utilizing Pydantic for validation. It defines configurations for data processing,
model architecture (encoder and decoder), and training settings.

Key components:
- DataConfig: Parameters for data loading and preprocessing.
- EncoderConfig: Architecture details for the encoder module.
- DecoderConfig: Architecture details for the decoder module.
- ModelConfig: Combined model architecture settings.
- TrainingConfig: Training hyperparameters and settings.
- DiaConfig: Master configuration combining all components.
"""

import os
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field


class DataConfig(BaseModel, frozen=True):
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

    text_length: Annotated[int, BeforeValidator(lambda x: (x + 127) // 128 * 128)] = Field(gt=0, multiple_of=128)
    audio_length: Annotated[int, BeforeValidator(lambda x: (x + 127) // 128 * 128)] = Field(gt=0, multiple_of=128)
    channels: int = Field(default=9, gt=0, multiple_of=1)
    text_pad_value: int = Field(default=0)
    audio_eos_value: int = Field(default=1024)
    audio_pad_value: int = Field(default=1025)
    audio_bos_value: int = Field(default=1026)
    delay_pattern: list[Annotated[int, Field(ge=0)]] = Field(default_factory=lambda: [0, 8, 9, 10, 11, 12, 13, 14, 15])

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


class EncoderConfig(BaseModel, frozen=True):
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

    n_layer: int = Field(gt=0)
    n_embd: int = Field(gt=0)
    n_hidden: int = Field(gt=0)
    n_head: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    mlp_activations: list[str] = Field(default=["silu", "linear"])
    use_pre_norm: bool = Field(default=False)


class DecoderConfig(BaseModel, frozen=True):
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

    n_layer: int = Field(gt=0)
    n_embd: int = Field(gt=0)
    n_hidden: int = Field(gt=0)
    gqa_query_heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    gqa_head_dim: int = Field(gt=0)
    cross_query_heads: int = Field(gt=0)
    cross_head_dim: int = Field(gt=0)
    mlp_activations: list[str] = Field(default=["silu", "linear"])
    use_pre_norm: bool = Field(default=False)


class ModelConfig(BaseModel, frozen=True):
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
    src_vocab_size: int = Field(default=128, gt=0)
    tgt_vocab_size: int = Field(default=1028, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    normalization_layer_epsilon: float = Field(default=1.0e-5, ge=0.0)
    weight_dtype: str = Field(default="float32", description="Weight precision")
    rope_min_timescale: int = Field(default=1, description="Timescale For global Attention")
    rope_max_timescale: int = Field(default=10_000, description="Timescale For global Attention")


class TrainingConfig(BaseModel, frozen=True):
    """Training process configuration and hyperparameters.

    Note: This configuration currently only includes precision settings.
    Other training parameters (like batch size, learning rate, optimizer settings)
    are assumed to be handled externally.

    Attributes:
        dtype: Data type for activations during training (e.g., "bfloat16", "float32").
        logits_dot_in_fp32: Whether to compute the final logits dot product in fp32 for stability.
    """

    dtype: str = Field(default="bfloat16", description="Activation precision")
    logits_dot_in_fp32: bool = Field(default=False)


class DiaConfig(BaseModel, frozen=True):
    """Master configuration for the Dia model.

    Combines all sub-configurations into a single validated object.

    Attributes:
        version: Configuration version string.
        model: Model architecture configuration.
        training: Training process configuration (precision settings).
        data: Data loading and processing configuration.
    """

    version: str = Field(default="1.0")
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    def save(self, path: str) -> None:
        """Save the current configuration instance to a JSON file.

        Ensures the parent directory exists and the file has a .json extension.

        Args:
            path: The target file path to save the configuration.

        Raises:
            ValueError: If the path is not a file with a .json extension.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config_json = self.model_dump_json(indent=2)
        with open(path, "w") as f:
            f.write(config_json)

    @classmethod
    def load(cls, path: str) -> "DiaConfig | None":
        """Load and validate a Dia configuration from a JSON file.

        Args:
            path: The path to the configuration file.

        Returns:
            A validated DiaConfig instance if the file exists and is valid,
            otherwise None if the file is not found.

        Raises:
            ValueError: If the path does not point to an existing .json file.
            pydantic.ValidationError: If the JSON content fails validation against the DiaConfig schema.
        """
        try:
            with open(path, "r") as f:
                content = f.read()
            return cls.model_validate_json(content)
        except FileNotFoundError:
            return None
