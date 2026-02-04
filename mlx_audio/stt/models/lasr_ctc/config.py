from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class LasrEncoderConfig:
    """Configuration for the LasrEncoder."""

    hidden_size: int = 512
    num_hidden_layers: int = 17
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    intermediate_size: int = 2048
    hidden_act: str = "silu"

    # Convolution
    conv_kernel_size: int = 32
    convolution_bias: bool = False

    # Subsampling
    num_mel_bins: int = 128
    subsampling_conv_channels: int = 256
    subsampling_conv_kernel_size: int = 5
    subsampling_conv_stride: int = 2

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    dropout_positions: float = 0.0
    layerdrop: float = 0.1

    # Normalization
    layer_norm_eps: float = 1e-06
    batch_norm_momentum: float = 0.01

    # Initializers
    initializer_range: float = 0.02

    # Positional embeddings
    max_position_embeddings: int = 10000
    attention_bias: bool = False

    # RoPE
    rope_theta: float = 10000.0
    rope_type: str = "default"

    # Residual scaling
    conv_residual_weights: List[float] = None
    feed_forward_residual_weights: List[float] = None

    def __post_init__(self):
        if self.conv_residual_weights is None:
            self.conv_residual_weights = [2.0, 1.0]
        if self.feed_forward_residual_weights is None:
            self.feed_forward_residual_weights = [1.5, 0.5]

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "LasrEncoderConfig":
        rope_params = params.get("rope_parameters", {})

        # Filter valid fields
        import inspect

        valid_params = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }

        config = cls(**valid_params)

        # Handle rope_params if they were nested
        if rope_params:
            if "rope_theta" in rope_params:
                config.rope_theta = rope_params["rope_theta"]
            if "rope_type" in rope_params:
                config.rope_type = rope_params["rope_type"]

        return config


@dataclass
class ModelConfig:  # To keep consistent naming with other mlx-audio models, although usually it is model specific
    """Configuration for the LasrCTC model."""

    vocab_size: int = 512
    encoder_config: LasrEncoderConfig = None
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    pad_token_id: int = 0
    initializer_range: float = 0.02
    model_type: str = "lasr"

    def __post_init__(self):
        if self.encoder_config is None:
            self.encoder_config = LasrEncoderConfig()
        elif isinstance(self.encoder_config, dict):
            self.encoder_config = LasrEncoderConfig.from_dict(self.encoder_config)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        encoder_config = params.pop("encoder_config", None)

        # Filter valid fields
        import inspect

        valid_params = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }

        config = cls(**valid_params)

        if encoder_config is not None:
            if isinstance(encoder_config, dict):
                config.encoder_config = LasrEncoderConfig.from_dict(encoder_config)
            else:
                config.encoder_config = encoder_config

        return config
