import inspect
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EncoderConfig:
    vocab_size: int = 16384
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 16
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    num_mel_bins: int = 80
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 512
    context_size: int = 128
    conv_kernel_size: int = 7
    conv_expansion_factor: int = 2
    subsample_layers: List[int] = field(default_factory=lambda: [0, 1])
    attention_bias: bool = True
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    initializer_range: float = 0.02
    model_type: str = "granite_speech5_encoder"

    @classmethod
    def from_dict(cls, params):
        values = {
            key: value
            for key, value in params.items()
            if key in inspect.signature(cls).parameters
        }
        if values.get("head_dim") is None:
            hidden_size = values.get("hidden_size", cls.hidden_size)
            num_heads = values.get("num_attention_heads", cls.num_attention_heads)
            values["head_dim"] = hidden_size // num_heads
        if values.get("num_key_value_heads") is None:
            values["num_key_value_heads"] = values.get(
                "num_attention_heads", cls.num_attention_heads
            )
        return cls(**values)


@dataclass
class ModelConfig:
    model_type: str = "granite_speech5_ctc"
    vocab_size: int = 16384
    encoder_config: Optional[EncoderConfig] = None
    pad_token_id: int = 0
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = EncoderConfig.from_dict(self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = EncoderConfig(vocab_size=self.vocab_size)

        if self.encoder_config.vocab_size != self.vocab_size:
            raise ValueError(
                "encoder_config.vocab_size must match the top-level vocab_size"
            )

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                key: value
                for key, value in params.items()
                if key in inspect.signature(cls).parameters
            }
        )
