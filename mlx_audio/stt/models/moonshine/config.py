import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    model_type: str = "moonshine"
    vocab_size: int = 32768
    hidden_size: int = 288
    intermediate_size: int = 1152
    encoder_num_hidden_layers: int = 6
    decoder_num_hidden_layers: int = 6
    encoder_num_attention_heads: int = 8
    decoder_num_attention_heads: int = 8
    encoder_num_key_value_heads: Optional[int] = None
    decoder_num_key_value_heads: Optional[int] = None
    encoder_hidden_act: str = "gelu"
    decoder_hidden_act: str = "silu"
    max_position_embeddings: int = 512
    attention_bias: bool = False
    attention_dropout: float = 0.0
    partial_rotary_factor: float = 0.9
    rope_theta: float = 10000.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    decoder_start_token_id: int = 1
    tie_word_embeddings: bool = True
    pad_head_dim_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.encoder_num_key_value_heads is None:
            self.encoder_num_key_value_heads = self.encoder_num_attention_heads
        if self.decoder_num_key_value_heads is None:
            self.decoder_num_key_value_heads = self.decoder_num_attention_heads

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
