from dataclasses import dataclass, field
from typing import List, Optional, Union

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class ModelConfig(BaseModelArgs):
    # Llama backbone
    vocab_size: int = 128256
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = True

    # TADA-specific
    acoustic_dim: int = 512
    num_time_classes: int = 256
    shift_acoustic: int = 5
    head_layers: int = 6
    head_ffn_ratio: float = 4.0
    bottleneck_dim: Optional[int] = None
    context_window: int = 8
    acoustic_mean: float = 0.0
    acoustic_std: float = 1.5
    diffusion_head_type: str = "vibevoice"

    # Decoder config
    decoder_hidden_dim: int = 1024
    decoder_d_model: int = 96
    decoder_embed_dim: int = 512
    decoder_strides: List[int] = field(default_factory=lambda: [4, 4, 5, 6])
    decoder_num_attn_layers: int = 6
    decoder_num_attn_heads: int = 8
    decoder_attn_dim_feedforward: int = 4096
    decoder_block_attention: str = "v2"

    # Encoder config
    encoder_hidden_dim: int = 1024
    encoder_d_model: int = 96
    encoder_embed_dim: int = 512
    encoder_strides: List[int] = field(default_factory=lambda: [6, 5, 4, 4])
    encoder_num_attn_layers: int = 6
    encoder_num_attn_heads: int = 8
    encoder_attn_dim_feedforward: int = 4096
    encoder_block_attention: str = "v2"
    encoder_std: float = 0.5

    # Audio
    sample_rate: int = 24000
    model_type: str = "tada"

    # EOS tokens
    eos_token_id: Union[int, List[int]] = 128001

    # Rope scaling (ignored for inference within max_position_embeddings)
    rope_scaling: Optional[dict] = None
    # Allow extra fields from HF config
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    mlp_bias: bool = False
    pretraining_tp: int = 1
    use_cache: bool = True
