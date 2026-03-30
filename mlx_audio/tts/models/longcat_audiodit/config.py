import math
from dataclasses import dataclass, field
from typing import List, Optional

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class VaeConfig:
    in_channels: int = 1
    channels: int = 128
    c_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    strides: List[int] = field(default_factory=lambda: [2, 4, 4, 8, 8])
    latent_dim: int = 64
    encoder_latent_dim: int = 128
    use_snake: bool = True
    downsample_shortcut: str = "averaging"
    upsample_shortcut: str = "duplicating"
    out_shortcut: str = "averaging"
    in_shortcut: str = "duplicating"
    final_tanh: bool = False
    downsampling_ratio: int = 2048
    sample_rate: int = 24000
    scale: float = 0.71


@dataclass
class TextEncoderConfig:
    vocab_size: int = 256384
    d_model: int = 768
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 12
    num_heads: int = 12
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    is_gated_act: bool = True
    dense_act_fn: str = "gelu_new"


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "audiodit"
    dit_dim: int = 1536
    dit_depth: int = 24
    dit_heads: int = 24
    dit_ff_mult: float = 4.0
    dit_text_dim: int = 768
    dit_dropout: float = 0.0
    dit_bias: bool = True
    dit_cross_attn: bool = True
    dit_adaln_type: str = "global"
    dit_adaln_use_text_cond: bool = True
    dit_long_skip: bool = True
    dit_text_conv: bool = True
    dit_qk_norm: bool = True
    dit_cross_attn_norm: bool = False
    dit_eps: float = 1e-6
    dit_use_latent_condition: bool = True
    repa_dit_layer: int = 8
    latent_dim: int = 64
    sigma: float = 0.0
    sampling_rate: int = 24000
    latent_hop: int = 2048
    max_wav_duration: float = 30.0
    text_encoder_model: str = "google/umt5-base"
    text_add_embed: bool = True
    text_norm_feat: bool = True
    vae_config: Optional[VaeConfig] = None
    text_encoder_config: Optional[TextEncoderConfig] = None

    def __post_init__(self):
        if isinstance(self.vae_config, dict):
            self.vae_config = VaeConfig(
                **{
                    k: v
                    for k, v in self.vae_config.items()
                    if k in VaeConfig.__dataclass_fields__
                }
            )
        if self.vae_config is None:
            self.vae_config = VaeConfig()
        if isinstance(self.text_encoder_config, dict):
            self.text_encoder_config = TextEncoderConfig(
                **{
                    k: v
                    for k, v in self.text_encoder_config.items()
                    if k in TextEncoderConfig.__dataclass_fields__
                }
            )
        if self.text_encoder_config is None:
            self.text_encoder_config = TextEncoderConfig()
