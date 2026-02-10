import inspect
from dataclasses import dataclass, field
from typing import Optional

from mlx_audio.base import BaseModelArgs


@dataclass
class FCEncoderConfig(BaseModelArgs):
    """FastConformer encoder configuration."""

    model_type: str = "sortformer_fc_encoder"
    hidden_size: int = 512
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    intermediate_size: int = 2048
    hidden_act: str = "silu"
    num_mel_bins: int = 80
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    max_position_embeddings: int = 5000
    attention_bias: bool = True
    scale_input: bool = True
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    dropout_positions: float = 0.0
    layerdrop: float = 0.1
    initializer_range: float = 0.02


@dataclass
class TFEncoderConfig(BaseModelArgs):
    """Transformer encoder configuration."""

    model_type: str = "sortformer_tf_encoder"
    d_model: int = 192
    encoder_layers: int = 18
    encoder_attention_heads: int = 8
    encoder_ffn_dim: int = 768
    activation_function: str = "relu"
    dropout: float = 0.5
    attention_dropout: float = 0.5
    activation_dropout: float = 0.5
    encoder_layerdrop: float = 0.5
    layer_norm_eps: float = 1e-5
    max_source_positions: int = 1500
    scale_embedding: bool = False
    init_std: float = 0.02
    initializer_range: float = 0.02
    num_mel_bins: int = 80
    k_proj_bias: bool = False


@dataclass
class ModulesConfig(BaseModelArgs):
    """Sortformer modules configuration."""

    model_type: str = "sortformer_modules"
    num_speakers: int = 4
    fc_d_model: int = 512
    tf_d_model: int = 192
    dropout_rate: float = 0.5
    subsampling_factor: int = 8
    chunk_len: int = 188
    fifo_len: int = 0
    spkcache_len: int = 188
    spkcache_update_period: int = 188
    chunk_left_context: int = 1
    chunk_right_context: int = 1
    spkcache_sil_frames_per_spk: int = 5
    causal_attn_rate: float = 0.5
    causal_attn_rc: int = 30
    scores_add_rnd: float = 2.0
    pred_score_threshold: float = 1e-6
    max_index: int = 10000
    scores_boost_latest: float = 0.5
    sil_threshold: float = 0.1
    strong_boost_rate: float = 0.3
    weak_boost_rate: float = 0.7
    min_pos_scores_rate: float = 0.5
    use_aosc: bool = False


@dataclass
class ProcessorConfig(BaseModelArgs):
    """Feature extractor configuration."""

    feature_size: int = 80
    sampling_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 512
    win_length: int = 400
    preemphasis: float = 0.97
    padding_value: float = 0.0


@dataclass
class ModelConfig(BaseModelArgs):
    """Sortformer diarization model configuration."""

    model_type: str = "sortformer"
    num_speakers: int = 4
    ats_weight: float = 0.5
    pil_weight: float = 0.5
    dtype: str = "float32"
    initializer_range: float = 0.02
    fc_encoder_config: Optional[FCEncoderConfig] = None
    tf_encoder_config: Optional[TFEncoderConfig] = None
    modules_config: Optional[ModulesConfig] = None
    processor_config: Optional[ProcessorConfig] = None

    def __post_init__(self):
        if isinstance(self.fc_encoder_config, dict):
            self.fc_encoder_config = FCEncoderConfig.from_dict(self.fc_encoder_config)
        if self.fc_encoder_config is None:
            self.fc_encoder_config = FCEncoderConfig()

        if isinstance(self.tf_encoder_config, dict):
            self.tf_encoder_config = TFEncoderConfig.from_dict(self.tf_encoder_config)
        if self.tf_encoder_config is None:
            self.tf_encoder_config = TFEncoderConfig()

        if isinstance(self.modules_config, dict):
            self.modules_config = ModulesConfig.from_dict(self.modules_config)
        if self.modules_config is None:
            self.modules_config = ModulesConfig()

        if isinstance(self.processor_config, dict):
            self.processor_config = ProcessorConfig.from_dict(self.processor_config)
        if self.processor_config is None:
            self.processor_config = ProcessorConfig()

    @classmethod
    def from_dict(cls, params):
        params = params.copy()
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
