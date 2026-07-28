from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from mlx_audio.base import BaseModelArgs


@dataclass
class AudioEncoderConfig(BaseModelArgs):
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    d_model: int = 1280
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    scale_embedding: bool = False
    max_source_positions: int = 1500
    frame_rate: int = 25
    model_type: str = "higgs_audio_encoder"


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str = "qwen3"
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True


@dataclass
class ModelConfig(BaseModelArgs):
    audio_encoder_config: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    text_config: TextConfig = field(default_factory=TextConfig)
    model_type: str = "higgs_audio_3"
    model_path: Optional[str] = None
    audio_adapter_type: str = "stack"
    projector_type: str = "mlp"
    projector_temporal_downsample: int = 2
    audio_num_codebooks: int = 1
    audio_codebook_size: int = 1
    audio_in_token_idx: int = 151672
    audio_out_token_idx: int = 151673
    audio_bos_token_id: int = 151669
    audio_eos_token_id: int = 151670
    chunk_size_seconds: float = 4.0
    pad_token_id: int = 151643
    sample_rate: int = 16000
    vad_cut: bool = True
    split_vads: bool = False

    def __post_init__(self):
        if isinstance(self.audio_encoder_config, dict):
            self.audio_encoder_config = AudioEncoderConfig.from_dict(
                self.audio_encoder_config
            )
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
