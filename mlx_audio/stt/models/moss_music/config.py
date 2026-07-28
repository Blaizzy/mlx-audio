from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mlx_audio.base import BaseModelArgs


@dataclass
class AudioEncoderConfig(BaseModelArgs):
    d_model: int = 1280
    output_dim: int = 1280
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    downsample_rate: int = 8
    downsample_hidden_size: int = 480
    encoder_attention_window_size: int = 100
    max_source_positions: int = 1500
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    layer_norm_eps: float = 1e-5
    pretrained_path: str = ""
    n_window: int = 200
    conv_chunksize: int = 64
    deepstack_encoder_layer_indexes: List[int] = field(
        default_factory=lambda: [8, 16, 24]
    )
    model_type: str = "moss_music_audio_encoder"


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str = "qwen3"
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    use_cache: bool = True


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "moss_music"
    audio_config: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    language_config: TextConfig = field(default_factory=TextConfig)
    adapter_hidden_size: int = 8192
    deepstack_num_inject_layers: int = 3
    ignore_index: int = -100
    dtype: str = "bfloat16"
    model_path: Optional[str] = None
    sample_rate: int = 16000
    audio_token_id: int = 151654
    audio_start_id: int = 151669
    audio_end_id: int = 151670
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    enable_time_marker: bool = True
    strip_thinking: bool = True
    default_prompt: str = "Please give a detailed musical description of this clip."

    def __post_init__(self):
        if isinstance(self.audio_config, dict):
            self.audio_config = AudioEncoderConfig.from_dict(self.audio_config)
        if isinstance(self.language_config, dict):
            self.language_config = TextConfig.from_dict(self.language_config)
