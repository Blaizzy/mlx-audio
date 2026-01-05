"""CosyVoice3 Configuration"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mlx_audio.tts.models.base import BaseModelArgs


@dataclass
class DiTConfig(BaseModelArgs):
    """DiT (Diffusion Transformer) configuration"""

    dim: int = 1024
    depth: int = 22
    heads: int = 16
    dim_head: int = 64
    ff_mult: int = 2
    dropout: float = 0.0
    mel_dim: int = 80
    mu_dim: int = 80
    spk_dim: int = 80
    out_channels: int = 80
    static_chunk_size: int = 50
    num_decoding_left_chunks: int = -1


@dataclass
class FlowConfig(BaseModelArgs):
    """Flow module configuration"""

    input_size: int = 80
    output_size: int = 80
    spk_embed_dim: int = 192
    output_type: str = "mel"
    vocab_size: int = 6561
    input_frame_rate: int = 25
    token_mel_ratio: int = 2
    pre_lookahead_len: int = 3
    sigma_min: float = 1e-06
    solver: str = "euler"
    t_scheduler: str = "cosine"
    training_cfg_rate: float = 0.2
    inference_cfg_rate: float = 0.7
    dit: DiTConfig = field(default_factory=DiTConfig)


@dataclass
class HIFTConfig(BaseModelArgs):
    """HIFT (HiFi Transformer) vocoder configuration"""

    in_channels: int = 80
    base_channels: int = 512
    nb_harmonics: int = 8
    sampling_rate: int = 24000
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: float = 10
    upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 3])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 11, 7])
    n_fft: int = 16
    hop_len: int = 4
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    source_resblock_kernel_sizes: List[int] = field(default_factory=lambda: [7, 7, 11])
    source_resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    lrelu_slope: float = 0.1
    audio_limit: float = 0.99


@dataclass
class LLMConfig(BaseModelArgs):
    """LLM configuration (based on Qwen2)"""

    llm_input_size: int = 896
    llm_output_size: int = 896
    speech_token_size: int = 6561
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    vocab_size: int = 151936


@dataclass
class CosyVoice3Config(BaseModelArgs):
    """Main CosyVoice3 model configuration"""

    sample_rate: int = 24000
    token_frame_rate: int = 25
    token_mel_ratio: int = 2
    chunk_size: int = 25
    spk_embed_dim: int = 192

    llm: LLMConfig = field(default_factory=LLMConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    hift: HIFTConfig = field(default_factory=HIFTConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CosyVoice3Config":
        """Load config from YAML file"""
        import yaml

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            sample_rate=config_dict.get("sample_rate", 24000),
            token_frame_rate=config_dict.get("token_frame_rate", 25),
            token_mel_ratio=config_dict.get("token_mel_ratio", 2),
            chunk_size=config_dict.get("chunk_size", 25),
            spk_embed_dim=config_dict.get("spk_embed_dim", 192),
        )
