from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

@dataclass
class LMConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 4096
    vocab_size: int = 73448
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling_type: str = "longrope"
    rope_long_factor: List[float] = field(default_factory=list)
    rope_short_factor: List[float] = field(default_factory=list)
    scale_emb: int = 12
    dim_model_base: int = 256
    scale_depth: float = 1.4
    original_max_position_embeddings: int = 32768
    max_position_embeddings: int = 32768
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_mup: bool = False

@dataclass
class EncoderConfig:
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 8

@dataclass
class CFMConfig:
    sigma_min: float = 1e-6
    solver: Literal["euler"] = "euler"
    t_scheduler: str = "log-norm"
    inference_cfg_rate: float = 2.0

@dataclass  
class DiTConfig:
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 8
    cfm_config: CFMConfig = field(default_factory=CFMConfig)

@dataclass
class AudioVAEConfig:
    encoder_dim: int = 64
    encoder_rates: List[int] = field(default_factory=lambda: [2, 3, 6, 7, 7])
    latent_dim: int = 64
    decoder_dim: int = 2048
    decoder_rates: List[int] = field(default_factory=lambda: [7, 7, 6, 3, 2])
    sample_rate: int = 44100

@dataclass
class ModelArgs:
    lm_config: LMConfig = field(default_factory=LMConfig)
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    audio_vae_config: AudioVAEConfig = field(default_factory=AudioVAEConfig)
    patch_size: int = 4
    feat_dim: int = 64
    scalar_quantization_latent_dim: int = 256
    scalar_quantization_scale: int = 9
    residual_lm_num_layers: int = 8
    max_length: int = 8192
    model_path: Optional[str] = None # Added for auxiliary file loading

    @classmethod
    def from_dict(cls, config: dict):
        lm_cfg = config.get("lm_config", {})
        if "rope_scaling" in lm_cfg:
            rs = lm_cfg["rope_scaling"]
            lm_cfg["rope_scaling_type"] = rs.get("type", "longrope")
            lm_cfg["rope_long_factor"] = rs.get("long_factor", [])
            lm_cfg["rope_short_factor"] = rs.get("short_factor", [])
            lm_cfg["original_max_position_embeddings"] = rs.get("original_max_position_embeddings", 32768)
            del lm_cfg["rope_scaling"]

        dit_cfg = config.get("dit_config", {})
        cfm_cfg = dit_cfg.get("cfm_config", {})
        dit_cfg["cfm_config"] = CFMConfig(**cfm_cfg)
        
        return cls(
            lm_config=LMConfig(**lm_cfg),
            encoder_config=EncoderConfig(**config.get("encoder_config", {})),
            dit_config=DiTConfig(**dit_cfg),
            audio_vae_config=AudioVAEConfig(**config.get("audio_vae_config", {})),
            patch_size=config.get("patch_size", 4),
            feat_dim=config.get("feat_dim", 64),
            scalar_quantization_latent_dim=config.get("scalar_quantization_latent_dim", 256),
            scalar_quantization_scale=config.get("scalar_quantization_scale", 9),
            residual_lm_num_layers=config.get("residual_lm_num_layers", 8),
            max_length=config.get("max_length", 8192)
        )
