"""Configuration for the exported two-speaker DialogueSidon model."""

from dataclasses import dataclass, field, fields


@dataclass
class EncoderConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 13
    num_attention_heads: int = 16
    feature_projection_input_dim: int = 160
    conv_depthwise_kernel_size: int = 31
    left_max_position_embeddings: int = 64
    right_max_position_embeddings: int = 8
    layer_norm_eps: float = 1e-5


@dataclass
class DiffusionConfig:
    hidden_size: int = 768
    num_layers: int = 8
    num_heads: int = 12
    ffn_ratio: float = 4.0
    frequency_embedding_size: int = 256
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "v_prediction"


@dataclass
class ModelConfig:
    model_type: str = "dialogue_sidon"
    sample_rate: int = 24000
    latent_dim: int = 32
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    decoder_channels: int = 1536
    decoder_rates: list[int] = field(default_factory=lambda: [8, 5, 4, 3])
    latent_norm_initialized: bool = False
    latent_norm_mean: list[float] = field(default_factory=list)
    latent_norm_std: list[float] = field(default_factory=list)
    source_repo: str = "sarulab-speech/DialogueSidon"
    source_revision: str = ""

    def __post_init__(self):
        if self.encoder.hidden_size % self.encoder.num_attention_heads:
            raise ValueError("Encoder hidden size must be divisible by its head count")
        head_dim = self.diffusion.hidden_size // self.diffusion.num_heads
        if self.diffusion.hidden_size % self.diffusion.num_heads or head_dim % 2:
            raise ValueError("Diffusion head dimension must be an even integer")
        if self.latent_norm_initialized:
            if (
                len(self.latent_norm_mean) != self.latent_dim * 2
                or len(self.latent_norm_std) != self.latent_dim * 2
            ):
                raise ValueError("Expected normalization statistics for both speakers")
            import math

            if not all(math.isfinite(x) for x in self.latent_norm_mean) or not all(
                math.isfinite(x) and x > 0 for x in self.latent_norm_std
            ):
                raise ValueError("Latent statistics must be finite with positive std")

    @classmethod
    def from_dict(cls, config):
        config = {k: v for k, v in config.items() if k in {f.name for f in fields(cls)}}
        for key, kind in (("encoder", EncoderConfig), ("diffusion", DiffusionConfig)):
            if isinstance(config.get(key), dict):
                config[key] = kind(**config[key])
        return cls(**config)
