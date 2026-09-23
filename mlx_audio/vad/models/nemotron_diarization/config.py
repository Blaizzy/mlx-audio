"""Configuration for NVIDIA Nemotron 3 Diarization."""

from dataclasses import dataclass, field

from mlx_audio.base import BaseModelArgs
from mlx_audio.vad.models.sortformer.config import ModulesConfig, ProcessorConfig


@dataclass
class EncoderConfig(BaseModelArgs):
    feat_in: int = 128
    d_model: int = 512
    n_layers: int = 31
    n_heads: int = 8
    subsampling_factor: int = 8
    ff_expansion: float = 4.0
    qkv_bias: bool = False
    qk_norm: bool = False
    pre_block_norm: bool = True
    xscaling: bool = False
    rope_base: float = 10000.0
    rotary_fraction: float = 1.0


@dataclass
class DiarizationModulesConfig(ModulesConfig):
    num_speakers: int = 8
    chunk_len: int = 340
    fifo_len: int = 40
    spkcache_len: int = 264
    spkcache_update_period: int = 300
    chunk_left_context: int = 0
    chunk_right_context: int = 40
    spkcache_sil_frames_per_spk: int = 1
    pred_score_threshold: float = 0.25
    max_index: int = 99999
    scores_boost_latest: float = 0.05
    sil_threshold: float = 0.2
    strong_boost_rate: float = 0.75
    weak_boost_rate: float = 1.5
    use_aosc: bool = True
    use_learnable_sil_emb: bool = True
    use_activity_head: bool = True


@dataclass
class MelConfig(ProcessorConfig):
    feature_size: int = 128
    pad_to: int = 16


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "nemotron_diarization"
    num_speakers: int = 8
    dtype: str = "float32"
    output_subsampling_factor: int = 1
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    modules_config: DiarizationModulesConfig = field(
        default_factory=DiarizationModulesConfig
    )
    processor_config: MelConfig = field(default_factory=MelConfig)

    def __post_init__(self):
        for name, cls in (
            ("encoder_config", EncoderConfig),
            ("modules_config", DiarizationModulesConfig),
            ("processor_config", MelConfig),
        ):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, cls.from_dict(value))
        enc, mod = self.encoder_config, self.modules_config
        if enc.d_model % enc.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        rotary_dim = int(enc.d_model // enc.n_heads * enc.rotary_fraction)
        if rotary_dim < 2 or rotary_dim % 2 or not 0 < enc.rotary_fraction <= 1:
            raise ValueError(
                "rotary_fraction must select an even, positive head dimension"
            )
        if enc.feat_in != self.processor_config.feature_size:
            raise ValueError("Encoder and preprocessor feature dimensions must match")
        if enc.d_model != mod.fc_d_model or self.num_speakers != mod.num_speakers:
            raise ValueError("Encoder and speaker head dimensions must match")
        if enc.subsampling_factor != mod.subsampling_factor:
            raise ValueError("Encoder and speaker head subsampling factors must match")
        for name in ("chunk_len", "spkcache_update_period", "spkcache_len"):
            if type(getattr(mod, name)) is not int or getattr(mod, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("fifo_len", "chunk_right_context", "spkcache_sil_frames_per_spk"):
            if type(getattr(mod, name)) is not int or getattr(mod, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if mod.chunk_left_context != 0:
            raise ValueError("Feature stacking requires chunk_left_context=0")
        if mod.spkcache_len < (1 + mod.spkcache_sil_frames_per_spk) * self.num_speakers:
            raise ValueError("Speaker cache is too small for the number of speakers")
        factor = self.output_subsampling_factor
        if type(factor) is not int or factor < 1:
            raise ValueError("output_subsampling_factor must be a positive integer")
        if mod.chunk_len * enc.subsampling_factor % factor:
            raise ValueError(
                "Output stride must divide the chunk's feature frame count"
            )
