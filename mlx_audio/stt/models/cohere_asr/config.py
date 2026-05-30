import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _filter_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if k in inspect.signature(cls).parameters}


@dataclass
class PreprocessorConfig:
    sample_rate: int = 16000
    normalize: str = "per_feature"
    features: int = 128
    n_fft: int = 512
    window_size: float = 0.025
    window_stride: float = 0.01
    window: str = "hann"
    dither: float = 1e-5
    pad_to: int = 0
    pad_value: float = 0.0
    preemph: float = 0.97
    log: bool = True
    log_zero_guard_value: float = 2**-24

    @property
    def win_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "PreprocessorConfig":
        return cls(**_filter_params(cls, params))


@dataclass
class EncoderConfig:
    feat_in: int = 128
    feat_out: int = -1
    n_layers: int = 48
    d_model: int = 1280
    n_heads: int = 8
    ff_expansion_factor: int = 4
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    pos_emb_max_len: int = 5000
    self_attention_model: str = "rel_pos"
    subsampling: str = "dw_striding"
    causal_downsampling: bool = False
    xscaling: bool = False

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "EncoderConfig":
        return cls(**_filter_params(cls, params))


@dataclass
class HeadConfig:
    hidden_size: int = 1024
    num_classes: int = 16384
    log_softmax: bool = True

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "HeadConfig":
        return cls(**_filter_params(cls, params))


@dataclass
class DecoderInnerConfig:
    hidden_size: int = 1024
    inner_size: int = 4096
    num_attention_heads: int = 8
    num_layers: int = 8
    hidden_act: str = "relu"
    max_sequence_length: int = 1024

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "DecoderInnerConfig":
        return cls(**_filter_params(cls, params))


@dataclass
class DecoderConfig:
    config_dict: DecoderInnerConfig = None

    def __post_init__(self):
        if self.config_dict is None:
            self.config_dict = DecoderInnerConfig()
        elif isinstance(self.config_dict, dict):
            self.config_dict = DecoderInnerConfig.from_dict(self.config_dict)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "DecoderConfig":
        params = params.copy()
        if "config_dict" in params and isinstance(params["config_dict"], dict):
            params["config_dict"] = DecoderInnerConfig.from_dict(params["config_dict"])
        return cls(**_filter_params(cls, params))


@dataclass
class ModelConfig:
    model_type: str = "cohere_asr"
    vocab_size: int = 16384
    encoder: EncoderConfig = None
    transf_decoder: DecoderConfig = None
    head: HeadConfig = None
    preprocessor: PreprocessorConfig = None
    max_audio_clip_s: float = 35.0
    overlap_chunk_second: float = 5.0
    min_energy_window_samples: int = 1600
    batch_size: int = 64
    max_safe_bytes: int = 10 * 1024 * 1024 * 1024
    sample_rate: int = 16000
    supported_languages: List[str] = field(
        default_factory=lambda: [
            "en",
            "fr",
            "de",
            "es",
            "it",
            "pt",
            "nl",
            "pl",
            "el",
            "ar",
            "ja",
            "zh",
            "vi",
            "ko",
        ]
    )

    def __post_init__(self):
        if self.encoder is None:
            self.encoder = EncoderConfig()
        elif isinstance(self.encoder, dict):
            self.encoder = EncoderConfig.from_dict(self.encoder)

        if self.transf_decoder is None:
            self.transf_decoder = DecoderConfig()
        elif isinstance(self.transf_decoder, dict):
            self.transf_decoder = DecoderConfig.from_dict(self.transf_decoder)

        if self.head is None:
            self.head = HeadConfig()
        elif isinstance(self.head, dict):
            self.head = HeadConfig.from_dict(self.head)

        if self.preprocessor is None:
            self.preprocessor = PreprocessorConfig()
        elif isinstance(self.preprocessor, dict):
            self.preprocessor = PreprocessorConfig.from_dict(self.preprocessor)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        params = params.copy()

        if "encoder" in params and isinstance(params["encoder"], dict):
            params["encoder"] = EncoderConfig.from_dict(params["encoder"])
        if "transf_decoder" in params and isinstance(params["transf_decoder"], dict):
            params["transf_decoder"] = DecoderConfig.from_dict(params["transf_decoder"])
        if "head" in params and isinstance(params["head"], dict):
            params["head"] = HeadConfig.from_dict(params["head"])
        if "preprocessor" in params and isinstance(params["preprocessor"], dict):
            params["preprocessor"] = PreprocessorConfig.from_dict(
                params["preprocessor"]
            )

        return cls(**_filter_params(cls, params))
