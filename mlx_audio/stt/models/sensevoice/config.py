from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EncoderConfig:
    output_size: int = 512
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 50
    tp_blocks: int = 20
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    kernel_size: int = 11
    sanm_shift: int = 0
    normalize_before: bool = True

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "EncoderConfig":
        import inspect

        valid = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }
        # Handle the typo in upstream config ("sanm_shfit" -> "sanm_shift")
        if "sanm_shfit" in params and "sanm_shift" not in params:
            valid["sanm_shift"] = params["sanm_shfit"]
        return cls(**valid)


@dataclass
class FrontendConfig:
    fs: int = 16000
    window: str = "hamming"
    n_mels: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    lfr_m: int = 7
    lfr_n: int = 6

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "FrontendConfig":
        import inspect

        valid = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }
        return cls(**valid)


@dataclass
class ModelConfig:
    model_type: str = "sensevoice"
    vocab_size: int = 25055
    input_size: int = 560
    encoder_conf: Optional[EncoderConfig] = None
    frontend_conf: Optional[FrontendConfig] = None
    # CMVN stats (loaded from am.mvn)
    cmvn_means: Optional[List[float]] = None
    cmvn_istd: Optional[List[float]] = None
    # Path for loading ancillary files
    model_path: Optional[str] = None

    def __post_init__(self):
        if self.encoder_conf is None:
            self.encoder_conf = EncoderConfig()
        elif isinstance(self.encoder_conf, dict):
            self.encoder_conf = EncoderConfig.from_dict(self.encoder_conf)
        if self.frontend_conf is None:
            self.frontend_conf = FrontendConfig()
        elif isinstance(self.frontend_conf, dict):
            self.frontend_conf = FrontendConfig.from_dict(self.frontend_conf)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        import inspect

        encoder_conf = params.pop("encoder_conf", None)
        frontend_conf = params.pop("frontend_conf", None)
        valid = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }
        config = cls(**valid)
        if encoder_conf is not None:
            if isinstance(encoder_conf, dict):
                config.encoder_conf = EncoderConfig.from_dict(encoder_conf)
            else:
                config.encoder_conf = encoder_conf
        if frontend_conf is not None:
            if isinstance(frontend_conf, dict):
                config.frontend_conf = FrontendConfig.from_dict(frontend_conf)
            else:
                config.frontend_conf = frontend_conf
        return config
