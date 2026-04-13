from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EncoderConfig:
    n_layers: int = 16
    n_head: int = 20
    d_model: int = 1280
    kernel_size: int = 33
    pe_maxlen: int = 5000

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DecoderConfig:
    n_layers: int = 16
    n_head: int = 20
    d_model: int = 1280
    pe_maxlen: int = 5000

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    model_type: str = "fireredasr2"
    idim: int = 80
    odim: int = 8667
    d_model: int = 1280
    sos_id: int = 3
    eos_id: int = 4
    pad_id: int = 2
    blank_id: int = 0
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    model_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d):
        enc_d = d.get("encoder", {})
        dec_d = d.get("decoder", {})
        enc = EncoderConfig.from_dict(enc_d) if enc_d else EncoderConfig()
        dec = DecoderConfig.from_dict(dec_d) if dec_d else DecoderConfig()
        top_fields = {
            k: v
            for k, v in d.items()
            if k in cls.__dataclass_fields__ and k not in ("encoder", "decoder")
        }
        return cls(encoder=enc, decoder=dec, **top_fields)
