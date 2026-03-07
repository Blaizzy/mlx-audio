import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PreprocessorConfig:
    """Audio preprocessor configuration."""

    sample_rate: int = 16000
    normalize: str = "per_feature"
    features: int = 128
    n_fft: int = 512
    window_size: float = 0.025
    window_stride: float = 0.01
    window: str = "hann"
    dither: float = 0.0
    pad_to: int = 0
    pad_value: float = 0.0
    preemph: float = 0.97

    @property
    def win_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "PreprocessorConfig":
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class EncoderConfig:
    """FastConformer encoder configuration."""

    feat_in: int = 128
    n_layers: int = 32
    d_model: int = 1024
    n_heads: int = 8
    ff_expansion_factor: int = 4
    subsampling_factor: int = 8
    self_attention_model: str = "rel_pos"
    subsampling: str = "dw_striding"
    conv_kernel_size: int = 9
    subsampling_conv_channels: int = 256
    pos_emb_max_len: int = 5000
    causal_downsampling: bool = False
    use_bias: bool = True
    xscaling: bool = True
    subsampling_conv_chunking_factor: int = 1

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "EncoderConfig":
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class DecoderConfig:
    """Transformer decoder configuration."""

    num_layers: int = 8
    hidden_size: int = 1024
    num_attention_heads: int = 16
    inner_size: int = 4096
    ffn_dropout: float = 0.0
    attn_score_dropout: float = 0.0
    attn_layer_dropout: float = 0.0

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "DecoderConfig":
        if "decoder" in params and isinstance(params["decoder"], dict):
            params = params["decoder"]
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig:
    """Complete canary model configuration."""

    model_type: str = "canary"
    preprocessor: PreprocessorConfig = None
    encoder: EncoderConfig = None
    transf_decoder: DecoderConfig = None
    vocab_size: int = 16384
    enc_output_dim: int = 1024  # Decoder hidden dimension (same as encoder d_model for this model)

    startofcontext_id: int = 0
    startoftranscript_id: int = 1
    emo_undefined_id: int = 2
    endoftext_id: int = 3

    supported_languages: List[str] = field(default_factory=lambda: [
        "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
        "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
        "sl", "es", "sv", "ru", "uk",
    ])

    def __post_init__(self):
        if self.preprocessor is None:
            self.preprocessor = PreprocessorConfig()
        elif isinstance(self.preprocessor, dict):
            self.preprocessor = PreprocessorConfig.from_dict(self.preprocessor)

        if self.encoder is None:
            self.encoder = EncoderConfig()
        elif isinstance(self.encoder, dict):
            self.encoder = EncoderConfig.from_dict(self.encoder)

        if self.transf_decoder is None:
            self.transf_decoder = DecoderConfig()
        elif isinstance(self.transf_decoder, dict):
            self.transf_decoder = DecoderConfig.from_dict(self.transf_decoder)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        preprocessor = params.pop("preprocessor", None)
        encoder = params.pop("encoder", None)
        transf_decoder = params.pop("transf_decoder", None)

        if preprocessor is not None and isinstance(preprocessor, dict):
            preprocessor = PreprocessorConfig.from_dict(preprocessor)

        if encoder is not None and isinstance(encoder, dict):
            encoder = EncoderConfig.from_dict(encoder)

        if transf_decoder is not None and isinstance(transf_decoder, dict):
            transf_decoder = DecoderConfig.from_dict(transf_decoder)

        filtered_params = {
            k: v for k, v in params.items()
            if k in inspect.signature(cls).parameters
        }

        return cls(
            preprocessor=preprocessor,
            encoder=encoder,
            transf_decoder=transf_decoder,
            **filtered_params,
        )
