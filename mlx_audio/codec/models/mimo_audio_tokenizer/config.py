from dataclasses import dataclass, field

from mlx_audio.base import BaseModelArgs


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "mimo_audio_tokenizer"
    d_model: int = 1280
    n_mels: int = 128
    sampling_rate: int = 24000
    nfft: int = 960
    hop_length: int = 240
    window_size: int = 960
    fmin: float = 0
    fmax: float | None = None
    kernel_size: int = 3
    stride_size: int = 2
    avg_pooler: int = 2
    encoder_layers: int = 32
    encoder_skip_layer_id: int | None = 3
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    encoder_causal: bool = False
    encoder_attn_window_size: list[int] = field(default_factory=lambda: [-1, -1])
    decoder_layers: int = 32
    decoder_attention_heads: int = 20
    decoder_ffn_dim: int = 5120
    decoder_kernel_size: int = 3
    decoder_stride_size: int = 2
    decoder_causal: bool = True
    decoder_attn_window_size: list[int] = field(default_factory=lambda: [-1, -1])
    vocoder_dim: int = 256
    vocoder_intermediate_dim: int = 1024
    vocoder_num_layers: int = 16
    vocoder_attention_heads: int = 16
    vocoder_attn_window_size: list[int] = field(default_factory=lambda: [40, 10])
    vocoder_padding: str = "same"
    num_quantizers: int = 20
    codebook_size: list[int] = field(default_factory=lambda: [1024, 1024] + [128] * 18)
    rope_theta: float = 10000
    rope_type: str = "default"
    ln_type: str = "LayerNorm"
    activation_function: str = "gelu"
    weights_format: str = "pytorch"

    def __post_init__(self):
        if len(self.codebook_size) != self.num_quantizers:
            raise ValueError("codebook_size must specify every RVQ codebook")
        if self.rope_type != "default" or self.activation_function != "gelu":
            raise ValueError("MiMo tokenizer requires default RoPE and GELU")
        if self.vocoder_padding != "same":
            raise ValueError("MiMo tokenizer requires same-padding ISTFT")
        for dim, heads in (
            (self.d_model, self.encoder_attention_heads),
            (self.d_model, self.decoder_attention_heads),
            (self.vocoder_dim, self.vocoder_attention_heads),
        ):
            if heads <= 0 or dim % heads or (dim // heads) % 2:
                raise ValueError("Attention dimensions must have an even head size")

    @property
    def downsample_rate(self):
        return self.hop_length * self.stride_size * self.avg_pooler
