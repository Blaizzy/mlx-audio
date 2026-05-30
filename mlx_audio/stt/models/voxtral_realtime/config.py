import inspect
import math
from dataclasses import dataclass
from typing import Optional

from mlx_audio.base import BaseModelArgs

SAMPLE_RATE = 16000
FRAME_RATE = 12.5
HOP_LENGTH = 160
RAW_AUDIO_LENGTH_PER_TOK = int(SAMPLE_RATE // FRAME_RATE)  # 1280
AUDIO_LENGTH_PER_TOK = RAW_AUDIO_LENGTH_PER_TOK // HOP_LENGTH  # 8


def _num_audio_tokens(audio_len: int) -> int:
    if audio_len % HOP_LENGTH != 0:
        audio_len = math.ceil(audio_len / HOP_LENGTH - 1)
    else:
        audio_len = audio_len // HOP_LENGTH
    return math.ceil(audio_len / AUDIO_LENGTH_PER_TOK)


def _num_delay_tokens(delay_ms: float) -> int:
    delay_len = int(delay_ms / 1000.0 * SAMPLE_RATE)
    return _num_audio_tokens(delay_len)


@dataclass
class AudioEncodingConfig(BaseModelArgs):
    sampling_rate: int = 16000
    frame_rate: float = 12.5
    num_mel_bins: int = 128
    hop_length: int = 160
    window_size: int = 400
    global_log_mel_max: float = 1.5


@dataclass
class EncoderConfig(BaseModelArgs):
    dim: int = 1280
    n_layers: int = 32
    n_heads: int = 32
    head_dim: int = 64
    hidden_dim: int = 5120
    n_kv_heads: int = 32
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    sliding_window: int = 750
    causal: bool = True
    use_biases: bool = True
    downsample_factor: int = 4


@dataclass
class DecoderConfig(BaseModelArgs):
    dim: int = 3072
    n_layers: int = 26
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 9216
    vocab_size: int = 131072
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    sliding_window: int = 8192
    tied_embeddings: bool = True
    ada_rms_norm_t_cond: bool = True
    ada_rms_norm_t_cond_dim: int = 32


@dataclass
class ModelConfig(BaseModelArgs):
    """Voxtral Mini 4B Realtime model config.

    Recommended settings: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602#recommended-settings
    """

    model_type: str = "voxtral_realtime"
    encoder_args: Optional[EncoderConfig] = None
    decoder: Optional[DecoderConfig] = None
    audio_encoding_args: Optional[AudioEncodingConfig] = None
    transcription_delay_ms: int = (
        480  # Recommended: 480ms (sweet spot of performance and low latency)
    )

    # Derived from decoder
    vocab_size: int = 131072
    hidden_size: int = 3072

    # Special token IDs
    bos_token_id: int = 1
    eos_token_id: int = 2
    streaming_pad_token_id: int = 32

    # Streaming constants
    n_left_pad_tokens: int = 32

    def __post_init__(self):
        if isinstance(self.encoder_args, dict):
            # Handle nested audio_encoding_args inside encoder_args
            if "audio_encoding_args" in self.encoder_args:
                audio_enc = self.encoder_args.pop("audio_encoding_args")
                if self.audio_encoding_args is None:
                    self.audio_encoding_args = AudioEncodingConfig.from_dict(audio_enc)
            self.encoder_args = EncoderConfig.from_dict(self.encoder_args)
        if self.encoder_args is None:
            self.encoder_args = EncoderConfig()

        if isinstance(self.decoder, dict):
            self.decoder = DecoderConfig.from_dict(self.decoder)
        if self.decoder is None:
            self.decoder = DecoderConfig()

        if isinstance(self.audio_encoding_args, dict):
            self.audio_encoding_args = AudioEncodingConfig.from_dict(
                self.audio_encoding_args
            )
        if self.audio_encoding_args is None:
            self.audio_encoding_args = AudioEncodingConfig()

        self.vocab_size = self.decoder.vocab_size
        self.hidden_size = self.decoder.dim

    @classmethod
    def from_dict(cls, params):
        params = params.copy()

        # Handle nested configs
        if "encoder_args" in params and isinstance(params["encoder_args"], dict):
            enc = params["encoder_args"].copy()
            if "audio_encoding_args" in enc:
                audio_enc = enc.pop("audio_encoding_args")
                params["audio_encoding_args"] = audio_enc
            params["encoder_args"] = enc

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
