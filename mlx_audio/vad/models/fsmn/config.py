import inspect
from dataclasses import dataclass, field
from typing import List

from mlx_audio.base import BaseModelArgs


@dataclass
class FSMNEncoderConfig(BaseModelArgs):
    """FSMN encoder configuration."""

    input_dim: int = 400
    input_affine_dim: int = 140
    fsmn_layers: int = 4
    linear_dim: int = 250
    proj_dim: int = 128
    lorder: int = 20
    rorder: int = 0
    lstride: int = 1
    rstride: int = 0
    output_affine_dim: int = 140
    output_dim: int = 248


@dataclass
class ModelConfig(BaseModelArgs):
    """FSMN-VAD model configuration."""

    model_type: str = "fsmn"
    architecture: str = "fsmn_vad"

    # Encoder
    encoder: FSMNEncoderConfig = None

    # Frontend
    sample_rate: int = 16000
    n_mels: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    lfr_m: int = 5
    lfr_n: int = 1

    # VAD post-processing
    max_end_silence_time: int = 800
    max_start_silence_time: int = 3000
    window_size_ms: int = 200
    sil_to_speech_time_thres: int = 150
    speech_to_sil_time_thres: int = 150
    speech_noise_thres: float = 0.6
    sil_pdf_ids: List[int] = field(default_factory=lambda: [0])
    frame_in_ms: int = 10

    def __post_init__(self):
        if isinstance(self.encoder, dict):
            self.encoder = FSMNEncoderConfig.from_dict(self.encoder)
        if self.encoder is None:
            self.encoder = FSMNEncoderConfig()

    @classmethod
    def from_dict(cls, params):
        params = params.copy()
        valid_keys = set(inspect.signature(cls).parameters.keys())
        return cls(**{k: v for k, v in params.items() if k in valid_keys})
