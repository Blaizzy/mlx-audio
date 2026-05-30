import inspect
from dataclasses import dataclass
from typing import Optional

from mlx_audio.base import BaseModelArgs


@dataclass
class BranchConfig(BaseModelArgs):
    """Configuration for one Silero VAD sample-rate branch."""

    sample_rate: int = 16000
    filter_length: int = 256
    hop_length: int = 128
    pad: int = 64
    cutoff: int = 129
    context_size: int = 64
    chunk_size: int = 512


@dataclass
class ModelConfig(BaseModelArgs):
    """Silero voice activity detector configuration."""

    model_type: str = "silero_vad"
    architecture: str = "silero_vad"
    dtype: str = "float32"
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    branch_16k: Optional[BranchConfig] = None
    branch_8k: Optional[BranchConfig] = None

    def __post_init__(self):
        if isinstance(self.branch_16k, dict):
            self.branch_16k = BranchConfig.from_dict(self.branch_16k)
        if self.branch_16k is None:
            self.branch_16k = BranchConfig()

        if isinstance(self.branch_8k, dict):
            self.branch_8k = BranchConfig.from_dict(self.branch_8k)
        if self.branch_8k is None:
            self.branch_8k = BranchConfig(
                sample_rate=8000,
                filter_length=128,
                hop_length=64,
                pad=32,
                cutoff=65,
                context_size=32,
                chunk_size=256,
            )

    @classmethod
    def from_dict(cls, params):
        params = params.copy()
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
