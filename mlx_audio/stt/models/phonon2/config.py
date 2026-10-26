from dataclasses import dataclass
from typing import Any

from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDTArgs
from mlx_audio.utils import from_dict


@dataclass
class ModelConfig(ParakeetTDTArgs):
    """Parakeet TDT configuration with Phonon-2/2.5 packed module metadata."""

    phonon2_modules: list[dict]
    model_type: str = "phonon2"
    model_path: str | None = None
    languages: list[str] | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelConfig":
        return from_dict(cls, params)
