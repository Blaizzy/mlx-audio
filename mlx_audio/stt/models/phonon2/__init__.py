from .config import ModelConfig
from .phonon2 import Model
from .transport import prepare_model_path

DETECTION_HINTS = {"model_type_aliases": {"parakeet_tdt_five_value"}}

__all__ = ["Model", "ModelConfig", "prepare_model_path"]


def prepare_config(config, model_path):
    # Generic conversion would cast packed codes and may requantize row palettes.
    raise ValueError(
        "Phonon-2/2.5 generic conversion and requantization are unsupported. "
        "Load FermionResearch/Phonon-2.5, FermionResearch/Phonon-2.5-English, "
        "or mlx-community/Phonon-2 directly with mlx_audio.stt.load."
    )
