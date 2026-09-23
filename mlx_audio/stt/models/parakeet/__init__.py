from .parakeet import Model, ModelConfig

DETECTION_HINTS = {"model_type_aliases": {"parakeet_tdt"}}

__all__ = ["Model", "ModelConfig"]


def prepare_config(config, model_path):
    if config.get("ternary_modules"):
        raise ValueError(
            "Use python -m mlx_audio.stt.models.parakeet.convert to preserve "
            "Parakeet Redux's ternary codes and scales without requantizing."
        )
    return config
