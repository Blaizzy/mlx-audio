from pathlib import Path

from mlx_audio.utils import (
    apply_quantization,
    get_model_path,
    load_config,
    load_weights,
)

from .models import gemma3

_MODELS = {"gemma3": gemma3}


def load_lm(model_id: str):
    model_path = get_model_path(model_id)
    config = load_config(model_path)
    model_type = config.get("model_type")
    module = _MODELS.get(model_type)
    if module is None:
        supported = ", ".join(sorted(_MODELS))
        raise ValueError(
            f"Unsupported embedded language model {model_type!r}; supported: {supported}"
        )
    model = module.Model(module.ModelArgs.from_dict(config))
    weights = load_weights(Path(model_path))
    apply_quantization(model, config, weights)
    model.load_weights(list(model.sanitize(weights).items()))

    from transformers import AutoTokenizer

    return model, AutoTokenizer.from_pretrained(model_path)
