import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def load_source_weights(path: Path, weights_name: str | None = None) -> dict:
    if weights_name is None and (path / "config.json").is_file():
        weights_name = json.loads((path / "config.json").read_text()).get(
            "weights_name"
        )
    # Voice embeddings are also safetensors; never merge them into model weights.
    preferred = [weights_name, "model.safetensors"]
    for name in preferred:
        if name and (path / name).is_file():
            return mx.load(str(path / name))
    files = sorted(path.glob("model*.safetensors"))
    if files:
        return {k: v for file in files for k, v in mx.load(str(file)).items()}
    files = sorted(path.glob("*.gguf"))
    if len(files) == 1:
        return mx.load(str(files[0]))
    # Load NPZ and custom-named safetensors shards, excluding voice embeddings.
    weights = {}
    for file in sorted([*path.glob("*.safetensors"), *path.glob("*.npz")]):
        shard = mx.load(str(file))
        if any(key.startswith(("flow_lm.", "mimi.")) for key in shard):
            weights.update(shard)
    if weights:
        return weights
    raise FileNotFoundError(
        f"Expected model safetensors or one GGUF checkpoint in {path}."
    )


def convert_weights(model, weights):
    return model.sanitize(weights)


def _prepare_quantized_modules(model, weights):
    # mx.load exposes GGUF Q8_0 matrices as MLX affine packed weights. Keep them
    # quantized; derive their packing from the target matrix and scale shapes.
    def quantized(path, module):
        if not hasattr(module, "to_quantized") or isinstance(
            module, nn.QuantizedLinear
        ):
            return False
        weight = weights.get(path + ".weight")
        scales = weights.get(path + ".scales")
        if weight is None or scales is None or weight.dtype != mx.uint32:
            return False
        width = module.weight.shape[-1]
        return {
            "bits": weight.shape[-1] * 32 // width,
            "group_size": width // scales.shape[-1],
        }

    if model.config.quantization is None:
        nn.quantize(model, class_predicate=quantized)


def convert(
    hf_path: str,
    mlx_path: str = "phonon-tts",
    quantize: bool = False,
    q_group_size: int = 32,
    q_bits: int = 8,
    dtype: str | None = None,
    dequantize: bool = False,
):
    """Preserve source precision and packing unless explicitly overridden."""
    from mlx_audio.convert import copy_model_files, load_config
    from mlx_audio.lm.convert import dequantize_model, save_config, save_model
    from mlx_audio.utils import apply_quantization, get_model_path

    from .pocket_tts import Model, ModelConfig

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")
    if dtype not in (None, "float16", "bfloat16", "float32"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    source = get_model_path(
        hf_path,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.gguf",
            "*.model",
            "*.npz",
        ],
    )
    destination = Path(mlx_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different directories.")
    config = load_config(source)
    model = Model(ModelConfig.from_dict({**config, "model_path": str(source)}))
    weights = convert_weights(model, load_source_weights(source))
    _prepare_quantized_modules(model, weights)
    apply_quantization(model, config, weights)
    model.load_weights(list(weights.items()), strict=True)
    if dtype is not None:
        model.load_weights(
            [
                (
                    key,
                    (
                        value.astype(getattr(mx, dtype))
                        if mx.issubdtype(value.dtype, mx.floating)
                        else value
                    ),
                )
                for key, value in tree_flatten(model.parameters())
            ]
        )
    if quantize:
        # Existing packed matrices retain their original format and values.
        nn.quantize(
            model,
            group_size=q_group_size,
            bits=q_bits,
            class_predicate=lambda _, module: isinstance(
                module, (nn.Linear, nn.Embedding)
            )
            and module.weight.shape[-1] % q_group_size == 0,
        )
    if dequantize:
        model = dequantize_model(model)
    packed_layers = {
        name: {
            "group_size": module.group_size,
            "bits": module.bits,
            "mode": module.mode,
        }
        for name, module in model.named_modules()
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    }
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    if packed_layers:
        defaults = min(packed_layers.values(), key=lambda spec: spec["group_size"])
        config["quantization"] = {**defaults, **packed_layers}
    config["model_type"] = "pocket_tts"
    config["weights_name"] = "model.safetensors"
    config["mimi"]["encoder_enabled"] = model.config.mimi.encoder_enabled
    config.pop("model_path", None)
    destination.mkdir(parents=True, exist_ok=True)
    copy_model_files(source, destination)
    save_model(destination, model, donate_model=True)
    save_config(config, destination / "config.json")
    print(f"Saved Pocket TTS model to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-path", required=True, help="Source directory or Hugging Face repository"
    )
    parser.add_argument("--mlx-path", default="phonon-tts")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize remaining eligible floating-point layers",
    )
    parser.add_argument("--q-bits", type=int, default=8)
    parser.add_argument("--q-group-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--dequantize", action="store_true")
    convert(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
