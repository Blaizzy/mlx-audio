"""Save Parakeet Redux in MLX's lossless 2-bit representation.

python -m mlx_audio.stt.models.parakeet.convert \
    --hf-path moondream/parakeet-redux --mlx-path ./parakeet-redux-mlx

No PyTorch or Photon installation is required. The default preserves source
floating-point dtypes; --dtype casts dense parameters only, retaining the exact
ternary scales. The auxiliary VAD head is not exported.
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_audio.stt import load
from mlx_audio.utils import get_model_path

from .redux import ParakeetRedux


def convert(
    hf_path: str,
    mlx_path: str,
    *,
    dtype: str | None = None,
    revision: str | None = None,
    upload_repo: str | None = None,
) -> Path:
    if dtype not in (None, "float16", "bfloat16", "float32"):
        raise ValueError(f"Unsupported dense dtype: {dtype}")
    source = get_model_path(
        hf_path,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors"],
    )
    destination = Path(mlx_path)
    if source.resolve() == destination.resolve():
        raise ValueError("The output directory must differ from the source checkpoint.")
    model = load(source, strict=True)
    if not isinstance(model, ParakeetRedux):
        raise ValueError("This converter supports ternary Parakeet Redux checkpoints.")
    weights = dict(tree_flatten(model.parameters()))
    if dtype is not None:
        target_dtype = getattr(mx, dtype)
        weights = {
            name: (
                value.astype(target_dtype)
                if mx.issubdtype(value.dtype, mx.floating)
                and not name.endswith((".scales", ".biases"))
                else value
            )
            for name, value in weights.items()
        }
    config = {k: v for k, v in model._source_config.items() if k != "model_path"}
    config.update(
        model_type="parakeet_tdt",
        redux_weight_format="mlx-2bit",
        supports_vad=False,
    )
    if dtype is not None:
        config["dtype"] = dtype
    destination.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(destination / "model.safetensors"), weights)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(source / "tokenizer.json", destination / "tokenizer.json")
    source_model = hf_path if not Path(hf_path).exists() else "moondream/parakeet-redux"
    display_repo = upload_repo or str(destination)
    (destination / "README.md").write_text(
        "---\nlicense: cc-by-4.0\nlibrary_name: mlx-audio\n"
        "pipeline_tag: automatic-speech-recognition\n"
        f"base_model: {source_model}\ntags: [mlx, mlx-audio, parakeet, ternary]\n"
        "---\n\n# Parakeet Redux for MLX\n\n"
        f"Converted from [{source_model}](https://huggingface.co/{source_model}), "
        "Moondream's ternary version of NVIDIA's Parakeet TDT 0.6B v3. "
        "The original ternary codes and scales are preserved in MLX's affine "
        "2-bit layout; no additional weight quantization is performed. "
        "Dense parameters retain their source dtypes unless explicitly cast.\n\n"
        "Supports the original 25 languages, transcription, timestamps, and "
        "overlapping chunked transcription. The auxiliary VAD head and "
        "Photon's automatic pause-based segmentation are not included.\n\n"
        "```python\nfrom mlx_audio.stt import load\n\n"
        f"model = load({display_repo!r})\n"
        'result = model.generate("speech.wav")\nprint(result.text)\n'
        "```\n\nFor long recordings, pass `chunk_duration=30.0` and "
        "`overlap_duration=2.0`, or use `stream=True`.\n"
    )
    if upload_repo is not None:
        from mlx_audio.convert import Domain, upload_to_hub

        upload_to_hub(
            destination,
            upload_repo,
            source_model,
            Domain.STT,
            model_card_path=destination / "README.md",
        )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default="moondream/parakeet-redux")
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--upload-repo")
    args = parser.parse_args()
    print(convert(**vars(args)))


if __name__ == "__main__":
    main()
