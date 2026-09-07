"""Convert Xiaomi's standalone MiMo tokenizer to MLX tensor layouts."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .model import MiMoAudioTokenizer


def convert(hf_path, mlx_path, dtype=None, revision=None):
    model = MiMoAudioTokenizer.from_pretrained(hf_path, dtype=dtype, revision=revision)
    destination = Path(mlx_path)
    destination.mkdir(parents=True, exist_ok=True)
    config = asdict(model.config)
    config["weights_format"] = "mlx"
    config["model_type"] = "mimo_audio_tokenizer"
    source = (
        f"local checkpoint `{Path(hf_path).name}`"
        if Path(hf_path).exists()
        else f"[{hf_path}](https://huggingface.co/{hf_path})"
    )
    mx.save_safetensors(
        str(destination / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "README.md").write_text(
        "---\nlicense: mit\nlibrary_name: mlx-audio\n"
        "tags:\n- mlx\n- audio\n---\n\n"
        "# MiMo-Audio-Tokenizer for MLX\n\n"
        "24 kHz mono audio tokenizer with 20 residual codebooks. MiMo-Audio uses the first eight.\n\n"
        "```python\nfrom mlx_audio.codec import MiMoAudioTokenizer\n"
        f"codec = MiMoAudioTokenizer.from_pretrained({str(mlx_path)!r})\n"
        "codes = codec.encode('input.wav', num_quantizers=20)\n"
        "audio = codec.decode(codes)\n```\n\n"
        f"Converted from {source}.\n"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default="XiaomiMiMo/MiMo-Audio-Tokenizer")
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()
    print(
        convert(
            args.hf_path,
            args.mlx_path,
            getattr(mx, args.dtype) if args.dtype else None,
            args.revision,
        )
    )


if __name__ == "__main__":
    main()
