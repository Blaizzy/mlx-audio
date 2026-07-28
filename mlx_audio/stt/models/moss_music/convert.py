from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from tqdm import tqdm

DEFAULT_REPO = "OpenMOSS-Team/MOSS-Music-8B-Thinking"
DEFAULT_MLX_REPO = "mlx-community/MOSS-Music-8B-Thinking-MLX"

SNAPSHOT_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.jinja",
    "*.txt",
    "*.py",
    "README.md",
]

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
]


def _resolve_source(hf_repo_or_dir: str | Path, revision: str | None = None) -> Path:
    candidate = Path(hf_repo_or_dir).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return Path(
        snapshot_download(
            str(hf_repo_or_dir),
            revision=revision,
            allow_patterns=SNAPSHOT_ALLOW_PATTERNS,
        )
    )


def _source_repo_id(hf_repo_or_dir: str | Path, source_dir: Path) -> str:
    source_arg = str(hf_repo_or_dir)
    if not Path(source_arg).expanduser().exists() and "/" in source_arg:
        return source_arg
    for part in source_dir.parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return DEFAULT_REPO


def _build_config(source_dir: Path, source_repo_id: str) -> dict[str, Any]:
    config = json.loads((source_dir / "config.json").read_text())
    config["model_type"] = "moss_music"
    config["model_repo"] = source_repo_id
    config["sample_rate"] = 16000
    config["audio_token_id"] = 151654
    config["audio_start_id"] = 151669
    config["audio_end_id"] = 151670
    config["pad_token_id"] = config.get("pad_token_id", 151643)
    config["enable_time_marker"] = True
    config["strip_thinking"] = True
    config["default_prompt"] = (
        "Please give a detailed musical description of this clip."
    )

    audio_config = dict(config.get("audio_config") or {})
    audio_config.setdefault("n_window", 200)
    audio_config.setdefault("conv_chunksize", 64)
    config["audio_config"] = audio_config
    return config


def _convert_weights(source_dir: Path, dtype: mx.Dtype) -> dict[str, mx.array]:
    from .moss_music import Model

    weight_files = sorted(source_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {source_dir}")

    converted: dict[str, mx.array] = {}
    for path in tqdm(weight_files, desc="Converting weights"):
        for key, value in mx.load(str(path)).items():
            sanitized = Model.sanitize({key: value})
            for out_key, out_value in sanitized.items():
                converted[out_key] = out_value.astype(dtype)
    return converted


def _quantize(
    config: dict[str, Any],
    weights: dict[str, mx.array],
    q_bits: int,
    q_group_size: int,
) -> tuple[dict[str, Any], dict[str, mx.array]]:
    from mlx_lm.utils import quantize_model

    from .config import ModelConfig
    from .moss_music import Model

    model = Model(ModelConfig.from_dict(config))
    model.load_weights(list(weights.items()))

    def predicate(path, module) -> bool:
        return (
            not str(path).startswith(("audio_encoder", "audio_adapter", "deepstack"))
            and hasattr(module, "to_quantized")
            and hasattr(module, "weight")
            and module.weight.shape[-1] % q_group_size == 0
        )

    model, q_config = quantize_model(
        model,
        config,
        q_group_size,
        q_bits,
        quant_predicate=predicate,
    )
    return q_config, dict(tree_flatten(model.parameters()))


def _copy_support_files(source_dir: Path, out_dir: Path) -> None:
    for name in TOKENIZER_FILES:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def _write_readme(out_dir: Path, source_repo_id: str, mlx_repo_id: str) -> None:
    readme = f"""---
library_name: mlx-audio
tags:
- mlx
- mlx-audio
- speech-to-text
- audio-text-to-text
- music-understanding
- lyrics-asr
license: apache-2.0
---

# MOSS-Music-8B-Thinking MLX

MLX port of [{source_repo_id}](https://huggingface.co/{source_repo_id}) for
`mlx-audio`.

```python
from mlx_audio.stt import load

model = load("{mlx_repo_id}")
result = model.generate("song.wav")
print(result.text)
```

```bash
mlx_audio.stt.generate \\
  --model {mlx_repo_id} \\
  --audio song.wav \\
  --output-path output \\
  --prompt "Please give a detailed musical description of this clip."
```

The default prompt asks for a general music description. Use `--prompt` for
lyrics ASR, chord analysis, key/tempo questions, or other music QA tasks. The
Thinking checkpoint may emit reasoning internally; mlx-audio strips
`<think>...</think>` blocks by default.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def convert(
    hf_repo_or_dir: str | Path = DEFAULT_REPO,
    out_dir: str | Path = "MOSS-Music-8B-Thinking-MLX",
    *,
    dtype: str = "bfloat16",
    revision: str | None = None,
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    mlx_repo_id: str = DEFAULT_MLX_REPO,
    overwrite: bool = True,
) -> Path:
    source_dir = _resolve_source(hf_repo_or_dir, revision=revision)
    source_repo_id = _source_repo_id(hf_repo_or_dir, source_dir)
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlx_dtype = getattr(mx, dtype)
    config = _build_config(source_dir, source_repo_id)
    weights = _convert_weights(source_dir, mlx_dtype)

    if quantize:
        config, weights = _quantize(config, weights, q_bits, q_group_size)

    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)
    _copy_support_files(source_dir, out_dir)
    _write_readme(out_dir, source_repo_id, mlx_repo_id)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MOSS-Music to MLX")
    parser.add_argument("--hf-path", default=DEFAULT_REPO)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--mlx-repo-id", default=DEFAULT_MLX_REPO)
    args = parser.parse_args()

    convert(
        hf_repo_or_dir=args.hf_path,
        out_dir=args.mlx_path,
        dtype=args.dtype,
        revision=args.revision,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        mlx_repo_id=args.mlx_repo_id,
    )


if __name__ == "__main__":
    main()
