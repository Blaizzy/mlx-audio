from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from tqdm import tqdm

import mlx_audio.convert as base_convert

DEFAULT_REPO = "FunAudioLLM/Fun-ASR-Nano-2512"
DEFAULT_MLX_REPO = "mlx-community/Fun-ASR-Nano-2512"
SNAPSHOT_ALLOW_PATTERNS = [
    "model.pt",
    "README.md",
    "Qwen3-0.6B/*",
    "example/*",
]
TORCH_DTYPES = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}
RUNTIME_CONFIG = {
    "input_size": 560,
    "qwen_tokenizer_path": "Qwen3-0.6B",
    "frontend_conf": {
        "fs": 16000,
        "window": "hamming",
        "n_mels": 80,
        "frame_length": 25,
        "frame_shift": 10,
        "lfr_m": 7,
        "lfr_n": 6,
        "cmvn_file": None,
    },
    "audio_encoder_conf": {
        "output_size": 512,
        "attention_heads": 4,
        "linear_units": 2048,
        "num_blocks": 50,
        "tp_blocks": 20,
        "dropout_rate": 0.1,
        "positional_dropout_rate": 0.1,
        "attention_dropout_rate": 0.1,
        "normalize_before": True,
        "kernel_size": 11,
        "sanm_shift": 0,
        "feat_permute": True,
    },
    "audio_adaptor_conf": {
        "downsample_rate": 1,
        "use_low_frame_rate": True,
        "ffn_dim": 2048,
        "llm_dim": 1024,
        "encoder_dim": 512,
        "n_layer": 2,
        "attention_heads": 8,
    },
}


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


def _unwrap_checkpoint(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(obj)!r}")
    for key in ("state_dict", "model", "model_state_dict"):
        value = obj.get(key)
        if isinstance(value, dict):
            return value
    return obj


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        obj = torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    return _unwrap_checkpoint(obj)


def _to_mlx_array(tensor: Any, dtype: str) -> mx.array:
    import torch

    if dtype not in TORCH_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    torch_dtype = getattr(torch, TORCH_DTYPES[dtype])
    if tensor.dtype != torch_dtype:
        tensor = tensor.to(dtype=torch_dtype)
    return mx.array(tensor.contiguous())


def _convert_weights(state: dict[str, Any], dtype: str) -> dict[str, mx.array]:
    converted: dict[str, mx.array] = {}
    for key, tensor in tqdm(state.items(), desc="Converting weights"):
        if key.startswith("module."):
            key = key[len("module.") :]
        if key == "llm.lm_head.weight":
            continue
        if not hasattr(tensor, "shape"):
            continue

        value = _to_mlx_array(tensor, dtype)
        if (
            key.endswith("fsmn_block.weight")
            and value.ndim == 3
            and value.shape[1] == 1
        ):
            value = value.transpose(0, 2, 1)
        converted[key] = value
    return converted


def _build_config(source_dir: Path, repo_id: str) -> dict[str, Any]:
    qwen_cfg = json.loads((source_dir / "Qwen3-0.6B" / "config.json").read_text())
    runtime_cfg = copy.deepcopy(RUNTIME_CONFIG)

    return {
        "model_type": "fun_asr_nano",
        "model_repo": repo_id,
        **runtime_cfg,
        "text_config": qwen_cfg,
        "default_max_tokens": 512,
    }


def _copy_support_files(
    source_dir: Path, out_dir: Path, include_examples: bool
) -> None:
    qwen_src = source_dir / "Qwen3-0.6B"
    qwen_dst = out_dir / "Qwen3-0.6B"
    if qwen_dst.exists():
        shutil.rmtree(qwen_dst)
    ignore = shutil.ignore_patterns(".DS_Store")
    shutil.copytree(qwen_src, qwen_dst, ignore=ignore)

    if include_examples and (source_dir / "example").exists():
        example_dst = out_dir / "example"
        if example_dst.exists():
            shutil.rmtree(example_dst)
        shutil.copytree(source_dir / "example", example_dst, ignore=ignore)


def _write_readme(
    out_dir: Path,
    source_repo_id: str,
    mlx_repo_id: str = DEFAULT_MLX_REPO,
) -> None:
    readme = f"""---
library_name: mlx-audio
tags:
- mlx
- mlx-audio
- speech-to-text
- automatic-speech-recognition
- funasr
license: apache-2.0
---

# Fun-ASR-Nano-2512 MLX

MLX implementation of [{source_repo_id}](https://huggingface.co/{source_repo_id}), a compact speech recognition model with a SenseVoice-style audio encoder and Qwen3-0.6B text decoder.

## Supported Models

| Model | Parameters | Languages | Description |
|-------|------------|-----------|-------------|
| [{mlx_repo_id}](https://huggingface.co/{mlx_repo_id}) | 0.8B | ZH, EN, JA | Transcription-only Fun-ASR-Nano checkpoint |

## Python Usage

```python
from mlx_audio.stt import load

model = load("{mlx_repo_id}")

result = model.generate(
    "audio.wav",
    language="zh",
    hotwords=["开放时间"],
)
print(result.text)
```

## CLI Usage

```bash
mlx_audio.stt.generate \\
  --model {mlx_repo_id} \\
  --audio audio.wav \\
  --output-path transcript \\
  --language zh \\
  --gen-kwargs '{{"hotwords": ["开放时间"]}}'
```

## Options

Language hints use ISO-style codes. Supported hints include `zh`, Chinese dialect codes that map to the Chinese prompt (`yue`, `wuu`, `nan`, `hak`, `gan`, `hsn`, `cjy`), `en`, and `ja`. Pass `None` or `auto` to omit the language constraint.

Hotwords can be supplied as a list of domain terms:

```python
result = model.generate("audio.wav", language="zh", hotwords=["开放时间", "地址"])
```

Inverse text normalization is enabled by default. Set `itn=False` to ask the model to keep spoken-form text:

```python
result = model.generate("audio.wav", language="zh", itn=False)
```

## Notes

This runtime is transcription-only. The public checkpoint does not include timestamp head weights, so segment and word timestamps are not emitted by this model.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def convert(
    hf_repo_or_dir: str | Path = DEFAULT_REPO,
    out_dir: str | Path = "Fun-ASR-Nano-2512-mlx",
    *,
    dtype: str = "bfloat16",
    revision: str | None = None,
    include_examples: bool = True,
    overwrite: bool = True,
) -> Path:
    source_dir = _resolve_source(hf_repo_or_dir, revision=revision)
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = _load_torch_checkpoint(source_dir / "model.pt")
    weights = _convert_weights(state, dtype=dtype)

    mx.save_safetensors(
        str(out_dir / "model.safetensors"), weights, metadata={"format": "mlx"}
    )
    del weights
    mx.clear_cache()

    repo_id = _source_repo_id(hf_repo_or_dir, source_dir)
    config = _build_config(source_dir, repo_id)
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _copy_support_files(source_dir, out_dir, include_examples=include_examples)
    _write_readme(out_dir, repo_id)

    print(f"[INFO] Conversion complete: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Fun-ASR-Nano-2512 PyTorch weights to MLX format"
    )
    parser.add_argument(
        "--hf-path",
        default=DEFAULT_REPO,
        help="Fun-ASR-Nano HF repo id or local snapshot path",
    )
    parser.add_argument(
        "--mlx-path",
        default="Fun-ASR-Nano-2512-mlx",
        help="Output directory for the converted MLX model",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=base_convert.MODEL_CONVERSION_DTYPES,
        help="Dense weight dtype for the converted MLX model",
    )
    parser.add_argument("--revision", default=None, help="Optional HF revision")
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Do not copy the upstream example audio files",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not remove an existing output directory before conversion",
    )
    args = parser.parse_args()
    convert(
        args.hf_path,
        args.mlx_path,
        dtype=args.dtype,
        revision=args.revision,
        include_examples=not args.no_examples,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
