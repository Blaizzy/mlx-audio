from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from safetensors.mlx import save_file

import mlx_audio.convert as base_convert
from mlx_audio.stt.models.qwen3_asr.config import ModelConfig as Qwen3ModelConfig

from .convert_lora import load_lora_adapter
from .convert_router import convert_router_weights

BASE_SUBDIR = "Qwen3-ASR-1.7B"
ROUTER_SOURCE = "audio_quality_router/best_acc_model.safetensors"
LORA_SUBDIR = "mega-asr-merged"
ROUTER_CONFIG = {
    "d_model": 256,
    "nhead": 4,
    "dim_feedforward": 1024,
    "num_layers": 1,
    "n_mels": 80,
    "max_len": 850,
}
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.json",
    "preprocessor_config.json",
]


def _resolve_layout(hf_repo_or_dir: str | Path) -> tuple[Path, Path]:
    candidate = Path(hf_repo_or_dir).expanduser()
    if candidate.exists():
        candidate = candidate.resolve()
        if candidate.name == BASE_SUBDIR:
            return candidate.parent, candidate
        base_dir = candidate / BASE_SUBDIR
        if base_dir.exists():
            return candidate, base_dir
        raise FileNotFoundError(
            f"{BASE_SUBDIR} not found under {candidate}; expected Mega-ASR root snapshot"
        )

    root_dir = Path(base_convert.get_model_path(str(hf_repo_or_dir)))
    base_dir = root_dir / BASE_SUBDIR
    if not base_dir.exists():
        raise FileNotFoundError(f"{BASE_SUBDIR} missing from snapshot {root_dir}")
    return root_dir, base_dir


def _mega_config(base_dir: Path) -> dict[str, object]:
    raw = json.loads((base_dir / "config.json").read_text())
    cfg = Qwen3ModelConfig.from_dict(raw)
    return {
        "model_type": "mega_asr",
        "model_repo": cfg.model_repo,
        "audio_token_id": cfg.audio_token_id,
        "audio_start_token_id": cfg.audio_start_token_id,
        "audio_end_token_id": cfg.audio_end_token_id,
        "support_languages": cfg.support_languages,
        "audio_config": dict(cfg.audio_config.__dict__),
        "text_config": dict(cfg.text_config.__dict__),
        "router_config": dict(ROUTER_CONFIG),
        "router_weights": "extras/router.safetensors",
        "lora_weights": "extras/lora.safetensors",
    }


def _copy_support_files(base_dir: Path, out_dir: Path) -> None:
    for name in TOKENIZER_FILES:
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def _flatten_lora(adapter_dir: Path) -> dict[str, mx.array]:
    adapter = load_lora_adapter(adapter_dir)
    flattened: dict[str, mx.array] = {}
    for module, factors in adapter.items():
        flattened[f"{module}.lora_A"] = factors["A"].astype(mx.float32)
        flattened[f"{module}.lora_B"] = (
            float(factors["scaling"]) * factors["B"]
        ).astype(mx.float32)
    return flattened


def convert(
    hf_repo_or_dir: str | Path,
    out_dir: str | Path,
    dtype: str = "bfloat16",
) -> Path:
    root_dir, base_dir = _resolve_layout(hf_repo_or_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_convert.convert(
        hf_path=str(base_dir),
        mlx_path=str(out_dir),
        dtype=dtype,
        quantize=False,
        model_domain="stt",
    )

    extras_dir = out_dir / "extras"
    extras_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        convert_router_weights(root_dir / ROUTER_SOURCE),
        str(extras_dir / "router.safetensors"),
    )

    save_file(
        _flatten_lora(root_dir / LORA_SUBDIR),
        str(extras_dir / "lora.safetensors"),
    )

    _copy_support_files(base_dir, out_dir)
    (out_dir / "config.json").write_text(
        json.dumps(_mega_config(base_dir), indent=2) + "\n"
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Mega-ASR Hugging Face weights to MLX format"
    )
    parser.add_argument(
        "--hf-path", required=True, help="Mega-ASR HF repo or local snapshot path"
    )
    parser.add_argument(
        "--mlx-path", required=True, help="Output directory for the MLX Mega-ASR model"
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=base_convert.MODEL_CONVERSION_DTYPES,
        help="Dense base weight dtype for the converted MLX model",
    )
    args = parser.parse_args()
    convert(args.hf_path, args.mlx_path, dtype=args.dtype)


if __name__ == "__main__":
    main()
