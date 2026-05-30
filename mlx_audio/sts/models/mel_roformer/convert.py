# Copyright (c) 2026 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""Convert PyTorch Mel-Band-RoFormer checkpoints to MLX format.

Features:
    - Accepts .ckpt, .pt, and .safetensors input
    - Idempotent: output is content-addressed by SHA-256 of input; skipped if exists
    - Writes a companion <basename>.<hash>.config.json with architecture hyperparameters
    - License-aware: warns when known checkpoint sources are detected in input path
    - Strips training state (optimizer, schedulers, EMA, running stats)
    - Splits packed to_qkv projections into separate to_q/to_k/to_v

Usage:
    python -m mlx_audio.sts.models.mel_roformer.convert \\
        --input path/to/model.ckpt \\
        --output path/to/output_dir/ \\
        [--preset kim_vocal_2|viperx_vocals|zfturbo_bs_roformer|zfturbo_vocals_v1|custom] \\
        [--force]

The converted weights will be saved as:
    <output_dir>/<input_basename>.<sha256[:8]>.safetensors
    <output_dir>/<input_basename>.<sha256[:8]>.config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Optional

from .config import MelRoFormerConfig

# ---------- License detection ----------
#
# Substring match on input path / URL. Not authoritative — catches the common
# "I forgot which checkpoint this was" case and prompts the user to confirm.

_LICENSE_HINTS = [
    # (substring, preset_name, license_tag, human_note)
    (
        "KimberleyJSN",
        "kim_vocal_2",
        "MIT",
        "Kim Vocal 2 was relicensed to MIT by the author in April 2026. "
        "Previously tagged GPL-3.0; inference and redistribution are both "
        "unrestricted under the current license.",
    ),
    (
        "kimvocal",
        "kim_vocal_2",
        "MIT",
        "Likely Kim Vocal 2 variant — MIT since April 2026 (was GPL-3.0). "
        "Confirm against the specific source repo's current LICENSE.",
    ),
    (
        "TRvlvr",
        "viperx_vocals",
        "undeclared",
        "No LICENSE file in TRvlvr/model_repo. Default copyright applies; "
        "no explicit redistribution right.",
    ),
    (
        "viperx",
        "viperx_vocals",
        "undeclared",
        "viperx checkpoints are typically hosted in TRvlvr/model_repo with "
        "no declared license. Default copyright applies.",
    ),
    (
        "anvuew",
        None,
        "GPL-3.0",
        "anvuew checkpoints tagged GPL-3.0 in response to Intel OpenVINO conversion.",
    ),
    (
        "ZFTurbo",
        "zfturbo_bs_roformer",
        "MIT (inherited)",
        "MSS-Training release-asset checkpoints inherit the repo's MIT license. "
        "Free to use and redistribute.",
    ),
    (
        "Music-Source-Separation-Training",
        "zfturbo_bs_roformer",
        "MIT (inherited)",
        "MSS-Training release assets inherit MIT. Free to redistribute.",
    ),
]


def _detect_license(input_path: Path) -> Optional[tuple]:
    """Match known substrings against input path/name.

    Returns (substring, suggested_preset, license_tag, note) or None.
    """
    path_str = str(input_path).lower()
    for substring, preset, license_tag, note in _LICENSE_HINTS:
        if substring.lower() in path_str:
            return substring, preset, license_tag, note
    return None


# ---------- Strip training state ----------

_STRIP_PREFIXES = (
    "optimizer_states",
    "lr_schedulers",
    "callbacks",
    "hyper_parameters",
    "ema.",
    "ema_model.",
)

_STRIP_SUFFIXES = (
    ".num_batches_tracked",
    ".running_mean",
    ".running_var",
)


def _should_strip(key: str) -> bool:
    """Return True if this weight key should be excluded from conversion."""
    for prefix in _STRIP_PREFIXES:
        if key.startswith(prefix):
            return True
    for suffix in _STRIP_SUFFIXES:
        if key.endswith(suffix):
            return True
    return False


def _extract_state_dict(obj) -> Dict:
    """Extract the model state dict from various PyTorch checkpoint formats.

    Handles:
    - Pure state dict (keys map directly to tensors)
    - PyTorch Lightning checkpoint (`{'state_dict': ...}`)
    - MSS-Training checkpoint (`{'state_dict': ...}` with `model.` prefix)
    """
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint root must be a dict")

    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
    else:
        state = obj

    if any(k.startswith("model.") for k in state.keys()):
        state = {
            (k[len("model.") :] if k.startswith("model.") else k): v
            for k, v in state.items()
        }

    return state


# ---------- Content addressing ----------


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file, streaming in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _content_addressed_name(input_path: Path, digest_hex: str, dtype: str) -> str:
    """Build the output basename: `<input_stem>.<hash[:8]>.<dtype>`.

    Including ``dtype`` in the basename keeps fp32 / fp16 / bf16 conversions of
    the same source from colliding when re-run with different precisions.
    """
    return f"{input_path.stem}.{digest_hex[:8]}.{dtype}"


def _resolve_dtype(name: str):
    """Resolve a dtype string to an MLX dtype.

    Imported lazily so the module loads without MLX.
    """
    import mlx.core as mx

    table = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    if name not in table:
        raise ValueError(
            f"Unknown dtype '{name}'. Choose one of: {sorted(table.keys())}."
        )
    return table[name]


# ---------- Config resolution ----------


def _resolve_preset(name: str) -> MelRoFormerConfig:
    """Resolve a preset name to a config object."""
    presets = {
        "kim_vocal_2": MelRoFormerConfig.kim_vocal_2,
        "viperx_vocals": MelRoFormerConfig.viperx_vocals,
        "zfturbo_bs_roformer": MelRoFormerConfig.zfturbo_bs_roformer,
        "zfturbo_vocals_v1": MelRoFormerConfig.zfturbo_vocals_v1,
    }
    if name not in presets:
        raise ValueError(
            f"Unknown preset '{name}'. Choose one of: {sorted(presets.keys())}, "
            f"or omit --preset and pass individual --depth/--num-bands flags."
        )
    return presets[name]()


def _config_to_dict(config: MelRoFormerConfig) -> Dict:
    """Serialize a MelRoFormerConfig to a JSON-safe dict."""
    return {f.name: getattr(config, f.name) for f in fields(config)}


# ---------- Main conversion ----------


def convert_checkpoint(
    input_path: Path,
    output_dir: Path,
    config: Optional[MelRoFormerConfig] = None,
    force: bool = False,
    dtype: str = "float32",
    print_fn=print,
) -> tuple:
    """Convert a PyTorch Mel-Band-RoFormer checkpoint to MLX safetensors.

    Args:
        input_path: Path to a PyTorch checkpoint.
        output_dir: Directory to write the converted weights to.
        config: Optional MelRoFormerConfig to freeze alongside the weights.
                If None, no companion config.json is written.
        force: Overwrite existing output even if hashes match.
        dtype: MLX output dtype — one of ``"float32"`` (default), ``"float16"``,
               or ``"bfloat16"``. Mel-Band-RoFormer separation parity holds
               down to bf16 for vocal stems; fp16 has marginally less dynamic
               range. Use bf16 for HuggingFace redistribution (smaller files,
               wider range than fp16).
        print_fn: Callable for diagnostics (defaults to builtin print).

    Returns:
        (weights_path, config_path) tuple of written file paths.
        config_path is None if no config was provided.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # License hint
    hint = _detect_license(input_path)
    if hint:
        substring, preset, license_tag, note = hint
        print_fn(f"ℹ License hint: input path contains '{substring}'")
        print_fn(f"  Likely weight license: {license_tag}")
        if preset:
            print_fn(f"  Suggested preset: MelRoFormerConfig.{preset}()")
        print_fn(f"  Note: {note}")
        print_fn("")

    # Content-addressed output path
    print_fn(f"Hashing input file: {input_path}")
    digest = _sha256_of_file(input_path)
    base = _content_addressed_name(input_path, digest, dtype)
    weights_path = output_dir / f"{base}.safetensors"
    config_path = output_dir / f"{base}.config.json"
    print_fn(f"Output dtype: {dtype}")

    if weights_path.exists() and not force:
        print_fn(f"✓ Output already exists: {weights_path}")
        print_fn(f"  (pass --force to re-convert)")
        existing_config = config_path if config_path.exists() else None
        return weights_path, existing_config

    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for checkpoint conversion. "
            "Install with `pip install torch`."
        )

    try:
        import mlx.core as mx
    except ImportError:
        raise ImportError("MLX is required. Install with `pip install mlx`.")

    import numpy as np

    print_fn(f"Loading PyTorch checkpoint: {input_path}")

    if input_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        raw = load_file(str(input_path))
    else:
        raw = torch.load(str(input_path), map_location="cpu", weights_only=False)

    state_dict = _extract_state_dict(raw)
    print_fn(f"  Found {len(state_dict)} keys in checkpoint")

    # Strip training state
    filtered = {k: v for k, v in state_dict.items() if not _should_strip(k)}
    stripped = len(state_dict) - len(filtered)
    if stripped > 0:
        print_fn(f"  Stripped {stripped} training-state keys")

    # Convert + split QKV
    mlx_weights: Dict[str, mx.array] = {}
    qkv_splits = 0
    mlx_dtype = _resolve_dtype(dtype)

    for key, tensor in filtered.items():
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().float().numpy()
        else:
            arr = np.asarray(tensor, dtype=np.float32)

        if key.endswith("to_qkv.weight"):
            qkv_splits += 1
            prefix = key[: -len("to_qkv.weight")]
            third = arr.shape[0] // 3
            mlx_weights[f"{prefix}to_q.weight"] = mx.array(arr[:third]).astype(
                mlx_dtype
            )
            mlx_weights[f"{prefix}to_k.weight"] = mx.array(
                arr[third : 2 * third]
            ).astype(mlx_dtype)
            mlx_weights[f"{prefix}to_v.weight"] = mx.array(arr[2 * third :]).astype(
                mlx_dtype
            )
        else:
            mlx_weights[key] = mx.array(arr).astype(mlx_dtype)

    if qkv_splits > 0:
        print_fn(
            f"  Split {qkv_splits} packed QKV projections → {qkv_splits * 3} separate weights"
        )

    print_fn(f"  Total MLX keys: {len(mlx_weights)}")

    # Write safetensors
    mx.save_safetensors(str(weights_path), mlx_weights)
    size_mb = weights_path.stat().st_size / (1024 * 1024)
    print_fn(f"✓ Wrote {weights_path}")
    print_fn(f"  Size: {size_mb:.1f} MB")

    # Write companion config.json if provided
    written_config_path: Optional[Path] = None
    if config is not None:
        config_dict = _config_to_dict(config)
        config_dict["_source_input"] = str(input_path.name)
        config_dict["_source_sha256"] = digest
        config_dict["_dtype"] = dtype
        config_path.write_text(json.dumps(config_dict, indent=2))
        print_fn(f"✓ Wrote {config_path}")
        written_config_path = config_path

    return weights_path, written_config_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Mel-Band-RoFormer checkpoints to MLX format."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint (.ckpt, .pt, or .safetensors)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted weights",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=[
            "kim_vocal_2",
            "viperx_vocals",
            "zfturbo_bs_roformer",
            "zfturbo_vocals_v1",
        ],
        help="Architecture preset. If omitted, no companion config.json is written.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Override: transformer depth (for custom checkpoints)",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=None,
        help="Override: number of mel bands (for custom checkpoints)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-convert even if output exists"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help=(
            "Output MLX dtype (default: float32). Use 'bfloat16' for "
            "HuggingFace redistribution — smaller file, wider dynamic range "
            "than fp16, parity-tested for vocal separation."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)

    # Resolve config
    config: Optional[MelRoFormerConfig] = None
    if args.preset:
        config = _resolve_preset(args.preset)
        if args.depth is not None:
            config.depth = args.depth
        if args.num_bands is not None:
            config.num_bands = args.num_bands
    elif args.depth is not None or args.num_bands is not None:
        config = MelRoFormerConfig.custom(
            depth=args.depth or 6,
            num_bands=args.num_bands or 60,
        )

    try:
        convert_checkpoint(
            input_path,
            output_dir,
            config=config,
            force=args.force,
            dtype=args.dtype,
        )
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
