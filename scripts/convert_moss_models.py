#!/usr/bin/env python
"""Batch converter for OpenMOSS models.

This script can convert any single MOSS model or the full MOSS catalog.

Supported catalog entries:
- `moss_tts`
- `moss_tts_local`
- `moss_ttsd`
- `moss_voice_generator`
- `moss_soundeffect`
- `moss_tts_realtime`
- `moss_audio_tokenizer`

Examples:
    # Convert one model
    uv run python scripts/convert_moss_models.py --models moss_tts_local

    # Convert two models with bf16 output
    uv run python scripts/convert_moss_models.py \
      --models moss_tts,moss_voice_generator --dtype bfloat16

    # Convert all MOSS models
    uv run python scripts/convert_moss_models.py --all

    # Convert all, quantizing TTS models to 4-bit (codec remains non-quantized)
    uv run python scripts/convert_moss_models.py --all --quantize --q-bits 4
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer
from mlx_audio.convert import MODEL_CONVERSION_DTYPES, QUANT_RECIPES
from mlx_audio.convert import convert as convert_generic_model
from mlx_audio.convert import get_model_path


@dataclass(frozen=True)
class MossModelSpec:
    key: str
    hf_id: str
    kind: str  # "tts" or "codec"
    aliases: tuple[str, ...] = ()


MOSS_CATALOG: Dict[str, MossModelSpec] = {
    "moss_tts": MossModelSpec(
        key="moss_tts",
        hf_id="OpenMOSS-Team/MOSS-TTS",
        kind="tts",
        aliases=("delay", "moss_tts_delay"),
    ),
    "moss_tts_local": MossModelSpec(
        key="moss_tts_local",
        hf_id="OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        kind="tts",
        aliases=("moss_tts_local_transformer", "local"),
    ),
    "moss_ttsd": MossModelSpec(
        key="moss_ttsd",
        hf_id="OpenMOSS-Team/MOSS-TTSD-v1.0",
        kind="tts",
        aliases=("ttsd",),
    ),
    "moss_voice_generator": MossModelSpec(
        key="moss_voice_generator",
        hf_id="OpenMOSS-Team/MOSS-Voice-Generator",
        kind="tts",
        aliases=(
            "voice_generator",
            "voicegenerator",
            "moss_voicegenerator",
            "moss-voicegenerator",
            "moss-voice-generator",
        ),
    ),
    "moss_soundeffect": MossModelSpec(
        key="moss_soundeffect",
        hf_id="OpenMOSS-Team/MOSS-SoundEffect",
        kind="tts",
        aliases=("soundeffect", "moss_sound_effect", "sound_effect"),
    ),
    "moss_tts_realtime": MossModelSpec(
        key="moss_tts_realtime",
        hf_id="OpenMOSS-Team/MOSS-TTS-Realtime",
        kind="tts",
        aliases=("realtime", "mossrealtime", "moss-tts-realtime"),
    ),
    "moss_audio_tokenizer": MossModelSpec(
        key="moss_audio_tokenizer",
        hf_id="OpenMOSS-Team/MOSS-Audio-Tokenizer",
        kind="codec",
        aliases=("audio_tokenizer", "moss_codec", "codec"),
    ),
}


def _normalize_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _build_lookup_table() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for key, spec in MOSS_CATALOG.items():
        lookup[_normalize_name(key)] = key
        lookup[_normalize_name(spec.hf_id)] = key
        for alias in spec.aliases:
            lookup[_normalize_name(alias)] = key
    return lookup


def _split_model_args(values: Sequence[str]) -> List[str]:
    entries: List[str] = []
    for raw in values:
        for part in str(raw).split(","):
            candidate = part.strip()
            if candidate:
                entries.append(candidate)
    return entries


def _resolve_selection(args: argparse.Namespace) -> List[MossModelSpec]:
    if not args.all and not args.models:
        raise ValueError("Specify --models or --all")

    lookup = _build_lookup_table()
    selected_keys: List[str] = []

    if args.all:
        selected_keys.extend(MOSS_CATALOG.keys())

    if args.models:
        for entry in _split_model_args(args.models):
            normalized = _normalize_name(entry)
            if normalized not in lookup:
                known = ", ".join(sorted(MOSS_CATALOG.keys()))
                raise ValueError(
                    f"Unknown model selection '{entry}'. Known keys: {known}"
                )
            selected_keys.append(lookup[normalized])

    deduped_keys: List[str] = []
    seen = set()
    for key in selected_keys:
        if key in seen:
            continue
        seen.add(key)
        deduped_keys.append(key)

    return [MOSS_CATALOG[key] for key in deduped_keys]


def _resolve_output_suffix(args: argparse.Namespace) -> str:
    suffix_parts: List[str] = []
    if args.quantize:
        suffix_parts.append(f"{args.q_bits}bit")
    elif args.dequantize:
        suffix_parts.append("dequantized")
    elif args.dtype:
        suffix_parts.append(args.dtype)

    if not suffix_parts:
        return ""
    return "-" + "-".join(suffix_parts)


def _prepare_output_dir(
    base_output: Path,
    spec: MossModelSpec,
    *,
    suffix: str,
    skip_existing: bool,
    overwrite: bool,
) -> tuple[Path, bool]:
    out_dir = base_output / f"{spec.key}{suffix}"

    if out_dir.exists() and any(out_dir.iterdir()):
        if skip_existing:
            return out_dir, True
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(
                f"Output directory exists and is not empty: {out_dir}. "
                "Use --skip-existing or --overwrite."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, False


def _convert_tts_model(
    spec: MossModelSpec,
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    convert_generic_model(
        hf_path=spec.hf_id,
        mlx_path=str(output_dir),
        quantize=bool(args.quantize),
        q_group_size=int(args.q_group_size),
        q_bits=int(args.q_bits),
        dtype=args.dtype,
        upload_repo=None,
        revision=args.revision,
        dequantize=bool(args.dequantize),
        quant_predicate=args.quant_predicate,
        model_domain="tts",
    )


def _copy_support_files(source_dir: Path, output_dir: Path) -> None:
    for pattern in ("*.json", "*.txt", "*.md", "*.py"):
        for src in source_dir.glob(pattern):
            if src.name in {"config.json", "model.safetensors"}:
                continue
            shutil.copy2(src, output_dir / src.name)


def _convert_moss_audio_tokenizer(
    spec: MossModelSpec,
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if args.dequantize:
        raise ValueError("--dequantize is not supported for moss_audio_tokenizer")
    if args.quantize:
        print(
            "[WARN] --quantize is not currently supported for moss_audio_tokenizer; "
            "exporting full-precision weights instead."
        )

    source_dir = get_model_path(spec.hf_id, revision=args.revision)
    model = MossAudioTokenizer.from_pretrained(source_dir, strict=True)

    if args.dtype:
        target_dtype = getattr(mx, args.dtype)
        casted = [
            (name, value.astype(target_dtype))
            for name, value in tree_flatten(model.parameters())
        ]
        model.load_weights(casted, strict=False)

    model.save_config(output_dir / "config.json")
    mx.save_safetensors(
        str(output_dir / "model.safetensors"),
        dict(tree_flatten(model.parameters())),
        metadata={"format": "mlx"},
    )
    _copy_support_files(source_dir, output_dir)


def _list_models() -> None:
    print("Available MOSS models:")
    for key, spec in MOSS_CATALOG.items():
        aliases = ", ".join(spec.aliases) if spec.aliases else "-"
        print(
            f"- {key:22s} | kind={spec.kind:5s} | hf_id={spec.hf_id} | aliases={aliases}"
        )


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert MOSS models to MLX format")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Model keys/aliases/HF ids to convert. Supports spaces and comma-separated "
            "values."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all known MOSS models in the catalog",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print model catalog and exit",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("converted") / "moss",
        help="Base output directory for converted models",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision for all selected models",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
        help="Target output dtype",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize TTS models during conversion",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Quantization group size",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        help="Quantization bits",
    )
    parser.add_argument(
        "--quant-predicate",
        type=str,
        choices=QUANT_RECIPES,
        default=None,
        help="Mixed-bit quantization recipe for TTS models",
    )
    parser.add_argument(
        "--dequantize",
        action="store_true",
        help="Dequantize selected models (TTS only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print conversion plan without executing",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose output directories already exist and are non-empty",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing per-model output directory before conversion",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining conversions when one model fails",
    )
    return parser


def main() -> None:
    parser = configure_parser()
    args = parser.parse_args()

    if args.list_models:
        _list_models()
        return

    selected_specs = _resolve_selection(args)
    suffix = _resolve_output_suffix(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("Conversion plan:")
    for spec in selected_specs:
        print(f"- {spec.key} ({spec.kind}) -> {spec.hf_id}")

    if args.dry_run:
        print("Dry run requested; exiting without conversion.")
        return

    successes: List[str] = []
    skipped: List[str] = []
    failures: List[tuple[str, str]] = []

    for spec in selected_specs:
        try:
            output_dir, was_skipped = _prepare_output_dir(
                args.output_root,
                spec,
                suffix=suffix,
                skip_existing=bool(args.skip_existing),
                overwrite=bool(args.overwrite),
            )
            if was_skipped:
                skipped.append(spec.key)
                print(f"[SKIP] {spec.key}: output already exists ({output_dir})")
                continue

            print(f"[RUN ] {spec.key} -> {output_dir}")
            if spec.kind == "tts":
                _convert_tts_model(spec, output_dir=output_dir, args=args)
            elif spec.kind == "codec":
                _convert_moss_audio_tokenizer(spec, output_dir=output_dir, args=args)
            else:
                raise ValueError(f"Unsupported model kind '{spec.kind}'")

            successes.append(spec.key)
            print(f"[ OK ] {spec.key}")
        except Exception as exc:
            failures.append((spec.key, str(exc)))
            print(f"[FAIL] {spec.key}: {exc}")
            if not args.continue_on_error:
                break

    print("\nSummary:")
    print(f"- Succeeded: {len(successes)}")
    if successes:
        print(f"  {', '.join(successes)}")
    print(f"- Skipped:   {len(skipped)}")
    if skipped:
        print(f"  {', '.join(skipped)}")
    print(f"- Failed:    {len(failures)}")
    for key, message in failures:
        print(f"  - {key}: {message}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
