#!/usr/bin/env python
"""Deterministic one-shot MOSS-TTS generation examples.

Default mode uses a deterministic hybrid strategy:
- text/control channel: greedy decode (`do_sample=False`)
- audio channels: seeded sampling (`do_sample=True`)

This avoids a known full-greedy failure mode where some runs can collapse
to silent codec trajectories.

Usage:
    uv run python examples/moss_tts_deterministic.py
    uv run python examples/moss_tts_deterministic.py --with-reference
    uv run python examples/moss_tts_deterministic.py --model OpenMOSS-Team/MOSS-TTS --preset moss_tts
"""

from __future__ import annotations

import argparse
from typing import Any

import mlx.core as mx

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic MOSS-TTS example")
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        help="MOSS model id or local path",
    )
    parser.add_argument(
        "--preset",
        default="moss_tts_local",
        help="Sampling preset",
    )
    parser.add_argument(
        "--text",
        default="Hello what is happening this is from MOSS on MLX.",
        help="Input text",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help="Optional style instruction",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help=(
            "Optional target duration in seconds. Leave unset for natural stopping "
            "(recommended for short no-reference prompts)."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=260,
        help="Safety cap for generation",
    )
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="Enable reference-conditioned deterministic voice",
    )
    parser.add_argument(
        "--ref-audio",
        default="REFERENCE/MOSS-Audio-Tokenizer/demo/demo_gt.wav",
        help="Reference audio path when --with-reference is set",
    )
    parser.add_argument(
        "--ref-text",
        default="Demo reference transcript.",
        help="Reference transcript when --with-reference is set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic audio-channel sampling",
    )
    parser.add_argument(
        "--determinism-mode",
        choices=["hybrid", "full_greedy"],
        default="hybrid",
        help=(
            "Deterministic decode strategy: "
            "hybrid=text greedy + audio seeded sampling (recommended), "
            "full_greedy=greedy on all channels"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/moss_tts_deterministic",
        help="Output directory",
    )
    parser.add_argument(
        "--file-prefix",
        default="deterministic",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print runtime stats",
    )
    return parser.parse_args()


def _build_channel_sampling_flags(model: Any, determinism_mode: str) -> list[bool]:
    channels = int(getattr(getattr(model, "config", None), "channels", 0))
    if channels <= 0:
        raise RuntimeError("Unable to infer model channel count for deterministic mode")
    if determinism_mode == "full_greedy":
        return [False] * channels
    # Hybrid strategy: deterministic but avoids full-greedy silent collapse.
    return [False] + ([True] * (channels - 1))


def main() -> None:
    args = parse_args()
    mx.random.seed(int(args.seed))

    model = load_model(args.model)
    do_samples = _build_channel_sampling_flags(model, args.determinism_mode)

    ref_audio = args.ref_audio if args.with_reference else None
    ref_text = args.ref_text if args.with_reference else None
    file_prefix = (
        f"{args.file_prefix}_with_ref" if args.with_reference else args.file_prefix
    )

    print(f"[moss_tts_deterministic] determinism_mode={args.determinism_mode}")
    print(
        "[moss_tts_deterministic] channel sampling flags: "
        f"text={do_samples[0]}, audio_sampled={any(do_samples[1:])}"
    )
    if args.determinism_mode == "full_greedy":
        print(
            "[moss_tts_deterministic] warning: full_greedy may collapse to silence "
            "for some prompts/checkpoints"
        )
    if args.with_reference:
        print(
            f"[moss_tts_deterministic] using reference audio: {args.ref_audio} "
            f"(ref_text={args.ref_text})"
        )

    generation_kwargs = {
        "text": args.text,
        "model": model,
        "preset": args.preset,
        "instruct": args.instruct,
        "max_tokens": args.max_tokens,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "do_samples": do_samples,
        "output_path": args.output_dir,
        "file_prefix": file_prefix,
        "verbose": args.verbose,
        "play": False,
    }
    if args.duration_s is not None:
        generation_kwargs["duration_s"] = float(args.duration_s)

    generate_audio(
        **generation_kwargs,
    )


if __name__ == "__main__":
    main()
