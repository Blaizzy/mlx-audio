#!/usr/bin/env python
"""Basic MOSS-TTS usage for Local/Delay checkpoints.

This example demonstrates:
- canonical model loading through `mlx_audio.tts.generate.generate_audio`
- variant presets (`moss_tts_local`, `moss_tts`)
- `tokens` / `duration_s` control
- input representation (`text`, `pinyin`, `ipa`)
- optional streaming output
- Local-only `n_vq_for_inference` override

Usage:
    uv run python examples/moss_tts_basic.py
    uv run python examples/moss_tts_basic.py --model OpenMOSS-Team/MOSS-TTS --preset moss_tts
    uv run python examples/moss_tts_basic.py --input-type pinyin --text "ni hao, huan ying shi yong MOSS"
    uv run python examples/moss_tts_basic.py --stream --streaming-interval 0.5
"""

from __future__ import annotations

import argparse

from mlx_audio.tts.generate import generate_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic MOSS-TTS generation example")
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        help="MOSS model id or local path",
    )
    parser.add_argument(
        "--text",
        default="Hello from the MOSS-TTS MLX runtime.",
        help="Input text",
    )
    parser.add_argument(
        "--preset",
        default="moss_tts_local",
        help="Sampling preset (e.g. moss_tts_local, moss_tts)",
    )
    parser.add_argument(
        "--input-type",
        choices=["text", "pinyin", "ipa"],
        default="text",
        help="Input representation for supported models",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help="Optional instruction/style hint",
    )
    parser.add_argument(
        "--quality",
        default="balanced",
        help="Quality hint (draft/balanced/high/max/custom:...)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="Target token budget (takes precedence over --duration-s)",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=8.0,
        help="Duration hint in seconds when --tokens is not provided",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=240,
        help="Safety cap for generation loop",
    )
    parser.add_argument(
        "--n-vq-for-inference",
        type=int,
        default=None,
        help="Local-only depth override (1..n_vq)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Emit and save stream chunks instead of one merged clip",
    )
    parser.add_argument(
        "--streaming-interval",
        type=float,
        default=0.5,
        help="Chunk interval in seconds for streaming mode",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/moss_tts_basic",
        help="Directory for generated audio",
    )
    parser.add_argument(
        "--file-prefix",
        default="basic",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print runtime stats for each generated segment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_audio(
        text=args.text,
        model=args.model,
        preset=args.preset,
        input_type=args.input_type,
        language=args.language,
        instruct=args.instruct,
        quality=args.quality,
        tokens=args.tokens,
        duration_s=args.duration_s,
        max_tokens=args.max_tokens,
        n_vq_for_inference=args.n_vq_for_inference,
        stream=args.stream,
        streaming_interval=args.streaming_interval,
        output_path=args.output_dir,
        file_prefix=args.file_prefix,
        verbose=args.verbose,
        play=False,
    )


if __name__ == "__main__":
    main()
