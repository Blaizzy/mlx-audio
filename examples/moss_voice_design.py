#!/usr/bin/env python
"""Voice design with MOSS-Voice-Generator.

This example demonstrates description-first voice creation using `instruct`
(voice design prompt), plus optional language/quality controls.

Usage:
    uv run python examples/moss_voice_design.py
    uv run python examples/moss_voice_design.py --instruct "Energetic sports commentator"
    uv run python examples/moss_voice_design.py --text "ni hao" --input-type pinyin
"""

from __future__ import annotations

import argparse

from mlx_audio.tts.generate import generate_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-Voice-Generator example")
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-Voice-Generator",
        help="VoiceGenerator model id or local path",
    )
    parser.add_argument(
        "--text",
        default="Welcome to the voice design example running on MLX.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--instruct",
        default="A calm, clear studio narrator with warm tone.",
        help="Voice design instruction prompt",
    )
    parser.add_argument(
        "--preset",
        default="voice_generator",
        help="Sampling preset",
    )
    parser.add_argument(
        "--input-type",
        choices=["text", "pinyin", "ipa"],
        default="text",
        help="Input representation",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Optional language hint",
    )
    parser.add_argument(
        "--quality",
        default="high",
        help="Quality hint (draft/balanced/high/max/custom:...)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=180,
        help="Target token budget",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional duration hint in seconds",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Safety cap for generation",
    )
    parser.add_argument(
        "--normalize-inputs",
        action="store_true",
        help="Force text/instruction normalization before prompt packing",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/moss_voice_design",
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--file-prefix",
        default="voice_design",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print runtime stats",
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
        quality=args.quality,
        instruct=args.instruct,
        tokens=args.tokens,
        duration_s=args.duration_s,
        max_tokens=args.max_tokens,
        normalize_inputs=args.normalize_inputs,
        output_path=args.output_dir,
        file_prefix=args.file_prefix,
        verbose=args.verbose,
        play=False,
    )


if __name__ == "__main__":
    main()
