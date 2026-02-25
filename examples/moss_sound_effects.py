#!/usr/bin/env python
"""Sound-effect generation with MOSS-SoundEffect.

This example demonstrates ambient and event-conditioned generation for the
SoundEffect variant.

Usage:
    uv run python examples/moss_sound_effects.py
    uv run python examples/moss_sound_effects.py --ambient-sound "busy cafe" --sound-event "cups clinking"
"""

from __future__ import annotations

import argparse

from mlx_audio.tts.generate import generate_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-SoundEffect example")
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-SoundEffect",
        help="SoundEffect model id or local path",
    )
    parser.add_argument(
        "--ambient-sound",
        default="Thunder and rain over a city street.",
        help="Ambient scene description",
    )
    parser.add_argument(
        "--sound-event",
        default="storm",
        help="Primary sound event cue",
    )
    parser.add_argument(
        "--quality",
        default="high",
        help="Quality hint",
    )
    parser.add_argument(
        "--preset",
        default="soundeffect",
        help="Sampling preset",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=140,
        help="Target token budget",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=260,
        help="Safety cap for generation",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/moss_sound_effects",
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--file-prefix",
        default="sound_effect",
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
    # `text` is intentionally omitted: generate_audio() mirrors CLI behavior and
    # backfills `text` from `ambient_sound` when needed for SoundEffect prompts.
    generate_audio(
        text=None,
        model=args.model,
        preset=args.preset,
        ambient_sound=args.ambient_sound,
        sound_event=args.sound_event,
        quality=args.quality,
        tokens=args.tokens,
        max_tokens=args.max_tokens,
        output_path=args.output_dir,
        file_prefix=args.file_prefix,
        verbose=args.verbose,
        play=False,
    )


if __name__ == "__main__":
    main()
