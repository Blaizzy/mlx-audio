#!/usr/bin/env python
"""Multi-speaker dialogue generation with MOSS-TTSD.

This example demonstrates `dialogue_speakers` schema usage and speaker-tagged
prompting (`[S1]`, `[S2]`, ...).

Usage:
    uv run python examples/moss_ttsd_dialogue.py
    uv run python examples/moss_ttsd_dialogue.py --dialogue-speakers-json /path/to/speakers.json
    uv run python examples/moss_ttsd_dialogue.py --use-zero-based-speaker-ids
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_audio.tts.generate import generate_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-TTSD multi-speaker dialogue example")
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTSD-v1.0",
        help="TTSD model id or local path",
    )
    parser.add_argument(
        "--text",
        default=(
            "[S1] Thanks for joining this dialogue demo. "
            "[S2] Happy to help, we are validating speaker switching and continuity."
        ),
        help="Dialogue text with [S#] speaker tags",
    )
    parser.add_argument(
        "--dialogue-speakers-json",
        default=None,
        help="Optional path to speaker schema JSON list",
    )
    parser.add_argument(
        "--speaker1-ref-audio",
        default="REFERENCE/MOSS-Audio-Tokenizer/demo/demo_gt.wav",
        help="Fallback speaker 1 reference audio",
    )
    parser.add_argument(
        "--speaker1-ref-text",
        default="Speaker one reference prompt.",
        help="Fallback speaker 1 transcript",
    )
    parser.add_argument(
        "--speaker2-ref-audio",
        default="REFERENCE/MOSS-Audio-Tokenizer/demo/demo_gt.wav",
        help="Fallback speaker 2 reference audio",
    )
    parser.add_argument(
        "--speaker2-ref-text",
        default="Speaker two reference prompt.",
        help="Fallback speaker 2 transcript",
    )
    parser.add_argument(
        "--use-zero-based-speaker-ids",
        action="store_true",
        help="Emit fallback schema with IDs [0,1] instead of [1,2]",
    )
    parser.add_argument(
        "--preset",
        default="ttsd",
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
        "--tokens",
        type=int,
        default=200,
        help="Target token budget",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=320,
        help="Safety cap for generation",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/moss_ttsd_dialogue",
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--file-prefix",
        default="dialogue",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print runtime stats",
    )
    return parser.parse_args()


def build_fallback_schema(args: argparse.Namespace, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    first_id = 0 if args.use_zero_based_speaker_ids else 1
    second_id = first_id + 1

    schema = [
        {
            "speaker_id": first_id,
            "ref_audio": args.speaker1_ref_audio,
            "ref_text": args.speaker1_ref_text,
        },
        {
            "speaker_id": second_id,
            "ref_audio": args.speaker2_ref_audio,
            "ref_text": args.speaker2_ref_text,
        },
    ]

    schema_path = output_dir / "dialogue_speakers_demo.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    dialogue_speakers_json = args.dialogue_speakers_json
    if dialogue_speakers_json is None:
        dialogue_speakers_json = str(build_fallback_schema(args, output_dir))
        print(f"Wrote fallback dialogue schema: {dialogue_speakers_json}")

    generate_audio(
        text=args.text,
        model=args.model,
        preset=args.preset,
        input_type=args.input_type,
        language=args.language,
        dialogue_speakers_json=dialogue_speakers_json,
        tokens=args.tokens,
        max_tokens=args.max_tokens,
        output_path=str(output_dir),
        file_prefix=args.file_prefix,
        verbose=args.verbose,
        play=False,
    )


if __name__ == "__main__":
    main()
