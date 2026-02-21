#!/usr/bin/env python
"""Voice cloning example for MOSS-TTS Local/Delay checkpoints.

This example demonstrates reference-conditioned generation using `ref_audio`
and `ref_text` with optional instruction/style control.

Usage:
    uv run python examples/moss_tts_voice_cloning.py
    uv run python examples/moss_tts_voice_cloning.py --model OpenMOSS-Team/MOSS-TTS --preset moss_tts
    uv run python examples/moss_tts_voice_cloning.py --ref-audio /path/to/ref.wav --ref-text "Reference transcript"
"""

from __future__ import annotations

import argparse

from mlx_audio.tts.generate import generate_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-TTS voice cloning example")
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        help="MOSS model id or local path",
    )
    parser.add_argument(
        "--preset",
        default="moss_tts_local",
        help="Sampling preset (typically moss_tts_local or moss_tts)",
    )
    parser.add_argument(
        "--text",
        default="This sample clones the style of the reference recording.",
        help="Target text to synthesize",
    )
    parser.add_argument(
        "--ref-audio",
        default="REFERENCE/MOSS-Audio-Tokenizer/demo/demo_gt.wav",
        help="Reference audio path",
    )
    parser.add_argument(
        "--ref-text",
        default="Demo reference transcript.",
        help="Transcript/caption for the reference audio",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help="Optional style instruction",
    )
    parser.add_argument(
        "--input-type",
        choices=["text", "pinyin", "ipa"],
        default="text",
        help="Input representation",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=180,
        help="Target token budget",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=280,
        help="Safety cap for generation",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Emit and persist streaming chunks",
    )
    parser.add_argument(
        "--streaming-interval",
        type=float,
        default=0.5,
        help="Chunk interval for streaming mode",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/moss_tts_voice_cloning",
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--file-prefix",
        default="voice_clone",
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
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        instruct=args.instruct,
        tokens=args.tokens,
        max_tokens=args.max_tokens,
        stream=args.stream,
        streaming_interval=args.streaming_interval,
        output_path=args.output_dir,
        file_prefix=args.file_prefix,
        verbose=args.verbose,
        play=False,
    )


if __name__ == "__main__":
    main()
