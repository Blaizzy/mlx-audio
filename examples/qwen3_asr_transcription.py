#!/usr/bin/env python
"""Example: Qwen3-ASR transcription with forced alignment.

This example demonstrates a two-pass workflow:
1. Transcribe audio using Qwen3-ASR
2. Run forced alignment using Qwen3-ForcedAligner to get word-level timestamps

Usage:
    python examples/qwen3_asr_transcription.py --audio path/to/audio.wav
    python examples/qwen3_asr_transcription.py --audio path/to/audio.wav --language Chinese
    python examples/qwen3_asr_transcription.py --audio path/to/audio.wav --output-path ./output
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR transcription with word-level forced alignment"
    )
    parser.add_argument(
        "--audio", "-a", required=True, help="Path to audio file to transcribe"
    )
    parser.add_argument(
        "--asr-model",
        default="mlx-community/Qwen3-ASR-0.6B-8bit",
        help="Qwen3-ASR model for transcription (default: mlx-community/Qwen3-ASR-0.6B-8bit)",
    )
    parser.add_argument(
        "--aligner-model",
        default="mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
        help="Qwen3-ForcedAligner model for alignment (default: mlx-community/Qwen3-ForcedAligner-0.6B-8bit)",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="English",
        help="Language name (e.g., 'English', 'Chinese', 'Japanese', 'Korean'). Default: English",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        default=None,
        help="Directory to save output JSON files. If not specified, only prints to console.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed progress"
    )
    args = parser.parse_args()

    from mlx_audio.stt.utils import load_model

    # Step 1: Transcribe audio with Qwen3-ASR
    print(f"Loading ASR model: {args.asr_model}")
    asr_model = load_model(args.asr_model)

    print(f"Transcribing: {args.audio}")
    print("-" * 50)

    asr_result = asr_model.generate(
        args.audio,
        language=args.language,
        verbose=args.verbose,
    )

    print(f"Transcription: {asr_result.text}")
    print("-" * 50)

    # Save ASR result if output path specified
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        asr_output_file = os.path.join(args.output_path, "transcription.json")
        with open(asr_output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "text": asr_result.text,
                    "segments": asr_result.segments,
                    "total_time": asr_result.total_time,
                    "prompt_tokens": asr_result.prompt_tokens,
                    "generation_tokens": asr_result.generation_tokens,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved transcription to: {asr_output_file}")

    # Step 2: Run forced alignment with Qwen3-ForcedAligner
    print(f"\nLoading aligner model: {args.aligner_model}")
    aligner_model = load_model(args.aligner_model)

    print("Running forced alignment...")
    print("-" * 50)

    alignment_result = aligner_model.generate(
        args.audio,
        text=asr_result.text,
        language=args.language,
    )

    # Print word-level timestamps
    print("Word-level alignment:")
    for item in alignment_result:
        print(f"  [{item.start_time:.3f}s - {item.end_time:.3f}s] {item.text}")

    # Save alignment result if output path specified
    if args.output_path:
        alignment_output_file = os.path.join(args.output_path, "alignment.json")
        with open(alignment_output_file, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "text": item.text,
                        "start_time": item.start_time,
                        "end_time": item.end_time,
                    }
                    for item in alignment_result
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nSaved alignment to: {alignment_output_file}")

    print("-" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
