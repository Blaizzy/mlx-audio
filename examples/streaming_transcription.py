#!/usr/bin/env python
"""Example: AlignAtt streaming transcription with Whisper.

This example demonstrates low-latency streaming transcription using the
AlignAtt algorithm, which monitors cross-attention weights to determine
when decoded tokens are stable enough to emit.

Usage:
    python examples/streaming_transcription.py --audio path/to/audio.wav
    python examples/streaming_transcription.py --audio path/to/audio.wav --model mlx-community/whisper-small
    python examples/streaming_transcription.py --audio path/to/audio.wav --chunk-duration 0.5

Reference: https://arxiv.org/abs/2211.00895 (SimulMT with AlignAtt)
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser(
        description="Streaming transcription using AlignAtt algorithm"
    )
    parser.add_argument(
        "--audio", "-a", required=True, help="Path to audio file to transcribe"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="mlx-community/whisper-tiny-asr-fp16",
        help="Whisper model to use (default: mlx-community/whisper-tiny-asr-fp16)",
    )
    parser.add_argument(
        "--chunk-duration",
        "-c",
        type=float,
        default=1.0,
        help="Audio chunk duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--frame-threshold",
        "-t",
        type=int,
        default=25,
        help="AlignAtt frame threshold (default: 25, ~0.5s lookahead)",
    )
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="Language code (e.g., 'en', 'ja'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task: 'transcribe' (default) or 'translate' (to English).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show timing information"
    )
    args = parser.parse_args()

    from mlx_audio.stt.utils import load_model

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    print(f"Transcribing: {args.audio}")
    print(
        f"Chunk duration: {args.chunk_duration}s, Frame threshold: {args.frame_threshold}"
    )
    print("-" * 50)

    start_time = time.time()
    first_result_time = None
    full_text = []

    for result in model.generate_streaming(
        args.audio,
        chunk_duration=args.chunk_duration,
        frame_threshold=args.frame_threshold,
        language=args.language,
        task=args.task,
    ):
        if first_result_time is None:
            first_result_time = time.time() - start_time

        marker = "[FINAL]" if result.is_final else "[partial]"
        progress_pct = result.progress * 100

        if args.verbose:
            print(
                f"{marker} [{result.start_time:.2f}s - {result.end_time:.2f}s] "
                f"({progress_pct:.1f}% @ {result.audio_position:.1f}s/{result.audio_duration:.1f}s): "
                f"{result.text}"
            )
        else:
            print(f"{marker} ({progress_pct:.0f}%) {result.text}")

        full_text.append(result.text)

    total_time = time.time() - start_time

    print("-" * 50)
    print(f"Full transcription: {''.join(full_text)}")

    if args.verbose:
        print(f"\nTiming:")
        print(f"  Time to first result: {first_result_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")


if __name__ == "__main__":
    main()
