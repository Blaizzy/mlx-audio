import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx

from mlx_audio.stt.utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate transcriptions from audio files"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to the audio file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the output"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="txt",
        choices=["txt", "srt", "vtt", "json"],
        help="Output format (txt, srt, vtt, or json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm format for SRT/VTT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def format_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for VTT"""
    return format_timestamp(seconds).replace(",", ".")


def save_as_txt(segments, output_path: str):
    with open(f"{output_path}.txt", "w", encoding="utf-8") as f:
        f.write(segments.text)


def save_as_srt(segments, output_path: str):
    with open(f"{output_path}.srt", "w", encoding="utf-8") as f:
        for i, sentence in enumerate(segments.sentences, 1):
            f.write(f"{i}\n")
            f.write(
                f"{format_timestamp(sentence.start)} --> {format_timestamp(sentence.end)}\n"
            )
            f.write(f"{sentence.text}\n\n")


def save_as_vtt(segments, output_path: str):
    with open(f"{output_path}.vtt", "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        if hasattr(segments, "sentences"):
            sentences = segments.sentences

            for i, sentence in enumerate(sentences, 1):
                f.write(f"{i}\n")
                f.write(
                    f"{format_vtt_timestamp(sentence.start)} --> {format_vtt_timestamp(sentence.end)}\n"
                )
                f.write(f"{sentence.text}\n\n")
        else:
            sentences = segments.segments
            for i, token in enumerate(sentences, 1):
                f.write(f"{i}\n")
                f.write(
                    f"{format_vtt_timestamp(token['start'])} --> {format_vtt_timestamp(token['end'])}\n"
                )
                f.write(f"{token['text']}\n\n")


def save_as_json(segments, output_path: str):
    if hasattr(segments, "sentences"):
        result = {
            "text": segments.text,
            "sentences": [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "duration": s.duration,
                    "tokens": [
                        {
                            "text": t.text,
                            "start": t.start,
                            "end": t.end,
                            "duration": t.duration,
                        }
                        for t in s.tokens
                    ],
                }
                for s in segments.sentences
            ],
        }
    else:
        result = {
            "text": segments.text,
            "segments": [
                {
                    "text": s["text"],
                    "start": s["start"],
                    "end": s["end"],
                    "duration": s["end"] - s["start"],
                }
                for s in segments.segments
            ],
        }

    with open(f"{output_path}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def generate(
    model_path: str,
    audio_path: str,
    output_path: str,
    format: str = "txt",
    verbose: bool = True,
):
    model = load_model(model_path)
    print(f"\n\033[94mModel:\033[0m {model_path}")
    print(f"\033[94mAudio path:\033[0m {audio_path}")
    print(f"\033[94mOutput path:\033[0m {output_path}")
    print(f"\033[94mFormat:\033[0m {format}")
    mx.reset_peak_memory()
    start_time = time.time()
    segments = model.generate(audio_path)
    end_time = time.time()

    if verbose:
        print("\n\033[94mTranscription:\033[0m")
        print(segments.text)
        print("\n\033[94mSegments:\033[0m")
        if hasattr(segments, "segments"):
            print(segments.segments)
        elif hasattr(segments, "tokens"):
            print(segments.tokens)
        else:
            print(segments)

    print(f"\033[94mProcessing time:\033[0m {end_time - start_time:.2f} seconds")
    print(f"\033[94mPeak memory:\033[0m {mx.get_peak_memory() / 1e9:.2f} GB")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if format == "txt":
        save_as_txt(segments, output_path)
    elif format == "srt":
        save_as_srt(segments, output_path)
    elif format == "vtt":
        save_as_vtt(segments, output_path)
    elif format == "json":
        save_as_json(segments, output_path)

    return segments


if __name__ == "__main__":
    args = parse_args()
    generate(args.model, args.audio, args.output, args.format, args.verbose)
