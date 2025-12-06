import argparse
import contextlib
import json
import os
import time
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce

from mlx_audio.stt.utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate transcriptions from audio files"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo",
        type=str,
        help="Path to the model",
    )
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
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
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


# A stream on the default device just for generation
generation_stream = mx.new_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        try:
            yield
        finally:
            pass
    else:
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
        )
        max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
        if model_bytes > 0.9 * max_rec_size:
            model_mb = model_bytes // 2**20
            max_rec_mb = max_rec_size // 2**20
            print(
                f"[WARNING] Generating with a model that requires {model_mb} MB "
                f"which is close to the maximum recommended size of {max_rec_mb} "
                "MB. This can be slow. See the documentation for possible work-arounds: "
                "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
            )
        old_limit = mx.set_wired_limit(max_rec_size)
        try:
            yield
        finally:
            if streams is not None:
                for s in streams:
                    mx.synchronize(s)
            else:
                mx.synchronize()
            mx.set_wired_limit(old_limit)


def generate(
    model: Optional[Union[str, nn.Module]] = None,
    audio_path: str = "",
    output_path: str = "",
    format: str = "txt",
    verbose: bool = True,
    **kwargs,
):
    """Generate transcriptions from audio files.

    Args:
        model: Path to the model or the model instance.
        audio_path: Path to the audio file.
        output_path: Path to save the output.
        format: Output format (txt, srt, vtt, or json).
        verbose: Verbose output.
        **kwargs: Additional arguments for the model's generate method.

    Returns:
        segments: The generated transcription segments.
    """
    if model is None:
        raise ValueError("Model path or model instance must be provided.")

    if isinstance(model, str):
        # Load model
        model = load_model(model)

    print("=" * 10)
    print(f"\033[94mAudio path:\033[0m {audio_path}")
    print(f"\033[94mOutput path:\033[0m {output_path}")
    print(f"\033[94mFormat:\033[0m {format}")
    mx.reset_peak_memory()
    start_time = time.time()
    if verbose:
        print("\033[94mTranscription:\033[0m")
    segments = model.generate(
        audio_path, verbose=verbose, generation_stream=generation_stream, **kwargs
    )
    end_time = time.time()

    print("\n" + "=" * 10)
    print(f"\033[94mProcessing time:\033[0m {end_time - start_time:.2f} seconds")
    print(f"\033[94mPeak memory:\033[0m {mx.get_peak_memory() / 1e9:.2f} GB")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if format == "txt" or segments.segments is None:
        save_as_txt(segments, output_path)
        if segments.segments is None and format != "txt":
            print(
                f"\033[93m[WARNING]\033[0m No segments found for '{format}' format, saving as txt only"
            )
    elif format == "srt":
        save_as_srt(segments, output_path)
    elif format == "vtt":
        save_as_vtt(segments, output_path)
    elif format == "json":
        save_as_json(segments, output_path)

    return segments


def main():
    args = parse_args()
    generate(
        args.model,
        args.audio,
        args.output,
        args.format,
        args.verbose,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
