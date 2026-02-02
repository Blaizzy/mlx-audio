import argparse
import contextlib
import inspect
import json
import os
import time
from pprint import pprint
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
        "--output-path", type=str, required=True, help="Path to save the output"
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
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (e.g. en, es, fr, de, etc.)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=30.0,
        help="Chunk duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--frame-threshold",
        type=int,
        default=25,
        help="Frame threshold (default: 25)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the transcription as it is generated (default: False)",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Context string with hotwords or metadata to guide transcription",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Prefill step size (default: 2048)",
    )
    parser.add_argument(
        "--gen-kwargs",
        type=json.loads,
        default=None,
        help="Additional generate kwargs as JSON (e.g. '{\"min_chunk_duration\": 1.0}')",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Text to align (for forced alignment models)",
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
        if hasattr(segments, "sentences"):
            # Parakeet model (AlignedResult)
            for i, sentence in enumerate(segments.sentences, 1):
                f.write(f"{i}\n")
                f.write(
                    f"{format_timestamp(sentence.start)} --> {format_timestamp(sentence.end)}\n"
                )
                f.write(f"{sentence.text}\n\n")
        else:
            # Whisper model
            for i, segment in enumerate(segments.segments, 1):
                f.write(f"{i}\n")
                f.write(
                    f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
                )
                f.write(f"{segment['text']}\n\n")


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
        # Add speaker_id only if it exists
        for i, s in enumerate(segments.sentences):
            if hasattr(s, "speaker_id"):
                result["sentences"][i]["speaker_id"] = s.speaker_id
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
        # Add speaker_id only if it exists
        for i, s in enumerate(segments.segments):
            if "speaker_id" in s:
                result["segments"][i]["speaker_id"] = s["speaker_id"]

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


def generate_transcription(
    model: Optional[Union[str, nn.Module]] = None,
    audio: Union[str, mx.array] = None,
    output_path: str = "transcript",
    format: str = "txt",
    verbose: bool = False,
    text: str = "",
    **kwargs,
):
    """Generate transcriptions from audio files.

    Args:
        model: Path to the model or the model instance.
        audio: Path to the audio file (str), or audio waveform (mx.array).
        output_path: Path to save the output.
        format: Output format (txt, srt, vtt, or json).
        verbose: Verbose output.
        text: Text to align (for forced alignment models).
        **kwargs: Additional arguments for the model's generate method.

    Returns:
        segments: The generated transcription segments.
    """
    from .models.base import STTOutput

    if model is None:
        raise ValueError("Model path or model instance must be provided.")

    if isinstance(model, str):
        # Load model
        model = load_model(model)

    mx.reset_peak_memory()
    start_time = time.time()
    if verbose:
        print("=" * 10)
        print(f"\033[94mAudio path:\033[0m {audio}")
        print(f"\033[94mOutput path:\033[0m {output_path}")
        print(f"\033[94mFormat:\033[0m {format}")

    # Handle gen_kwargs (additional generate parameters as JSON)
    gen_kwargs = kwargs.pop("gen_kwargs", None)
    if gen_kwargs:
        kwargs.update(gen_kwargs)

    # Add text to kwargs if provided (for forced alignment)
    if text:
        kwargs["text"] = text

    signature = inspect.signature(model.generate)
    kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}

    if kwargs.get("stream", False):
        all_segments = []
        accumulated_text = ""
        language = "en"
        prompt_tokens = 0
        generation_tokens = 0
        for result in model.generate(audio, verbose=verbose, **kwargs):
            segment_dict = {
                "text": result.text,
                "start": result.start_time,
                "end": result.end_time,
                "is_final": result.is_final,
            }

            all_segments.append(segment_dict)
            # Accumulate text (handles both incremental and cumulative streaming)
            accumulated_text += result.text
            language = result.language

            # Extract token counts from results (final result has cumulative totals)
            if hasattr(result, "prompt_tokens") and result.prompt_tokens > 0:
                prompt_tokens = result.prompt_tokens
            if hasattr(result, "generation_tokens") and result.generation_tokens > 0:
                generation_tokens = result.generation_tokens

        stream_end_time = time.time()
        stream_duration = stream_end_time - start_time
        segments = STTOutput(
            text=accumulated_text.strip(),
            segments=all_segments,
            language=language,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            total_tokens=prompt_tokens + generation_tokens,
            total_time=stream_duration,
            prompt_tps=prompt_tokens / stream_duration if stream_duration > 0 else 0,
            generation_tps=generation_tokens / stream_duration if stream_duration > 0 else 0,
        )
    else:
        segments = model.generate(
            audio, verbose=verbose, generation_stream=generation_stream, **kwargs
        )

    if verbose:
        if hasattr(segments, "text"):
            print("\033[94mTranscription:\033[0m\n")
            print(f"{segments.text[:500]}...\n")

        if hasattr(segments, "segments") and segments.segments is not None:
            print("\033[94mSegments:\033[0m\n")
            pprint(segments.segments[:3] + ["..."])

    end_time = time.time()

    if verbose:
        print("\n" + "=" * 10)
        print(f"\033[94mSaving file to:\033[0m ./{output_path}.{format}")
        print(f"\033[94mProcessing time:\033[0m {end_time - start_time:.2f} seconds")
        if isinstance(segments, STTOutput):
            print(
                f"\033[94mPrompt:\033[0m {segments.prompt_tokens} tokens, "
                f"{segments.prompt_tps:.3f} tokens-per-sec"
            )
            print(
                f"\033[94mGeneration:\033[0m {segments.generation_tokens} tokens, "
                f"{segments.generation_tps:.3f} tokens-per-sec"
            )
        print(f"\033[94mPeak memory:\033[0m {mx.get_peak_memory() / 1e9:.2f} GB")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Check for segments (Whisper) or sentences (Parakeet)
    has_segments = hasattr(segments, "segments") and segments.segments is not None
    has_sentences = hasattr(segments, "sentences") and segments.sentences is not None

    if format == "txt" or (not has_segments and not has_sentences):
        if not has_segments and not has_sentences:
            print("[WARNING] No segments found, saving as plain text.")
        save_as_txt(segments, output_path)
    elif format == "srt":
        save_as_srt(segments, output_path)
    elif format == "vtt":
        save_as_vtt(segments, output_path)
    elif format == "json":
        save_as_json(segments, output_path)

    return segments


def main():
    args = parse_args()
    generate_transcription(**vars(args))


if __name__ == "__main__":
    main()
