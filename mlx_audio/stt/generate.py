import argparse
import contextlib
import json
import os
import time
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

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
            return


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


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    This is adapted from mlx-lm's generate_step for STT models that use
    input embeddings (audio + text).

    Args:
        prompt (mx.array): The input prompt token ids.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use ``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.
        prompt_progress_callback (Callable[[int, int], None]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.
        input_embeddings (mx.array, optional): Input embeddings to use instead of or in
          conjunction with prompt tokens. Default: ``None``.

    Yields:
        Tuple[mx.array, mx.array]: One token and a vector of log probabilities.
    """
    from mlx_lm.models import cache

    if input_embeddings is None and len(prompt) == 0:
        raise ValueError(
            "Either input_embeddings or prompt (or both) must be provided."
        )

    tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
        if input_embeddings is not None:
            return model(
                input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
            )
        else:
            return model(input_tokens, cache=prompt_cache)

    def _step(input_tokens: mx.array, input_embeddings: Optional[mx.array] = None):
        nonlocal tokens

        with mx.stream(generation_stream):
            logits = _model_call(
                input_tokens=input_tokens[None],
                input_embeddings=(
                    input_embeddings[None] if input_embeddings is not None else None
                ),
            )

            logits = logits[:, -1, :]

            if logits_processors and len(input_tokens) > 0:
                tokens = (
                    mx.concat([tokens, input_tokens])
                    if tokens is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(tokens, logits)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0)

    with mx.stream(generation_stream):
        total_prompt_tokens = (
            len(input_embeddings) if input_embeddings is not None else len(prompt)
        )
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            n_to_process = min(prefill_step_size, remaining)
            _model_call(
                input_tokens=prompt[:n_to_process][None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            mx.eval([c.state for c in prompt_cache])
            prompt_processed_tokens += n_to_process
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt = prompt[n_to_process:]
            input_embeddings = (
                input_embeddings[n_to_process:]
                if input_embeddings is not None
                else input_embeddings
            )
            mx.clear_cache()

        y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)

    mx.async_eval(y, logprobs)
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1



def generate_transcription(
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
    from .models.base import STTOutput

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

    if verbose:
        print("\n" + "=" * 10)
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

    is_plain_text_output = (not hasattr(segments, "segments") and not hasattr(segments, "sentences"))
    if format == "txt" or is_plain_text_output:
        if is_plain_text_output:
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
    generate_transcription(
        args.model,
        args.audio,
        args.output,
        args.format,
        args.verbose,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
