"""Generate speech, text, or enhanced audio with the standard STS interface.

Models implement ``generate(audio, **kwargs)`` and return a result with
``audio``, ``sample_rate``, and optionally ``text`` or named ``audios``.
Loading and task-specific processing belong to the model packages.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

from mlx_audio import audio_io
from mlx_audio.sts.utils import load_model


def generate_audio(
    model=None,
    audio=None,
    output_path=None,
    verbose=False,
    gen_kwargs=None,
    model_kwargs=None,
    **kwargs,
):
    """Run STS generation with a model ID, local path, or loaded model.

    ``kwargs`` and ``gen_kwargs`` are passed to ``model.generate``; values in
    ``gen_kwargs`` take precedence. ``model_kwargs`` are passed to the loader.
    Omitted options use the model's defaults. The original result is returned.

    Audio is saved to ``output_path`` and text to a sibling ``.txt`` file.
    Named audio outputs are saved as ``<stem>_<name>.wav``. Without an explicit
    path, use ``<input stem>_output.wav`` or ``sts_output.wav`` for text input;
    models may supply an ``output_suffix`` for their default audio filename.
    """
    if model is None:
        raise ValueError("Model path or model instance must be provided.")
    if isinstance(model, (str, Path)):
        model = load_model(model, **(model_kwargs or {}))
    elif model_kwargs:
        raise ValueError("model_kwargs require a model ID or path")

    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise TypeError("This STS model does not implement generate(audio, **kwargs)")
    options = {**kwargs, **(gen_kwargs or {})}
    if (
        "stream" in options
        and "stream" not in inspect.signature(generate).parameters
        and options.pop("stream")
    ):
        raise ValueError("This STS model does not support streaming generation")

    started = time.perf_counter()
    result = generate(audio, **options)
    if not hasattr(result, "audio") or not hasattr(result, "sample_rate"):
        raise TypeError(
            "STS generate() must return a result with audio and sample_rate attributes"
        )

    if output_path is None:
        if isinstance(audio, (str, Path)):
            source = Path(audio).expanduser()
            suffix = getattr(model, "output_suffix", "_output")
            output_path = source.with_name(source.stem + suffix + ".wav")
        else:
            output_path = "sts_output.wav"
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if result.audio is not None:
        audio_io.write(str(output), result.audio, result.sample_rate)
        print(f"Saved: {output}")
    for name, waveform in getattr(result, "audios", {}).items():
        path = output.with_name(f"{output.stem}_{name}.wav")
        audio_io.write(str(path), waveform, result.sample_rate)
        print(f"Saved: {path}")
    text = getattr(result, "text", None)
    if text is not None:
        print(text)
        text_path = output.with_suffix(".txt")
        text_path.write_text(text + "\n", encoding="utf-8")
        print(f"Saved: {text_path}")
    if verbose:
        reasoning = getattr(result, "reasoning", None)
        if reasoning:
            print(f"Reasoning: {reasoning}")
        print(f"Time: {time.perf_counter() - started:.2f}s")
        tokens = getattr(result, "generation_tokens", None)
        if tokens is not None:
            print(f"Generated {tokens} tokens ({getattr(result, 'finish_reason', '')})")
    return result


def _json_object(value):
    result = json.loads(value)
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate speech, text, or enhanced audio using STS models",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--model", default="mlx-community/DeepFilterNet-mlx")
    parser.add_argument("--audio", help="Path to input audio")
    parser.add_argument(
        "--output-path", help="Output audio path; text is saved to a sibling .txt file"
    )
    parser.add_argument("--verbose", action="store_true")

    inputs = parser.add_argument_group("Input options")
    inputs.add_argument("--task", help="Generation task supported by the model")
    inputs.add_argument("--text")
    inputs.add_argument("--system-prompt")
    inputs.add_argument("--messages", help="JSON file containing dialogue messages")
    inputs.add_argument("--segments", help="JSON file containing completion segments")
    inputs.add_argument(
        "--prompt-examples", help="JSON file containing few-shot examples"
    )
    inputs.add_argument("--instruction", help="Base few-shot task instruction")
    inputs.add_argument("--instruct")
    inputs.add_argument("--thinking", action="store_true")
    inputs.add_argument(
        "--natural-instruction", dest="read_text_only", action="store_false"
    )
    inputs.add_argument("--sample-rate", type=int, help="Sample rate for audio arrays")

    generation = parser.add_argument_group("Generation options")
    generation.add_argument("--max-new-tokens", type=int)
    generation.add_argument("--temperature", type=float)
    generation.add_argument("--audio-temperature", type=float)
    generation.add_argument("--top-p", type=float)
    generation.add_argument("--top-k", type=int)
    generation.add_argument("--seed", type=int)
    generation.add_argument("--no-repeat-ngram-size", type=int)
    generation.add_argument("--stream", action="store_true")
    generation.add_argument(
        "--gen-kwargs", type=_json_object, help="Generate kwargs as JSON"
    )

    speech = parser.add_argument_group("Speech output options")
    speech.add_argument("--voice")
    speech.add_argument("--ref-audio")
    speech.add_argument("--ref-text")
    speech.add_argument("--max-talker-steps", type=int)
    speech.add_argument("--audio-tokenizer-path")

    separation = parser.add_argument_group("Separation options")
    separation.add_argument("--num-steps", type=int)
    separation.add_argument("--chunk-seconds", type=float)
    separation.add_argument("--overlap-seconds", type=float)

    loading = parser.add_argument_group("Loading options")
    loading.add_argument("--version", type=int, help="Checkpoint version, if supported")
    loading.add_argument("--subfolder", help="Subfolder within the model repo")
    loading.add_argument(
        "--model-kwargs", type=_json_object, help="Loader kwargs as JSON"
    )
    return parser.parse_args(argv)


def main(argv=None):
    options = vars(parse_args(argv))
    model_kwargs = options.pop("model_kwargs", {})
    for name in ("version", "subfolder"):
        if name in options:
            model_kwargs[name] = options.pop(name)
    for name in ("messages", "segments", "prompt_examples"):
        if name in options:
            options[name] = json.loads(Path(options[name]).expanduser().read_text())
    generate_audio(**options, model_kwargs=model_kwargs)


if __name__ == "__main__":
    main()
