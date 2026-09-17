"""Generate speech, text, or enhanced audio using speech-to-speech models.

Usage:
    python -m mlx_audio.sts.generate --model mlx-community/DeepFilterNet-mlx --audio noisy.wav
    python -m mlx_audio.sts.generate --model mlx-community/DeepFilterNet-mlx --audio noisy.wav --version 2
    python -m mlx_audio.sts.generate --model mlx-community/DeepFilterNet-mlx --audio noisy.wav --stream
    python -m mlx_audio.sts.generate --model starkdmi/MossFormer2_SE_48K_MLX --audio noisy.wav
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Repo ID substrings to model type mapping
REPO_HINTS = {
    "dialoguesidon": "dialogue_sidon",
    "dialogue_sidon": "dialogue_sidon",
    "deepfilter": "deepfilternet",
    "mossformer": "mossformer2",
    "nemotronlabs-voicechat": "nemotron_voicechat",
    "nemotron_voicechat": "nemotron_voicechat",
}


def _detect_model_type(model_name: str) -> str:
    """Resolve known enhancement aliases or multimodal checkpoint metadata."""
    config_path = Path(model_name).expanduser() / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        if config.get("model_type") == "dialogue_sidon":
            return "dialogue_sidon"
    lower = model_name.lower()
    for hint, model_type in REPO_HINTS.items():
        if hint in lower:
            return model_type
    from mlx_audio.registry import model_type_from_config
    from mlx_audio.utils import load_config

    if model_type_from_config(load_config(model_name)) == "mimo_audio":
        return "mimo_audio"
    raise ValueError(
        f"Cannot detect model type from '{model_name}'. "
        f"Supported models: MiMo-Audio, {', '.join(REPO_HINTS.keys())}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech, text, or enhanced audio using STS models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/DeepFilterNet-mlx",
        help="HuggingFace repo ID or local path to the model",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to the input audio file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path (MiMo default: mimo_output.wav; enhancement: <input>_enhanced.wav)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt for conversational speech models",
    )
    mimo = parser.add_argument_group("MiMo-Audio offline options")
    mimo.add_argument(
        "--task",
        default="spoken_dialogue",
        choices=[
            "tts",
            "asr",
            "audio_understanding",
            "spoken_dialogue",
            "speech_to_text",
            "text",
            "few_shot",
            "continuation",
            "completion",
        ],
    )
    mimo.add_argument("--text")
    mimo.add_argument("--audio-tokenizer-path")
    mimo.add_argument("--ref-audio")
    mimo.add_argument("--instruct")
    mimo.add_argument("--instruction", help="Base few-shot task instruction")
    mimo.add_argument("--thinking", action="store_true")
    mimo.add_argument("--natural-instruction", action="store_true")
    mimo.add_argument("--messages", help="JSON file containing dialogue messages")
    mimo.add_argument(
        "--segments", help="JSON file containing Base completion segments"
    )
    mimo.add_argument(
        "--prompt-examples", help="JSON file containing Base few-shot examples"
    )
    mimo.add_argument("--max-new-tokens", type=int, default=2048)
    mimo.add_argument("--temperature", type=float)
    mimo.add_argument("--audio-temperature", type=float, default=0.9)

    # DeepFilterNet-specific options
    dfn = parser.add_argument_group("DeepFilterNet options")
    dfn.add_argument(
        "--version",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="DeepFilterNet version (1, 2, or 3). Default: 3",
    )
    dfn.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder within the model repo (e.g. v1, v2, v3)",
    )
    dfn.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming enhancement mode (DeepFilterNet v2/v3 only)",
    )

    sidon = parser.add_argument_group("DialogueSidon separation options")
    sidon.add_argument("--num-steps", type=int, default=30)
    sidon.add_argument("--seed", type=int, default=None)
    sidon.add_argument("--chunk-seconds", type=float, default=20.0)
    sidon.add_argument("--overlap-seconds", type=float, default=5.0)

    return parser.parse_args()


def main():
    args = parse_args()

    model_type = _detect_model_type(args.model)
    if model_type == "mimo_audio":
        _generate_mimo(args)
        return

    if args.audio is None:
        raise ValueError("--audio is required for this STS model")

    in_path = Path(args.audio).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {in_path}")

    if args.output_path:
        out_path = Path(args.output_path).expanduser().resolve()
    else:
        suffix = "_separated" if model_type == "dialogue_sidon" else "_enhanced"
        out_path = in_path.with_stem(in_path.stem + suffix)
    output_paths = [out_path]

    if args.verbose:
        print(f"Model:  {args.model}")
        print(f"Type:   {model_type}")
        print(f"Input:  {in_path}")
        print(f"Output: {out_path}")

    start = time.time()

    if model_type == "deepfilternet":
        from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

        load_kwargs = {"model_name_or_path": args.model}
        if args.version is not None:
            load_kwargs["version"] = args.version
        elif args.subfolder is not None:
            load_kwargs["subfolder"] = args.subfolder

        model = DeepFilterNetModel.from_pretrained(**load_kwargs)

        if args.stream:
            model.enhance_file_streaming(str(in_path), str(out_path))
            mode = "streaming"
        else:
            model.enhance_file(str(in_path), str(out_path))
            mode = "offline"

    elif model_type == "dialogue_sidon":
        from mlx_audio import audio_io
        from mlx_audio.sts import load

        model = load(args.model, strict=True)
        result = model.separate(
            str(in_path),
            num_steps=args.num_steps,
            seed=args.seed,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )
        output_paths = [
            out_path.with_name(f"{out_path.stem}_speaker_{i}.wav") for i in (1, 2)
        ]
        for path, speaker in zip(output_paths, result.speakers):
            path.parent.mkdir(parents=True, exist_ok=True)
            audio_io.write(str(path), speaker, result.sample_rate)
        mode = "offline separation"

    elif model_type == "mossformer2":
        from mlx_audio import audio_io
        from mlx_audio.sts.models.mossformer2_se import MossFormer2SEModel

        model = MossFormer2SEModel.from_pretrained(args.model)
        enhanced = model.enhance(str(in_path))
        audio_io.write(str(out_path), enhanced, model.config.sample_rate)
        mode = "offline"

    elif model_type == "nemotron_voicechat":
        from mlx_audio import audio_io
        from mlx_audio.sts import load

        model = load(args.model)
        generate_kwargs = {}
        if args.system_prompt is not None:
            generate_kwargs["system_prompt"] = args.system_prompt
        result = model.generate(str(in_path), **generate_kwargs)
        audio_io.write(str(out_path), result.audio, result.sample_rate)
        text_path = out_path.with_suffix(".txt")
        text_path.write_text(result.text + "\n", encoding="utf-8")
        print(result.text)
        mode = "offline"

    elapsed = time.time() - start

    if args.verbose:
        print(f"Mode:   {mode}")
        print(f"Time:   {elapsed:.2f}s")

    for path in output_paths:
        print(f"Saved:  {path}")


def _generate_mimo(args):
    from mlx_audio import audio_io
    from mlx_audio.sts.models.mimo_audio import Model

    if args.stream:
        raise ValueError("MiMo-Audio supports offline generation only")
    model = Model.from_pretrained(
        args.model, audio_tokenizer_path=args.audio_tokenizer_path
    )
    options = {
        "task": args.task,
        "text": args.text,
        "ref_audio": args.ref_audio,
        "instruct": args.instruct,
        "instruction": args.instruction,
        "thinking": args.thinking,
        "read_text_only": not args.natural_instruction,
        "system_prompt": args.system_prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "audio_temperature": args.audio_temperature,
    }
    for field in ("messages", "segments", "prompt_examples"):
        path = getattr(args, field)
        if path:
            options[field] = json.loads(Path(path).read_text())
    result = model.generate(args.audio, **options)
    print(result.text)
    output = Path(args.output_path or "mimo_output.wav").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if result.audio is not None:
        audio_io.write(str(output), result.audio, result.sample_rate)
        print(f"Saved: {output}")
    text_path = output.with_suffix(".txt")
    text_path.write_text(result.text + "\n", encoding="utf-8")
    if args.verbose:
        print(
            f"Generated {result.generation_tokens} positions in {result.processing_time:.2f}s ({result.finish_reason})"
        )
    print(f"Saved: {text_path}")


if __name__ == "__main__":
    main()
