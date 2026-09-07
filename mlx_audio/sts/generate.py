"""Generate enhanced audio using speech-to-speech models.

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
    """Detect model type from repo ID or path name."""
    config_path = Path(model_name).expanduser() / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        if config.get("model_type") == "dialogue_sidon":
            return "dialogue_sidon"
    lower = model_name.lower()
    for hint, model_type in REPO_HINTS.items():
        if hint in lower:
            return model_type
    raise ValueError(
        f"Cannot detect model type from '{model_name}'. "
        f"Supported models: {', '.join(REPO_HINTS.keys())}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhance audio using speech-to-speech models"
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
        required=True,
        help="Path to the input audio file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output audio file path (default: <input>_enhanced.wav)",
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

    in_path = Path(args.audio).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {in_path}")

    model_type = _detect_model_type(args.model)
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


if __name__ == "__main__":
    main()
