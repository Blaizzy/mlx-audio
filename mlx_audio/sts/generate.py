"""Generate enhanced audio using speech-to-speech models.

Usage:
    python -m mlx_audio.sts.generate --model mlx-community/DeepFilterNet-mlx --audio noisy.wav
    python -m mlx_audio.sts.generate --model mlx-community/DeepFilterNet-mlx --audio noisy.wav --version 2
    python -m mlx_audio.sts.generate --model mlx-community/DeepFilterNet-mlx --audio noisy.wav --stream
    python -m mlx_audio.sts.generate --model starkdmi/MossFormer2_SE_48K_MLX --audio noisy.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# Repo ID substrings to model type mapping
REPO_HINTS = {
    "deepfilter": "deepfilternet",
    "mossformer": "mossformer2",
}


def _detect_model_type(model_name: str) -> str:
    """Detect model type from repo ID or path name."""
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

    return parser.parse_args()


def main():
    args = parse_args()

    in_path = Path(args.audio).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {in_path}")

    if args.output_path:
        out_path = Path(args.output_path).expanduser().resolve()
    else:
        out_path = in_path.with_stem(in_path.stem + "_enhanced")

    model_type = _detect_model_type(args.model)

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

    elif model_type == "mossformer2":
        from mlx_audio import audio_io
        from mlx_audio.sts.models.mossformer2_se import MossFormer2SEModel

        model = MossFormer2SEModel.from_pretrained(args.model)
        enhanced = model.enhance(str(in_path))
        audio_io.write(str(out_path), enhanced, model.config.sample_rate)
        mode = "offline"

    elapsed = time.time() - start

    if args.verbose:
        print(f"Mode:   {mode}")
        print(f"Time:   {elapsed:.2f}s")

    print(f"Saved:  {out_path}")


if __name__ == "__main__":
    main()
