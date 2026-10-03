"""Generate music from a caption and structured lyrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.audio_io import write as audio_write

from .utils import load_model


def generate_music(
    caption: str,
    lyrics: str,
    model: Union[str, Path, nn.Module],
    duration: Optional[float] = None,
    steps: int = 30,
    seed: int = 0,
    output_path: Union[str, Path] = "music.wav",
    verbose: bool = True,
    **kwargs: Any,
) -> Path:
    """Generate a song and write it to ``output_path``."""
    if isinstance(model, (str, Path)):
        model = load_model(model)

    generate_kwargs = {
        "text": caption,
        "lyrics": lyrics,
        "steps": steps,
        "seed": seed,
        **kwargs,
    }
    if duration is not None:
        generate_kwargs["duration"] = duration

    chunks = []
    sample_rate = None
    for result in model.generate(**generate_kwargs):
        if sample_rate is None:
            sample_rate = result.sample_rate
        elif result.sample_rate != sample_rate:
            raise ValueError("Music generation returned inconsistent sample rates")
        chunks.append(result.audio)

    if not chunks or sample_rate is None:
        raise RuntimeError("Music generation produced no audio")

    audio = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    audio_write(destination, audio, sample_rate)
    if verbose:
        print(f"Saved: {destination}")
    return destination


def configure_parser() -> argparse.ArgumentParser:
    """Build the music-generation command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate music from a caption and structured lyrics"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face repo ID or local model path",
    )
    parser.add_argument(
        "--caption",
        required=True,
        help="Music style, vocals, instrumentation, and arrangement description",
    )
    lyrics = parser.add_mutually_exclusive_group(required=True)
    lyrics.add_argument(
        "--lyrics",
        help="Structured lyrics with section tags on separate lines",
    )
    lyrics.add_argument(
        "--lyrics-file",
        type=Path,
        help="UTF-8 text file containing structured lyrics",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Maximum requested duration in seconds",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Flow-matching inference steps",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("music.wav"),
        help="Output audio path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show generation progress output",
    )
    return parser


def main() -> None:
    """Run music generation from command-line arguments."""
    args = configure_parser().parse_args()
    lyrics = (
        args.lyrics
        if args.lyrics is not None
        else args.lyrics_file.read_text(encoding="utf-8")
    )
    generate_music(
        caption=args.caption,
        lyrics=lyrics,
        model=args.model,
        duration=args.duration,
        steps=args.steps,
        seed=args.seed,
        output_path=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
