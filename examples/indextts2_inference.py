#!/usr/bin/env python
"""Example: IndexTTS2 inference with quality-focused settings."""

import argparse
from pathlib import Path

import numpy as np
from mlx_audio.audio_io import sf_write
from mlx_audio.tts import load


def main():
    parser = argparse.ArgumentParser(description="Run IndexTTS2 TTS inference")
    parser.add_argument(
        "--model",
        default="indextts2_mlx",
        help="Path to converted IndexTTS2 MLX model directory",
    )
    parser.add_argument(
        "--ref-audio",
        default="examples/bible-audiobook/audios/bible-akjv/af_heart/00000001-Genesis-1:1.wav",
        help="Reference speaker audio path",
    )
    parser.add_argument(
        "--text",
        default="In the beginning, God created the heavens and the earth.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--out",
        default="indextts2_out.wav",
        help="Output WAV path",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=50,
        help="Higher values are slower but often clearer (40-60 recommended)",
    )
    parser.add_argument(
        "--diffusion-cfg-rate",
        type=float,
        default=0.7,
        help="Classifier-free guidance rate for s2mel diffusion",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=10.0,
        help="AR repetition penalty for semantic token decoding",
    )
    args = parser.parse_args()

    model = load(Path(args.model), strict=True)

    result = next(
        model.generate(
            args.text,
            ref_audio=args.ref_audio,
            diffusion_steps=args.diffusion_steps,
            diffusion_cfg_rate=args.diffusion_cfg_rate,
            repetition_penalty=args.repetition_penalty,
        )
    )

    audio = np.array(result.audio, dtype=np.float32)
    sf_write(args.out, audio, result.sample_rate)

    print(f"Saved: {args.out}")
    print(f"Sample rate: {result.sample_rate}")
    print(f"Audio duration: {result.audio_duration}")
    print(f"RTF: {result.real_time_factor:.4f}")


if __name__ == "__main__":
    main()
