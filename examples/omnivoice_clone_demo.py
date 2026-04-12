#!/usr/bin/env python3
"""OmniVoice voice cloning demo with automatic reference transcription.

Usage:
    python examples/omnivoice_clone_demo.py --ref_audio reference.wav --text "Hello world."

This script shows the recommended two-step workflow:
1. Transcribe reference audio with Qwen3-ASR (or any mlx-audio STT model)
2. Generate speech with OmniVoice using both ref_audio and ref_text
"""

import argparse
import time

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="OmniVoice voice cloning demo")
    parser.add_argument(
        "--ref_audio", required=True, help="Path to reference audio (WAV)"
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", default="english", help="Target language")
    parser.add_argument(
        "--ref_text",
        default=None,
        help="Reference transcript (auto-transcribed if omitted)",
    )
    parser.add_argument("--output", default="clone_output.wav", help="Output WAV path")
    parser.add_argument(
        "--stt_model",
        default="mlx-community/Qwen3-ASR-small",
        help="STT model for auto-transcription",
    )
    parser.add_argument(
        "--tts_model", default="mlx-community/OmniVoice-bf16", help="TTS model"
    )
    parser.add_argument("--num_steps", type=int, default=32, help="Unmasking steps")
    parser.add_argument(
        "--guidance_scale", type=float, default=2.0, help="CFG guidance scale"
    )
    args = parser.parse_args()

    # Step 1: Get reference transcript
    if args.ref_text is None:
        print(f"Transcribing reference audio with {args.stt_model}...")
        from mlx_audio.stt.utils import load_model as load_stt

        stt = load_stt(args.stt_model)
        t0 = time.time()
        args.ref_text = stt.generate(args.ref_audio)
        print(f"  Transcript ({time.time() - t0:.1f}s): {args.ref_text}")
    else:
        print(f"Using provided ref_text: {args.ref_text}")

    # Step 2: Load TTS and generate
    print(f"Loading {args.tts_model}...")
    from mlx_audio.tts.utils import load_model as load_tts

    tts = load_tts(args.tts_model)

    print(f'Generating: "{args.text}" [{args.language}]')
    t0 = time.time()
    results = list(
        tts.generate(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )
    )
    elapsed = time.time() - t0

    audio = np.array(results[0].audio)
    sf.write(args.output, audio, results[0].sample_rate)
    print(
        f"Saved {args.output} ({results[0].audio_duration}, generated in {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
