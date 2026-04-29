#!/usr/bin/env python3
"""OmniVoice voice cloning demo with automatic reference transcription.

Usage:
    python examples/omnivoice_clone_demo.py --ref_audio reference.wav --text "Hello world."

Workflow (matches original k2-fsa/OmniVoice):
1. Preprocess reference audio (silence removal, RMS normalization)
2. Decode preprocessed tokens back to waveform
3. Transcribe the PREPROCESSED audio with STT (not the original file)
4. Generate with ref_tokens + ref_text from the same preprocessed source

The original Python OmniVoice runs Whisper on audio AFTER silence removal,
so ref_text always matches the actual reference content. This demo replicates
that approach using any mlx-audio STT model.
"""

import argparse
import os
import tempfile
import time

import numpy as np

from mlx_audio.audio_io import write as audio_write


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
        default="mlx-community/Qwen3-ASR-0.6B-8bit",
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

    import mlx.core as mx

    print(f"Loading {args.tts_model}...")
    from mlx_audio.tts.utils import load_model as load_tts

    tts = load_tts(args.tts_model)
    tokenizer = getattr(tts, "audio_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("TTS model has no audio_tokenizer.")

    from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt

    print("Preprocessing reference audio...")
    ref_tokens = create_voice_clone_prompt(args.ref_audio, tokenizer=tokenizer)
    mx.eval(ref_tokens)
    ref_duration_s = ref_tokens.shape[0] * 960 / 24000
    print(f"  {ref_tokens.shape[0]} frames ({ref_duration_s:.1f}s after preprocessing)")

    if args.ref_text is None:
        print(f"Transcribing preprocessed audio with {args.stt_model}...")
        preprocessed_audio = np.array(tokenizer.decode(ref_tokens).astype(mx.float32))

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_write(tmp.name, preprocessed_audio, 24000)
        tmp.close()

        from mlx_audio.stt.utils import load_model as load_stt

        stt = load_stt(args.stt_model)
        t0 = time.time()
        result = stt.generate(tmp.name)
        args.ref_text = result.text if hasattr(result, "text") else str(result)
        print(f"  Transcript ({time.time() - t0:.1f}s): {args.ref_text}")
        os.unlink(tmp.name)
    else:
        print(f"Using provided ref_text: {args.ref_text}")

    print(f'Generating: "{args.text}" [{args.language}]')
    t0 = time.time()
    results = list(
        tts.generate(
            text=args.text,
            language=args.language,
            ref_tokens=ref_tokens,
            ref_text=args.ref_text,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )
    )
    elapsed = time.time() - t0

    audio = np.array(results[0].audio)
    audio_write(args.output, audio, results[0].sample_rate)
    print(
        f"Saved {args.output} ({results[0].audio_duration}, generated in {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
