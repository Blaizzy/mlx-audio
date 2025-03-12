import argparse
import json
import os
import sys

import mlx.core as mx
import soundfile as sf

from .audio_player import AudioPlayer
from .utils import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="prince-canuma/Kokoro-82M",
        help="Path or repo id of the model",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to generate (leave blank to input via stdin)",
    )
    parser.add_argument("--voice", type=str, default="af_heart", help="Voice name")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed of the audio")
    parser.add_argument("--lang_code", type=str, default="a", help="Language code")
    parser.add_argument(
        "--file_prefix", type=str, default="audio", help="Output file name prefix"
    )
    parser.add_argument("--verbose", action="store_false", help="Print verbose output")
    parser.add_argument(
        "--join_audio", action="store_true", help="Join all audio files into one"
    )
    parser.add_argument("--play", action="store_true", help="Play the output audio")
    args = parser.parse_args()

    if args.text is None:
        if not sys.stdin.isatty():
            args.text = sys.stdin.read().strip()
        else:
            print("Please enter the text to generate:")
            args.text = input("> ").strip()

    return args


def main():
    args = parse_args()
    try:
        player = AudioPlayer() if args.play else None

        model = load_model(model_path=args.model)
        print(
            f"\n\033[94mModel:\033[0m {args.model}\n"
            f"\033[94mText:\033[0m {args.text}\n"
            f"\033[94mVoice:\033[0m {args.voice}\n"
            f"\033[94mSpeed:\033[0m {args.speed}x\n"
            f"\033[94mLanguage:\033[0m {args.lang_code}"
        )
        print("==========")
        results = model.generate(
            text=args.text,
            voice=args.voice,
            speed=args.speed,
            lang_code=args.lang_code,
            verbose=True,
        )
        print(
            f"\033[92mAudio generated successfully, saving to\033[0m {args.file_prefix}!"
        )

        audio_list = []
        for i, result in enumerate(results):
            if args.play:
                player.queue_audio(result.audio)
            if args.join_audio:
                audio_list.append(result.audio)
            else:
                sf.write(f"{args.file_prefix}_{i:03d}.wav", result.audio, 24000)

            if args.verbose:
                print("==========")
                print(f"Duration:              {result.audio_duration}")
                print(
                    f"Samples/sec:           {result.audio_samples['samples-per-sec']:.1f}"
                )
                print(
                    f"Prompt:                {result.token_count} tokens, {result.prompt['tokens-per-sec']:.1f} tokens-per-sec"
                )
                print(
                    f"Audio:                 {result.audio_samples['samples']} samples, {result.audio_samples['samples-per-sec']:.1f} samples-per-sec"
                )
                print(f"Real-time factor:      {result.real_time_factor:.2f}x")
                print(f"Processing time:       {result.processing_time_seconds:.2f}s")
                print(f"Peak memory usage:     {result.peak_memory_usage:.2f}GB")

        if args.join_audio:
            print(f"Joining {len(audio_list)} audio files")
            audio = mx.concatenate(audio_list, axis=0)
            sf.write(f"{args.file_prefix}.wav", audio, 24000)

        if args.play:
            player.wait_for_drain()
            player.stop()
    except ImportError as e:
        print(f"Import error: {e}")
        print(
            "This might be due to incorrect Python path. Check your project structure."
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
