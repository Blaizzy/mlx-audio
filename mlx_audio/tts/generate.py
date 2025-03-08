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
            if args.join_audio or args.play:
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
            audio = mx.concatenate(audio_list, axis=0)

            player = AudioPlayer()
            player.queue_audio(audio)
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


def generate_audio(text: str,
                   model: str = "prince-canuma/Kokoro-82M",
                   voice: str = "af_heart",
                   speed: float = 1.0,
                   lang_code: str = "a",
                   file_path: str = "audio",
                   audio_format: str = "wav",
                   sample_rate: int = 24000,
                   verbose: bool = True) -> None:
    """
    Generates audio from text using a specified TTS model.

    Parameters:
    - text (str): The input text to be converted to speech.
    - model (str): The TTS model to use (default: "prince-canuma/Kokoro-82M").
    - voice (str): The voice style to use (default: "af_heart").
    - speed (float): Playback speed multiplier (default: 1.0).
    - lang_code (str): The language code (default: "a").
    - file_path (str): The output file path without extension (default: "audio").
    - audio_format (str): Output audio format (e.g., "wav", "flac") (default: "wav").
    - sample_rate (int): Sampling rate in Hz (default: 24000).
    - verbose (bool): Whether to print status messages (default: True).

    Returns:
    - None: The function writes the generated audio to a file.
    """
    try:
        # Load the specified TTS model
        model_instance = load_model(model_path=model)

        # Generate audio using the model
        results = model_instance.generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            verbose=verbose,  # Pass verbose to model if applicable
        )

        # Concatenate generated audio segments
        audio_list = [result.audio for result in results]
        final_audio = mx.concatenate(audio_list, axis=0)

        # Save the audio to the specified file format
        output_file = f"{file_path}.{audio_format}"
        sf.write(output_file, final_audio, sample_rate)

        if verbose:
            print(f"✅ Audio successfully generated and saved as: {output_file}")

    except Exception as e:
        print(f"❌ Error generating audio: {e}")


if __name__ == "__main__":
    main()
