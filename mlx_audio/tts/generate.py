import argparse
import sys

import mlx.core as mx
import soundfile as sf

from .audio_player import AudioPlayer
from .utils import load_model


def generate_audio(text: str,
                   model: str = "prince-canuma/Kokoro-82M",
                   voice: str = "af_heart",
                   speed: float = 1.0,
                   lang_code: str = "a",
                   file_path: str = "audio",
                   audio_format: str = "wav",
                   sample_rate: int = 24000,
                   join_audio: bool = False,
                   play: bool = False,
                   verbose: bool = True,
                   from_cli: bool = False) -> None:
    """
    Generates audio from text using a specified TTS model.

    Parameters:
    - text (str): The input text to be converted to speech.
    - model (str): The TTS model to use.
    - voice (str): The voice style to use.
    - speed (float): Playback speed multiplier.
    - lang_code (str): The language code.
    - file_path (str): The output file path without extension.
    - audio_format (str): Output audio format (e.g., "wav", "flac").
    - sample_rate (int): Sampling rate in Hz.
    - join_audio (bool): Whether to join multiple audio files into one.
    - play (bool): Whether to play the generated audio.
    - verbose (bool): Whether to print status messages.
    - from_cli (bool): Indicates whether the function is called from the command line.

    Returns:
    - None: The function writes the generated audio to a file.
    """
    try:
        model_instance = load_model(model_path=model)

        if verbose:
            print(f"\n\033[94mModel:\033[0m {model}")
            print(f"\033[94mText:\033[0m {text}")
            print(f"\033[94mVoice:\033[0m {voice}")
            print(f"\033[94mSpeed:\033[0m {speed}x")
            print(f"\033[94mLanguage:\033[0m {lang_code}")
            print("==========")

        results = model_instance.generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            verbose=verbose
        )

        audio_list = [result.audio for result in results]

        if join_audio or play:
            final_audio = mx.concatenate(audio_list, axis=0)
            sf.write(f"{file_path}.{audio_format}", final_audio, sample_rate)
        else:
            final_audio = None
            # Only add suffix (_000) when running from CLI
            for i, audio in enumerate(audio_list):
                output_file = f"{file_path}_{i:03d}.{audio_format}" if from_cli else f"{file_path}.{audio_format}"
                sf.write(output_file, audio, sample_rate)

        if play:
            player = AudioPlayer()
            player.queue_audio(final_audio)
            player.wait_for_drain()
            player.stop()

        if verbose:
            print(f"✅ Audio successfully generated and saved as: {file_path}.{audio_format}")

    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        import traceback
        traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio from text using TTS.")
    parser.add_argument("--model", type=str, default="prince-canuma/Kokoro-82M", help="Path or repo ID of the model")
    parser.add_argument("--text", type=str, default=None, help="Text to generate (leave blank to input via stdin)")
    parser.add_argument("--voice", type=str, default="af_heart", help="Voice name")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed of the audio")
    parser.add_argument("--lang_code", type=str, default="a", help="Language code")
    parser.add_argument("--file_prefix", type=str, default="audio", help="Output file name prefix")
    parser.add_argument("--audio_format", type=str, default="wav", help="Output audio format")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Audio sample rate in Hz")
    parser.add_argument("--join_audio", action="store_true", help="Join all audio files into one")
    parser.add_argument("--play", action="store_true", help="Play the output audio")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
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
    generate_audio(
        text=args.text,
        model=args.model,
        voice=args.voice,
        speed=args.speed,
        lang_code=args.lang_code,
        file_path=args.file_prefix,
        audio_format=args.audio_format,
        sample_rate=args.sample_rate,
        join_audio=args.join_audio,
        play=args.play,
        verbose=args.verbose,
        from_cli=True  # Indicate that this was called from CLI
    )


if __name__ == "__main__":
    main()
