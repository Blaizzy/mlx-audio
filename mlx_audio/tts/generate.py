import argparse
import os
import sys
from typing import Optional

import mlx.core as mx
import soundfile as sf

from .audio_player import AudioPlayer
from .utils import load_model
import torchaudio


def load_audio(
    audio_path: str,
    sample_rate: int=24000
    ) -> mx.array:
    audio_tensor, orig_sample_rate = torchaudio.load(audio_path)
    shape = audio_tensor.shape
    # Collapse multi channel as mono
    if len(shape)>1:
        audio_tensor = audio_tensor.sum(dim=0)
        # Divide summed samples by channel count.
        audio_tensor = audio_tensor/shape[0]
    if sample_rate != orig_sample_rate:
        print(f"Resampling from {orig_sample_rate} to {sample_rate}")
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=orig_sample_rate, new_freq=sample_rate
        )
    audio = mx.array(audio_tensor, dtype=mx.float32)
    return audio


def generate_audio(
    text: str,
    model_path: str = "prince-canuma/Kokoro-82M",
    voice: str = "af_heart",
    speed: float = 1.0,
    lang_code: str = "a",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    file_prefix: str = "audio",
    audio_format: str = "wav",
    sample_rate: int = 24000,
    join_audio: bool = False,
    play: bool = False,
    verbose: bool = True,
    temperature: float = 0.7,
    **kwargs,
) -> None:
    """
    Generates audio from text using a specified TTS model.

    Parameters:
    - text (str): The input text to be converted to speech.
    - model (str): The TTS model to use.
    - voice (str): The voice style to use.
    - speed (float): Playback speed multiplier.
    - lang_code (str): The language code.
    - ref_audio (mx.array): Reference audio you would like to clone the voice from.
    - ref_text (str): Caption for reference audio.
    - file_prefix (str): The output file path without extension.
    - audio_format (str): Output audio format (e.g., "wav", "flac").
    - sample_rate (int): Sampling rate in Hz.
    - join_audio (bool): Whether to join multiple audio files into one.
    - play (bool): Whether to play the generated audio.
    - verbose (bool): Whether to print status messages.

    Returns:
    - None: The function writes the generated audio to a file.
    """
    try:
        # Load reference audio for voice matching if specified

        if ref_audio:
            if not os.path.exists(ref_audio):
                raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")
            ref_audio = load_audio(ref_audio)
            if not ref_text:
                print("Ref_text not found. Transcribing ref_audio...")
                # mlx_whisper seems takes long time to import. Import only necessary.
                import mlx_whisper
                ref_text = mlx_whisper.transcribe(ref_audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")["text"]
                print("Ref_text", ref_text)

        # Load AudioPlayer
        player = AudioPlayer() if play else None

        # Load model
        model = load_model(model_path=model_path)
        print(
            f"\n\033[94mModel:\033[0m {model_path}\n"
            f"\033[94mText:\033[0m {text}\n"
            f"\033[94mVoice:\033[0m {voice}\n"
            f"\033[94mSpeed:\033[0m {speed}x\n"
            f"\033[94mLanguage:\033[0m {lang_code}"
        )

        results = model.generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            ref_audio=ref_audio,
            ref_text=ref_text,
            temperature=temperature,
            verbose=True,
        )

        audio_list = []
        file_name = f"{file_prefix}.{audio_format}"
        for i, result in enumerate(results):
            if play:
                player.queue_audio(result.audio)
            if join_audio:
                audio_list.append(result.audio)

            else:
                file_name = f"{file_prefix}_{i:03d}.{audio_format}"
                sf.write(file_name, result.audio, 24000)

            if verbose:

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
                print(f"✅ Audio successfully generated and saving as: {file_name}")

        if join_audio:
            if verbose:
                print(f"Joining {len(audio_list)} audio files")
            audio = mx.concatenate(audio_list, axis=0)
            sf.write(f"{file_prefix}.{audio_format}", audio, 24000)

        if play:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio from text using TTS.")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Kokoro-82M-bf16",
        help="Path or repo id of the model",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to generate (leave blank to input via stdin)",
    )
    parser.add_argument("--voice", type=str, default=None, help="Voice name")
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
    parser.add_argument(
        "--audio_format", type=str, default="wav", help="Output audio format"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=24000, help="Audio sample rate in Hz"
    )
    parser.add_argument(
        "--ref_audio", type=str, default=None, help="Path to reference audio"
    )
    parser.add_argument(
        "--ref_text", type=str, default=None, help="Caption for reference audio"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for the model"
    )

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
        model_path=args.model,
        voice=args.voice,
        speed=args.speed,
        lang_code=args.lang_code,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        file_prefix=args.file_prefix,
        audio_format=args.audio_format,
        sample_rate=args.sample_rate,
        join_audio=args.join_audio,
        play=args.play,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
