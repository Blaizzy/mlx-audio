import argparse
import os
import sys
from typing import Optional

import mlx.core as mx
import soundfile as sf
from scipy.signal import resample

from .audio_player import AudioPlayer
from .utils import load_model


def load_audio(audio_path: str, sample_rate: int = 24000) -> mx.array:
    samples, orig_sample_rate = sf.read(audio_path)
    shape = samples.shape
    # Collapse multi channel as mono
    if len(shape) > 1:
        samples = samples.sum(axis=1)
        # Divide summed samples by channel count.
        samples = samples / shape[1]
    if sample_rate != orig_sample_rate:
        print(f"Resampling from {orig_sample_rate} to {sample_rate}")
        duration = samples.shape[0] / orig_sample_rate
        num_samples = int(duration * sample_rate)
        samples = resample(samples, num_samples)
    audio = mx.array(samples, dtype=mx.float32)
    return audio


def generate_audio(
    text: str,
    model_path: str = "prince-canuma/Kokoro-82M",
    voice: str = "af_heart",
    speed: float = 1.0,
    lang_code: str = "a",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    stt_model: str = "mlx-community/whisper-large-v3-turbo",
    file_prefix: str = "audio",
    audio_format: str = "wav",
    sample_rate: int = 24000,
    join_audio: bool = False,
    play: bool = False,
    verbose: bool = True,
    temperature: float = 0.7,
    stream: bool = False,
    streaming_interval: float = 2.0,
    **kwargs,
) -> None:
    """
    Generates audio from text using a specified TTS model.

    Parameters:
    - text (str): The input text to be converted to speech.
    - model (str): The TTS model to use.
    - voice (str): The voice style to use.
    - temperature (float): The temperature for the model.
    - speed (float): Playback speed multiplier.
    - lang_code (str): The language code.
    - ref_audio (mx.array): Reference audio you would like to clone the voice from.
    - ref_text (str): Caption for reference audio.
    - stt_model (str): A mlx whisper model to use to transcribe.
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
        play = play or stream

        # Load reference audio for voice matching if specified

        if ref_audio:
            if not os.path.exists(ref_audio):
                raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")
            ref_audio = load_audio(ref_audio, sample_rate=sample_rate)
            if not ref_text:
                print("Ref_text not found. Transcribing ref_audio...")
                from mlx_audio.stt.models.whisper import Model as Whisper

                stt_model = Whisper.from_pretrained(path_or_hf_repo=stt_model)
                ref_text = stt_model.generate(ref_audio).text
                print("Ref_text", ref_text)

        # Load AudioPlayer
        player = AudioPlayer(sample_rate=sample_rate) if play else None

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
            verbose=verbose,
            stream=stream,
            streaming_interval=streaming_interval,
            **kwargs,
        )

        audio_list = []
        file_name = f"{file_prefix}.{audio_format}"
        for i, result in enumerate(results):
            if play:
                player.queue_audio(result.audio)

            if join_audio:
                audio_list.append(result.audio)
            elif not stream:
                file_name = f"{file_prefix}_{i:03d}.{audio_format}"
                sf.write(file_name, result.audio, result.sample_rate)
                print(f"✅ Audio successfully generated and saving as: {file_name}")

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

        if join_audio and not stream:
            if verbose:
                print(f"Joining {len(audio_list)} audio files")
            audio = mx.concatenate(audio_list, axis=0)
            sf.write(f"{file_prefix}.{audio_format}", audio, 24000)
            if verbose:
                print(f"✅ Audio successfully generated and saving as: {file_name}")

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
    parser.add_argument(
        "--gender", type=str, default="male", help="Gender of the voice [male, female]"
    )
    parser.add_argument("--pitch", type=float, default=1.0, help="Pitch of the voice")
    parser.add_argument("--lang_code", type=str, default="a", help="Language code")
    parser.add_argument(
        "--file_prefix", type=str, default="audio", help="Output file name prefix"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
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
        "--stt_model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo",
        help="STT model to use to transcribe reference audio",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for the model"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for the model")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for the model")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty for the model",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the audio as segments instead of saving to a file",
    )
    parser.add_argument(
        "--streaming_interval",
        type=float,
        default=2.0,
        help="The time interval in seconds for streaming segments",
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
    generate_audio(model_path=args.model, **vars(args))


if __name__ == "__main__":
    main()
