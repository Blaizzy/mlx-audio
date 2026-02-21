import argparse
import os
import sys
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mlx_audio.audio_io import write as audio_write
from mlx_audio.utils import load_audio

from .audio_player import AudioPlayer
from .utils import load_model


def detect_speech_boundaries(
    wav: np.ndarray,
    sample_rate: int,
    window_duration: float = 0.1,
    energy_threshold: float = 0.01,
    margin_factor: int = 2,
) -> Tuple[int, int]:
    """Detect the start and end points of speech in an audio signal using RMS energy.

    Args:
        wav: Input audio signal array with values in [-1, 1]
        sample_rate: Audio sample rate in Hz
        window_duration: Duration of detection window in seconds
        energy_threshold: RMS energy threshold for speech detection
        margin_factor: Factor to determine extra margin around detected boundaries

    Returns:
        tuple: (start_index, end_index) of speech segment

    Raises:
        ValueError: If the audio contains only silence
    """
    window_size = int(window_duration * sample_rate)
    margin = margin_factor * window_size
    step_size = window_size // 10

    # Create sliding windows using stride tricks to avoid loops
    windows = sliding_window_view(wav, window_size)[::step_size]

    # Calculate RMS energy for each window
    energy = np.sqrt(np.mean(windows**2, axis=1))
    speech_mask = energy >= energy_threshold

    if not np.any(speech_mask):
        raise ValueError("No speech detected in audio (only silence)")

    start = max(0, np.argmax(speech_mask) * step_size - margin)
    end = min(
        len(wav),
        (len(speech_mask) - 1 - np.argmax(speech_mask[::-1])) * step_size + margin,
    )

    return start, end


def remove_silence_on_both_ends(
    wav: np.ndarray,
    sample_rate: int,
    window_duration: float = 0.1,
    volume_threshold: float = 0.01,
) -> np.ndarray:
    """Remove silence from both ends of an audio signal.

    Args:
        wav: Input audio signal array
        sample_rate: Audio sample rate in Hz
        window_duration: Duration of detection window in seconds
        volume_threshold: Amplitude threshold for silence detection

    Returns:
        np.ndarray: Audio signal with silence removed from both ends

    Raises:
        ValueError: If the audio contains only silence
    """
    start, end = detect_speech_boundaries(
        wav, sample_rate, window_duration, volume_threshold
    )
    return wav[start:end]


def hertz_to_mel(pitch: float) -> float:
    """
    Converts a frequency from the Hertz scale to the Mel scale.

    Parameters:
    - pitch: float or ndarray
        Frequency in Hertz.

    Returns:
    - mel: float or ndarray
        Frequency in Mel scale.
    """
    mel = 2595 * np.log10(1 + pitch / 700)
    return mel


def generate_audio(
    text: str,
    model: Optional[Union[str, nn.Module]] = None,
    max_tokens: int = 1200,
    voice: str = "af_heart",
    instruct: Optional[str] = None,
    speed: float = 1.0,
    lang_code: str = "en",
    cfg_scale: Optional[float] = None,
    ddpm_steps: Optional[int] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    stt_model: Optional[
        Union[str, nn.Module]
    ] = "mlx-community/whisper-large-v3-turbo-asr-fp16",
    output_path: Optional[str] = None,
    file_prefix: str = "audio",
    audio_format: str = "wav",
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
    - voice (str): The voice style to use (also used as speaker for Qwen3-TTS models).
    - instruct (str): Instruction for emotion/style (CustomVoice) or voice description (VoiceDesign).
    - temperature (float): The temperature for the model.
    - speed (float): Playback speed multiplier.
    - lang_code (str): The language code.
    - ref_audio (mx.array): Reference audio you would like to clone the voice from.
    - ref_text (str): Caption for reference audio.
    - stt_model_path (str): A mlx whisper model to use to transcribe.
    - output_path (str): Directory path where audio files will be saved.
    - file_prefix (str): The output file path without extension.
    - audio_format (str): Output audio format (e.g., "wav", "flac").
    - join_audio (bool): Whether to join multiple audio files into one.
    - play (bool): Whether to play the generated audio.
    - verbose (bool): Whether to print status messages.
    - model (object): A already loaded model.
    - stt_model (object): A already loaded stt model.
    Returns:
    - None: The function writes the generated audio to a file.
    """
    try:
        play = play or stream

        if model is None:
            raise ValueError("Model path or model instance must be provided.")

        if stt_model is None and (ref_audio and ref_text is None):
            raise ValueError(
                "STT model path or model instance must be provided when ref_text is missing."
            )

        if isinstance(model, str):
            # Load model
            model = load_model(model_path=model)

        # Load reference audio for voice matching if specified
        if ref_audio:
            if not os.path.exists(ref_audio):
                raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")

            normalize = False
            if hasattr(model, "model_type") and model.model_type == "spark":
                normalize = True

            ref_audio = load_audio(
                ref_audio, sample_rate=model.sample_rate, volume_normalize=normalize
            )
            if not ref_text:
                import inspect

                if "ref_text" in inspect.signature(model.generate).parameters:
                    print("Ref_text not found. Transcribing ref_audio...")
                    from mlx_audio.stt import load as load_stt_model

                    if isinstance(stt_model, str):
                        stt_model = load_stt_model(stt_model)
                    ref_text = stt_model.generate(ref_audio).text

                    del stt_model
                    mx.clear_cache()
                    print(f"\033[94mRef_text:\033[0m {ref_text}")

        # Load AudioPlayer
        player = AudioPlayer(sample_rate=model.sample_rate) if play else None

        # Handle output path
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            file_prefix = os.path.join(output_path, file_prefix)

        if instruct is not None:
            print(f"\033[94mInstruct:\033[0m {instruct}")

        print(
            f"\033[94mText:\033[0m {text}\n"
            f"\033[94mVoice:\033[0m {voice}\n"
            f"\033[94mSpeed:\033[0m {speed}x\n"
            f"\033[94mLanguage:\033[0m {lang_code}"
        )

        gen_kwargs = dict(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            ref_audio=ref_audio,
            ref_text=ref_text,
            cfg_scale=cfg_scale,
            ddpm_steps=ddpm_steps,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            stream=stream,
            streaming_interval=streaming_interval,
            instruct=instruct,
            **kwargs,
        )

        results = model.generate(**gen_kwargs)

        audio_list = []
        file_name = f"{file_prefix}.{audio_format}"
        for i, result in enumerate(results):
            if play:
                player.queue_audio(result.audio)

            if join_audio:
                audio_list.append(result.audio)
            elif not stream:
                file_name = f"{file_prefix}_{i:03d}.{audio_format}"
                audio_write(
                    file_name,
                    np.array(result.audio),
                    result.sample_rate,
                    format=audio_format,
                )
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
            audio_write(
                f"{file_prefix}.{audio_format}",
                audio,
                model.sample_rate,
            )
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
        required=True,
        help="Path or repo id of the model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1200,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to generate (leave blank to input via stdin)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice/speaker name (e.g., Chelsie, Ethan, Vivian for Qwen3-TTS)",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Instruction for CustomVoice (emotion/style) or VoiceDesign (voice description)",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Exaggeration factor for the voice",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale. Lower (≈1.0-1.5) is often more stable.",
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=None,
        help="Override diffusion steps. Higher = better quality, slower (try 30-50).",
    )

    parser.add_argument("--speed", type=float, default=1.0, help="Speed of the audio")
    parser.add_argument(
        "--gender", type=str, default="male", help="Gender of the voice [male, female]"
    )
    parser.add_argument("--pitch", type=float, default=1.0, help="Pitch of the voice")
    parser.add_argument("--lang_code", type=str, default="en", help="Language code")
    parser.add_argument(
        "--output_path", type=str, default=None, help="Directory path for output files"
    )
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
        "--ref_audio", type=str, default=None, help="Path to reference audio"
    )
    parser.add_argument(
        "--ref_text", type=str, default=None, help="Caption for reference audio"
    )
    parser.add_argument(
        "--stt_model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo-asr-fp16",
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
    generate_audio(**vars(args))


if __name__ == "__main__":
    main()
