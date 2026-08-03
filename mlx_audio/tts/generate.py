import argparse
import inspect
import os
import sys
from os import PathLike
from typing import Any, Optional, Tuple, Union

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


def write_joined_audio(
    file_name: str,
    audio_chunks: list,
    sample_rate: int,
    audio_format: str,
) -> None:
    if not audio_chunks:
        return

    audio = (
        mx.concatenate(audio_chunks, axis=0)
        if len(audio_chunks) > 1
        else audio_chunks[0]
    )
    audio_write(file_name, audio, sample_rate, format=audio_format)


def _as_reference_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _collapse_reference_list(values: list[Any]) -> Any:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def _model_accepts_ref_text(model: nn.Module) -> bool:
    try:
        return "ref_text" in inspect.signature(model.generate).parameters
    except (TypeError, ValueError):
        return False


def _model_preserves_ref_audio_paths(model: nn.Module) -> bool:
    return getattr(model, "preserve_ref_audio_path", False) is True


def generate_audio(
    text: str = "",
    model: Optional[Union[str, nn.Module]] = None,
    max_tokens: Optional[int] = 1200,
    voice: str = "af_heart",
    prompt: Optional[str] = None,
    instruct: Optional[str] = None,
    speed: float = 1.0,
    lang_code: str = "en",
    cfg_scale: Optional[float] = None,
    ddpm_steps: Optional[int] = None,
    sigma: Optional[float] = None,
    ref_audio: Optional[Union[str, list[str]]] = None,
    ref_text: Optional[Union[str, list[str]]] = None,
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
    save: bool = False,
    use_zero_spk_emb: bool = False,
    add_spk: str = "",
    save_spkinfo: str = "",
    load_spkinfo: str = "",
    strict: bool = True,
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
    - save (bool): Whether to save streamed audio to a file when using stream mode.
    - model (object): A already loaded model.
    - stt_model (object): A already loaded stt model.
    - strict (bool): Whether to require every checkpoint weight to be present when
      loading a model from a path. Some models (e.g. cosyvoice3) have buffers that
      are generated at init time rather than stored in the checkpoint; set this to
      False to allow loading those. Ignored if `model` is already a loaded instance.
    Returns:
    - None: The function writes the generated audio to a file when not streaming, or when streaming with saving enabled.
    """
    try:
        play = play or stream
        cross_lingual = kwargs.get("cross_lingual", False)

        if model is None:
            raise ValueError("Model path or model instance must be provided.")

        if isinstance(model, str):
            # Load model
            model = load_model(model_path=model, strict=strict)

        # load speaker presets from a previously saved spk2info file
        if load_spkinfo:
            if not hasattr(model, "load_spkinfo"):
                raise ValueError(
                    f"Model {type(model).__name__} does not support spk2info "
                    f"(no load_spkinfo method)"
                )
            count = model.load_spkinfo(load_spkinfo)
            print(f"Loaded {count} speaker preset(s) from {load_spkinfo}")
            print(f"  Speakers: {model.list_spks()}")

        ref_audio_values = _as_reference_list(ref_audio)
        ref_text_values = _as_reference_list(ref_text)
        if (
            ref_audio_values
            and ref_text_values
            and len(ref_text_values) != len(ref_audio_values)
        ):
            raise ValueError("ref_audio and ref_text lists must have the same length.")
        if len(ref_text_values) > 1 and not ref_audio_values:
            raise ValueError(
                "Multiple ref_text values require matching ref_audio values."
            )

        # Load reference audio for voice matching if specified. Some models own
        # reference preprocessing and should receive paths unchanged.
        if ref_audio_values:
            normalize = False
            if hasattr(model, "model_type") and model.model_type == "spark":
                normalize = True

            preserve_ref_paths = _model_preserves_ref_audio_paths(model)
            if preserve_ref_paths:
                loaded_ref_audio = []
                for ref_audio_item in ref_audio_values:
                    if isinstance(ref_audio_item, (str, PathLike)):
                        ref_audio_path = os.fspath(ref_audio_item)
                        if not os.path.exists(ref_audio_path):
                            raise FileNotFoundError(
                                f"Reference audio file not found: {ref_audio_path}"
                            )
                        loaded_ref_audio.append(ref_audio_path)
                    else:
                        loaded_ref_audio.append(ref_audio_item)
            else:
                loaded_ref_audio = []
                for ref_audio_item in ref_audio_values:
                    if isinstance(ref_audio_item, (str, PathLike)):
                        ref_audio_path = os.fspath(ref_audio_item)
                        if not os.path.exists(ref_audio_path):
                            raise FileNotFoundError(
                                f"Reference audio file not found: {ref_audio_path}"
                            )
                        loaded_ref_audio.append(
                            load_audio(
                                ref_audio_path,
                                sample_rate=model.sample_rate,
                                volume_normalize=normalize,
                            )
                        )
                    else:
                        loaded_ref_audio.append(ref_audio_item)
            ref_audio = _collapse_reference_list(loaded_ref_audio)

            if ref_text_values:
                ref_text = _collapse_reference_list(ref_text_values)
            elif preserve_ref_paths:
                ref_text = None
            elif cross_lingual:
                # cross_lingual mode never feeds ref_text to the LLM (see
                # CosyVoice3.generate's cross_lingual docstring), so
                # transcribing ref_audio here would be pointless work.
                ref_text = None
            elif _model_accepts_ref_text(model):
                if stt_model is None:
                    raise ValueError(
                        "STT model path or model instance must be provided when "
                        "ref_text is missing."
                    )
                print("Ref_text not found. Transcribing ref_audio...")
                from mlx_audio.stt import load as load_stt_model

                if isinstance(stt_model, str):
                    stt_model = load_stt_model(stt_model)
                transcribed_ref_text = [
                    stt_model.generate(audio).text for audio in loaded_ref_audio
                ]

                del stt_model
                mx.clear_cache()
                ref_text = _collapse_reference_list(transcribed_ref_text)
                print(f"\033[94mRef_text:\033[0m {ref_text}")
            else:
                ref_text = None
        elif ref_text_values:
            ref_text = _collapse_reference_list(ref_text_values)

        # cache a zero-shot speaker preset (requires ref_audio + ref_text)
        if add_spk:
            if not hasattr(model, "add_zero_shot_spk"):
                raise ValueError(
                    f"Model {type(model).__name__} does not support spk2info "
                    f"(no add_zero_shot_spk method)"
                )
            if not ref_audio or not ref_text:
                raise ValueError(
                    "--add_spk requires --ref_audio and --ref_text "
                    "(or a --stt_model to transcribe ref_audio)"
                )
            ok = model.add_zero_shot_spk(ref_text, ref_audio, add_spk)
            if ok:
                print(f"Speaker '{add_spk}' cached (prompt features pre-extracted)")

        # Load AudioPlayer
        player = AudioPlayer(sample_rate=model.sample_rate) if play else None

        # Handle output path
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            file_prefix = os.path.join(output_path, file_prefix)

        if instruct is not None:
            print(f"\033[94mInstruct:\033[0m {instruct}")

        # auto-detect display language from text content
        _display_lang = lang_code
        if _display_lang in ("en", "auto"):
            from .models.cosyvoice3.frontend_utils import contains_chinese
            _display_lang = "zh" if contains_chinese(text) else "en"

        _lines = [
            f"\033[94mText:\033[0m {text}",
        ]
        if voice:
            _lines.append(f"\033[94mVoice:\033[0m {voice}")
        _lines.append(f"\033[94mSpeed:\033[0m {speed}x")
        _lines.append(f"\033[94mLanguage:\033[0m {_display_lang}")
        print("\n".join(_lines))

        extra_kwargs = {
            key: value for key, value in kwargs.items() if value is not None
        }

        gen_kwargs = dict(
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
            instruct=instruct,
            use_zero_spk_emb=use_zero_spk_emb,
            **extra_kwargs,
        )
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens
        if cfg_scale is not None:
            gen_kwargs["cfg_scale"] = cfg_scale
        if ddpm_steps is not None:
            gen_kwargs["ddpm_steps"] = ddpm_steps
        if prompt is not None:
            gen_kwargs["prompt"] = prompt
        if sigma is not None:
            gen_kwargs["sigma"] = sigma

        if text:
            results = model.generate(**gen_kwargs)
        else:
            results = []

        save_streamed_audio = stream and save
        audio_list = []
        streamed_audio_chunks = []
        streamed_segment_audio = {}
        streamed_segment_sample_rates = {}
        file_name = f"{file_prefix}.{audio_format}"
        for i, result in enumerate(results):
            if play:
                player.queue_audio(result.audio)

            if save_streamed_audio:
                if join_audio:
                    streamed_audio_chunks.append(result.audio)
                else:
                    segment_idx = result.segment_idx
                    if segment_idx not in streamed_segment_audio:
                        streamed_segment_audio[segment_idx] = []
                        streamed_segment_sample_rates[segment_idx] = result.sample_rate
                    streamed_segment_audio[segment_idx].append(result.audio)
            elif join_audio and not stream:
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

        if save_streamed_audio and join_audio and streamed_audio_chunks:
            if verbose:
                print(f"Joining {len(streamed_audio_chunks)} streamed audio chunks")
            write_joined_audio(
                file_name,
                streamed_audio_chunks,
                model.sample_rate,
                audio_format,
            )
            print(f"✅ Audio successfully generated and saving as: {file_name}")
        elif save_streamed_audio and streamed_segment_audio:
            for segment_idx in sorted(streamed_segment_audio):
                file_name = f"{file_prefix}_{segment_idx:03d}.{audio_format}"
                audio_chunks = streamed_segment_audio[segment_idx]
                sample_rate = streamed_segment_sample_rates[segment_idx]
                if verbose:
                    print(
                        "Joining "
                        f"{len(audio_chunks)} streamed audio chunks for segment "
                        f"{segment_idx}"
                    )
                write_joined_audio(
                    file_name,
                    audio_chunks,
                    sample_rate,
                    audio_format,
                )
                print(f"✅ Audio successfully generated and saving as: {file_name}")
        elif join_audio and not stream and audio_list:
            if verbose:
                print(f"Joining {len(audio_list)} audio files")
            write_joined_audio(
                file_name,
                audio_list,
                model.sample_rate,
                audio_format,
            )
            if verbose:
                print(f"✅ Audio successfully generated and saving as: {file_name}")

        if play:
            player.wait_for_drain()
            player.stop()

        # persist speaker presets to disk after generation (or after --add_spk)
        if save_spkinfo:
            if not hasattr(model, "save_spkinfo"):
                raise ValueError(
                    f"Model {type(model).__name__} does not support spk2info "
                    f"(no save_spkinfo method)"
                )
            model.save_spkinfo(save_spkinfo)
            if verbose:
                print(f"Speaker presets saved to {save_spkinfo}")

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
        default=None,
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
        "--prompt",
        type=str,
        default=None,
        help="Optional model-specific prompt prefix.",
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
        default=None,
        help="Classifier-free guidance scale. Defaults to the model configuration.",
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=None,
        help="Override diffusion steps. Higher = better quality, slower (try 30-50).",
    )

    parser.add_argument("--speed", type=float, default=1.0, help="Speed of the audio")
    parser.add_argument(
        "--gen_duration",
        type=float,
        default=None,
        help="Optional model-specific generation duration in seconds.",
    )
    parser.add_argument(
        "--duration_multiplier",
        type=float,
        default=None,
        help="Optional model-specific automatic duration multiplier.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional model-specific generation step count.",
    )
    parser.add_argument(
        "--stg_scale",
        type=float,
        default=None,
        help="Optional model-specific spatiotemporal guidance scale.",
    )
    parser.add_argument(
        "--stg_block",
        type=int,
        default=None,
        help="Optional model-specific spatiotemporal guidance block.",
    )
    parser.add_argument(
        "--rescale_scale",
        default=None,
        help="Optional model-specific CFG rescale value.",
    )
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
        "--ref_audio",
        type=str,
        action="append",
        default=None,
        help="Path to reference audio. Repeat for multiple references.",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        action="append",
        default=None,
        help="Caption for reference audio. Repeat to match repeated --ref_audio.",
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
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Optional model-specific sigma value (e.g., Ming Omni).",
    )
    parser.add_argument(
        "--use_zero_spk_emb",
        action="store_true",
        help="Optional model-specific zero speaker embedding mode (e.g., Ming Omni).",
    )
    parser.add_argument(
        "--cross_lingual",
        action="store_true",
        help=(
            "Optional model-specific cross-lingual voice cloning mode (e.g. "
            "cosyvoice3): withhold ref_text/ref_audio speech tokens from the "
            "LLM so the first generated tokens aren't biased towards "
            "ref_audio's language; the reference's timbre is still cloned."
        ),
    )
    parser.add_argument(
        "--spk_id",
        type=str,
        default="",
        help=(
            "Optional cached zero-shot speaker id (e.g. cosyvoice3). "
            "When set, the model reuses previously extracted prompt features "
            "for this speaker (saved via add_zero_shot_spk), skipping the "
            "speech tokenizer, mel extractor, and speaker encoder."
        ),
    )
    parser.add_argument(
        "--add_spk",
        type=str,
        default="",
        help=(
            "Cache a zero-shot speaker preset under this name. Requires "
            "--ref_audio and --ref_text (or --stt_model to transcribe). "
            "Combine with --save_spkinfo to persist to disk."
        ),
    )
    parser.add_argument(
        "--save_spkinfo",
        type=str,
        default="",
        help=(
            "Persist cached speaker presets to a safetensors file "
            "(e.g. speakers.safetensors). Use after --add_spk, or combine "
            "with --spk_id to save previously loaded speakers."
        ),
    )
    parser.add_argument(
        "--load_spkinfo",
        type=str,
        default="",
        help=(
            "Load speaker presets from a safetensors file saved by "
            "--save_spkinfo. Speakers are then available via --spk_id."
        ),
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for the model")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for the model")
    parser.add_argument(
        "--min_p",
        type=float,
        default=None,
        help="Optional model-specific min-p sampling threshold.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty for the model",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the audio as segments during generation",
    )
    parser.add_argument(
        "--streaming_interval",
        type=float,
        default=2.0,
        help="The time interval in seconds for streaming segments",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save streamed audio to a file. Requires --stream.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help=(
            "Allow loading a model even if some checkpoint weights are missing "
            "(e.g. cosyvoice3's init-time-generated buffers)."
        ),
    )

    args = parser.parse_args()

    if args.save and not args.stream:
        parser.error("--save requires --stream")

    if args.text is None:
        # --add_spk without --text means "only cache the speaker, don't generate"
        if args.add_spk:
            args.text = ""
        elif not sys.stdin.isatty():
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
