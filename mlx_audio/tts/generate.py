import argparse
import json
import os
import sys
from typing import Any, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mlx_audio.audio_io import write as audio_write
from mlx_audio.utils import load_audio

from .audio_player import AudioPlayer
from .utils import load_model

_MOSS_PRESET_ALIASES = {
    "voicegenerator": "voice_generator",
    "moss_voice_generator": "voice_generator",
    "moss_voicegenerator": "voice_generator",
    "moss_voice_design": "voice_generator",
    "moss_sound_effect": "soundeffect",
    "sound_effect": "soundeffect",
    "moss_soundeffect": "soundeffect",
}

_MOSS_PRESET_TASKS = {
    "moss_tts": "default_tts",
    "moss_tts_local": "default_tts",
    "ttsd": "ttsd",
    "soundeffect": "soundeffect",
    "voice_generator": "voice_generator",
    "realtime": "realtime",
}

_TASK_DEFAULT_PRESETS = {
    "ttsd": "ttsd",
    "soundeffect": "soundeffect",
    "voice_generator": "voice_generator",
    "realtime": "realtime",
}

_REALTIME_ONLY_KWARGS = frozenset(
    {
        "chunk_frames",
        "overlap_frames",
        "decode_chunk_duration",
        "max_pending_frames",
        "repetition_window",
    }
)


def _normalize_preset(preset: Optional[str]) -> Optional[str]:
    if preset is None:
        return None
    normalized = str(preset).strip().lower().replace("-", "_")
    return _MOSS_PRESET_ALIASES.get(normalized, normalized)


def _normalize_model_type(model_type: Optional[str]) -> Optional[str]:
    if model_type is None:
        return None
    return str(model_type).strip().lower().replace("-", "_")


def _looks_like_moss_model(model: Optional[Union[str, nn.Module]]) -> bool:
    if isinstance(model, str):
        return "moss" in model.lower()
    if model is None:
        return False
    model_type = _normalize_model_type(getattr(model, "model_type", None))
    if model_type:
        return "moss" in model_type
    return "moss" in str(type(model)).lower()


def _is_explicit_realtime_model(model: Optional[Union[str, nn.Module]]) -> bool:
    if isinstance(model, str):
        normalized = model.strip().lower().replace("-", "_")
        return "moss_tts_realtime" in normalized or "moss/tts/realtime" in normalized
    if model is None:
        return False
    model_type = _normalize_model_type(getattr(model, "model_type", None))
    return model_type == "moss_tts_realtime"


def _parse_model_kwargs_json(
    model_kwargs_json: Optional[Union[str, dict[str, Any]]],
) -> dict[str, Any]:
    if model_kwargs_json is None:
        return {}
    parsed_kwargs = model_kwargs_json
    if isinstance(parsed_kwargs, str):
        try:
            parsed_kwargs = json.loads(parsed_kwargs)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --model_kwargs_json value: {exc.msg}") from exc
    if not isinstance(parsed_kwargs, dict):
        raise ValueError("--model_kwargs_json must decode to a JSON object")
    return dict(parsed_kwargs)


def _load_dialogue_speakers(
    dialogue_speakers_json: Optional[str],
) -> Optional[list[dict[str, Any]]]:
    if dialogue_speakers_json is None:
        return None
    with open(dialogue_speakers_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("dialogue_speakers_json must contain a JSON list")
    dialogue_speakers: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each dialogue_speakers item must be a JSON object")
        dialogue_speakers.append(item)
    return dialogue_speakers


def _infer_gateway_task_and_preset(
    *,
    model: Optional[Union[str, nn.Module]],
    preset: Optional[str],
    dialogue_speakers: Optional[list[dict[str, Any]]],
    sound_event: Optional[str],
    ambient_sound: Optional[str],
    instruct: Optional[str],
    repetition_window: Optional[int],
    passthrough_kwargs: dict[str, Any],
) -> tuple[str, Optional[str]]:
    normalized_preset = _normalize_preset(preset)
    explicit_preset_task = _MOSS_PRESET_TASKS.get(normalized_preset)

    explicit_realtime_marker = _is_explicit_realtime_model(model)
    realtime_kwarg_marker = bool(_REALTIME_ONLY_KWARGS & set(passthrough_kwargs.keys()))
    moss_hint = (
        _looks_like_moss_model(model)
        or normalized_preset in _MOSS_PRESET_TASKS
        or bool(dialogue_speakers)
        or bool(ambient_sound or sound_event)
    )
    realtime_marker = (
        explicit_realtime_marker
        or normalized_preset == "realtime"
        or (moss_hint and (repetition_window is not None or realtime_kwarg_marker))
    )

    ttsd_marker = bool(dialogue_speakers) or normalized_preset == "ttsd"
    soundeffect_marker = bool(ambient_sound or sound_event) or (
        normalized_preset == "soundeffect"
    )

    moss_context = moss_hint or ttsd_marker or soundeffect_marker or realtime_marker
    has_instruct = bool(str(instruct).strip()) if instruct is not None else False
    # `instruct` can represent style guidance for base TTS; only treat it as
    # VoiceGenerator routing when no explicit non-voice-design preset is pinned.
    voice_design_marker = normalized_preset == "voice_generator" or (
        moss_context and has_instruct and explicit_preset_task is None
    )

    inferred_task = "default_tts"
    if realtime_marker:
        inferred_task = "realtime"
    elif ttsd_marker:
        inferred_task = "ttsd"
    elif soundeffect_marker:
        inferred_task = "soundeffect"
    elif voice_design_marker:
        inferred_task = "voice_generator"

    if (
        explicit_preset_task is not None
        and inferred_task != explicit_preset_task
        and inferred_task != "default_tts"
    ):
        raise ValueError(
            "Incompatible task markers: "
            f"preset={preset!r} conflicts with inferred task '{inferred_task}'"
        )

    effective_preset = preset
    if normalized_preset in _MOSS_PRESET_TASKS:
        effective_preset = normalized_preset
    elif inferred_task in _TASK_DEFAULT_PRESETS:
        effective_preset = _TASK_DEFAULT_PRESETS[inferred_task]

    return inferred_task, effective_preset


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
    text: Optional[str],
    model: Optional[Union[str, nn.Module]] = None,
    max_tokens: int = 1200,
    tokens: Optional[int] = None,
    duration_s: Optional[float] = None,
    seconds: Optional[float] = None,
    n_vq_for_inference: Optional[int] = None,
    voice: str = "af_heart",
    instruct: Optional[str] = None,
    quality: Optional[str] = None,
    sound_event: Optional[str] = None,
    ambient_sound: Optional[str] = None,
    language: Optional[str] = None,
    preset: Optional[str] = None,
    model_kwargs_json: Optional[Union[str, dict[str, Any]]] = None,
    dialogue_speakers_json: Optional[str] = None,
    input_type: str = "text",
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
    seed: Optional[int] = None,
    repetition_window: Optional[int] = None,
    stream: bool = False,
    streaming_interval: float = 2.0,
    long_form: bool = False,
    long_form_min_chars: int = 160,
    long_form_target_chars: int = 320,
    long_form_max_chars: int = 520,
    long_form_prefix_audio_seconds: float = 2.0,
    long_form_prefix_audio_max_tokens: int = 25,
    long_form_prefix_text_chars: int = 0,
    long_form_retry_attempts: int = 0,
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
        if seed is not None:
            mx.random.seed(int(seed))

        # Keep streaming generation usable in headless/non-interactive runs.
        # Playback remains explicitly opt-in via --play.
        play = bool(play)

        if (text is None or text.strip() == "") and ambient_sound:
            text = ambient_sound

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
        has_stream_output_sink = bool(output_path)
        if stream and not play and not has_stream_output_sink:
            raise ValueError(
                "Streaming mode requires at least one sink: enable --play or provide --output_path."
            )

        parsed_model_kwargs_json = _parse_model_kwargs_json(model_kwargs_json)
        dialogue_speakers = _load_dialogue_speakers(dialogue_speakers_json)
        _, effective_preset = _infer_gateway_task_and_preset(
            model=model,
            preset=preset,
            dialogue_speakers=dialogue_speakers,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            instruct=instruct,
            repetition_window=repetition_window,
            passthrough_kwargs=kwargs,
        )

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
            tokens=tokens,
            duration_s=duration_s if duration_s is not None else seconds,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            language=language,
            preset=effective_preset,
            n_vq_for_inference=n_vq_for_inference,
            dialogue_speakers=dialogue_speakers,
            input_type=input_type,
            temperature=temperature,
            repetition_window=repetition_window,
            max_tokens=max_tokens,
            verbose=verbose,
            stream=stream,
            streaming_interval=streaming_interval,
            instruct=instruct,
            **kwargs,
        )
        gen_kwargs.update(parsed_model_kwargs_json)

        if long_form:
            gen_kwargs.update(
                {
                    "long_form": True,
                    "long_form_min_chars": int(long_form_min_chars),
                    "long_form_target_chars": int(long_form_target_chars),
                    "long_form_max_chars": int(long_form_max_chars),
                    "long_form_prefix_audio_seconds": float(
                        long_form_prefix_audio_seconds
                    ),
                    "long_form_prefix_audio_max_tokens": int(
                        long_form_prefix_audio_max_tokens
                    ),
                    "long_form_prefix_text_chars": int(long_form_prefix_text_chars),
                    "long_form_retry_attempts": int(long_form_retry_attempts),
                }
            )

        results = model.generate(**gen_kwargs)

        audio_list = []
        file_name = f"{file_prefix}.{audio_format}"
        for i, result in enumerate(results):
            if play:
                player.queue_audio(result.audio)

            if join_audio and not stream:
                audio_list.append(result.audio)
            if stream and has_stream_output_sink:
                file_name = f"{file_prefix}_{i:03d}.{audio_format}"
                audio_write(
                    file_name,
                    np.array(result.audio),
                    result.sample_rate,
                    format=audio_format,
                )
                print(
                    f"✅ Stream chunk successfully generated and saved as: {file_name}"
                )
            elif not stream and not join_audio:
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
            joined_file_name = f"{file_prefix}.{audio_format}"
            audio = mx.concatenate(audio_list, axis=0)
            audio_write(
                joined_file_name,
                audio,
                model.sample_rate,
                format=audio_format,
            )
            if verbose:
                print(
                    f"✅ Audio successfully generated and saving as: {joined_file_name}"
                )

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


def generate_stream(text: Optional[str], **kwargs) -> None:
    kwargs["stream"] = True
    generate_audio(text=text, **kwargs)


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
        "--tokens",
        type=int,
        default=None,
        help="Target duration control for models that support token-based timing",
    )
    parser.add_argument(
        "--duration_s",
        "--seconds",
        dest="duration_s",
        type=float,
        default=None,
        help=(
            "Convenience duration control in seconds (mapped to tokens at 12.5 Hz; "
            "ignored when --tokens is provided)"
        ),
    )
    parser.add_argument(
        "--n_vq_for_inference",
        type=int,
        default=None,
        help=(
            "Local-only inference depth override (1..n_vq) for quality/performance "
            "trade-offs"
        ),
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
        "--quality", type=str, default=None, help="Quality hint for supported models"
    )
    parser.add_argument(
        "--sound_event",
        type=str,
        default=None,
        help="Sound event description for supported models",
    )
    parser.add_argument(
        "--ambient_sound",
        type=str,
        default=None,
        help="Ambient sound description for supported models",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language hint for supported models",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help=(
            "Variant sampling preset (MOSS family): moss_tts, moss_tts_local, "
            "ttsd, voice_generator, soundeffect, realtime"
        ),
    )
    parser.add_argument(
        "--model_kwargs_json",
        type=str,
        default=None,
        help="JSON object for advanced model.generate kwargs (escape hatch)",
    )
    parser.add_argument(
        "--dialogue_speakers_json",
        type=str,
        default=None,
        help=(
            "Path to TTSD speaker schema JSON "
            "(list of {speaker_id, ref_audio, ref_text/text})"
        ),
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
        "--input_type",
        type=str,
        default="text",
        choices=["text", "pinyin", "ipa"],
        help="Input representation for supported models",
    )
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling paths",
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
        "--repetition_window",
        type=int,
        default=None,
        help=(
            "Realtime repetition-history window size; <=0 disables windowing "
            "and applies repetition penalty over full history"
        ),
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
    parser.add_argument(
        "--long_form",
        action="store_true",
        help="Enable segmented long-form generation for MOSS-TTS variants",
    )
    parser.add_argument(
        "--long_form_min_chars",
        type=int,
        default=160,
        help="Minimum per-segment text budget for long-form planning",
    )
    parser.add_argument(
        "--long_form_target_chars",
        type=int,
        default=320,
        help="Target per-segment text budget for long-form planning",
    )
    parser.add_argument(
        "--long_form_max_chars",
        type=int,
        default=520,
        help="Maximum per-segment text budget for long-form planning",
    )
    parser.add_argument(
        "--long_form_prefix_audio_seconds",
        type=float,
        default=2.0,
        help="Carry-forward tail duration (seconds) between long-form segments",
    )
    parser.add_argument(
        "--long_form_prefix_audio_max_tokens",
        type=int,
        default=25,
        help="Carry-forward tail budget in audio tokens (stricter cap wins)",
    )
    parser.add_argument(
        "--long_form_prefix_text_chars",
        type=int,
        default=0,
        help="Optional carry-forward text window size in characters",
    )
    parser.add_argument(
        "--long_form_retry_attempts",
        type=int,
        default=0,
        help="Retry attempts per long-form segment before failing",
    )

    args = parser.parse_args()

    if args.text is None and args.ambient_sound is not None:
        args.text = args.ambient_sound

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
