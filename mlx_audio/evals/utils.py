"""Common utilities for MLX-Audio evaluations."""

from typing import Generator, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.tts.utils import load as load_tts_model
from mlx_audio.utils import load_audio


def inference(
    model: nn.Module,
    text: str,
    voice: Optional[str] = None,
    instruct: Optional[str] = None,
    ref_audio: Optional[mx.array] = None,
    ref_text: Optional[str] = None,
    lang_code: str = "auto",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    speed: float = 1.0,
    verbose: bool = False,
) -> Generator[GenerationResult, None, None]:
    """
    Run TTS inference on a single text input.

    Args:
        model: Loaded TTS model.
        text: Text to synthesize.
        voice: Voice/speaker name for the model.
        instruct: Instruction for style control (CustomVoice/VoiceDesign models).
        ref_audio: Reference audio for voice cloning.
        ref_text: Transcript of reference audio.
        lang_code: Language code (e.g., 'en', 'zh', 'auto').
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        speed: Speech speed multiplier.
        verbose: Whether to print verbose output.

    Returns:
        Generator yielding GenerationResult objects.
    """
    gen_kwargs = dict(
        text=text,
        voice=voice,
        instruct=instruct,
        ref_audio=ref_audio,
        ref_text=ref_text,
        lang_code=lang_code,
        max_tokens=max_tokens,
        temperature=temperature,
        speed=speed,
        verbose=verbose,
        stream=False,
    )

    # Filter out None values
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    return model.generate(**gen_kwargs)


def get_audio_from_result(result: GenerationResult) -> mx.array:
    """Extract audio array from a GenerationResult."""
    return result.audio


def load_reference_audio(
    audio_path: str,
    sample_rate: int = 24000,
    volume_normalize: bool = False,
) -> mx.array:
    """
    Load a reference audio file.

    Args:
        audio_path: Path to the audio file.
        sample_rate: Target sample rate.
        volume_normalize: Whether to normalize volume.

    Returns:
        Audio array.
    """
    return load_audio(
        audio_path,
        sample_rate=sample_rate,
        volume_normalize=volume_normalize,
    )
