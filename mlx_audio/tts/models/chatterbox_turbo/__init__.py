# Copyright (c) 2025 Resemble AI
# MIT License
# Chatterbox TTS - MLX Port
# Optimized for Apple Silicon

"""
Chatterbox MLX - Text-to-Speech on Apple Silicon

This is an MLX port of the Chatterbox TTS model, optimized for Apple Silicon.

Example usage:
    from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS

    # Load model
    model = ChatterboxTurboTTS.from_pretrained()

    # Generate speech
    wav = model.generate(
        "Hello, this is a test of Chatterbox TTS on Apple Silicon!",
        audio_prompt_path="reference.wav"  # Optional: for voice cloning
    )

    # Save output
    import soundfile as sf
    sf.write("output.wav", wav[0].numpy(), model.sr)
"""

from .chatterbox_turbo import ChatterboxTurboTTS, Conditionals, punc_norm
from .models.s3gen import S3GEN_SIL, S3GEN_SR, S3Gen
from .models.t3 import T3, T3Cond, T3Config
from .models.voice_encoder import VoiceEncoder

__version__ = "0.1.0"

# Alias for load_model compatibility
Model = ChatterboxTurboTTS

__all__ = [
    "ChatterboxTurboTTS",
    "Model",
    "Conditionals",
    "punc_norm",
    "T3",
    "T3Config",
    "T3Cond",
    "S3Gen",
    "S3GEN_SR",
    "S3GEN_SIL",
    "VoiceEncoder",
]
