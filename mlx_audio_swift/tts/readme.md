# MLX Audio TTS examples for macOS

This is an example of using MLX Audio to run local TTS models.

Steps to run:

 - Open project in Xcode
 - Copy your required files to relevant Resources folder (Kokoro, Orpheus, and Marvis)
 - Change project signing in "Signing and Capabilities" project settings
 - Run the App

 nSpeak framework is embedeed for Kokoro already.

# Kokoro

 - Required files in Kokoro/Resources folder: 
    - kokoro-v1_0.safetensors
    - Voice json files (already in repo)
 
Implemented and working. Based on [Kokoro TTS for iOS](https://github.com/mlalma/kokoro-ios).  All credit to mlalma for that work!

Uses MLX Swift and eSpeak NG.  M1 chip or better is requied.


# Orpheus

Files required from [MLX Community/Orpheus](https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-4bit) and [MLX Community/Snac-24khz](https://huggingface.co/mlx-community/snac_24khz)

Currently runs quite slow due to MLX-Swift not letting us compile layers with caching.  On an M1 we see a 0.1x processing speed so be patient!

 - Required files in Orpheus/Resources folder: 
    - orpheus-3b-0.1-ft-4bit.safetensors
    - config.json
    - model.safetensors.index.json
    - snac_model.safetensors
    - snac_config.json
    - tokenizer_config.json
    - tokenizer.json
    
The full Orpheus functionality is implemented including:
 - Voices: tara, leah, jess, leo, dan, mia, zac, zoe
 - Expressions: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>

# Marvis

Marvis is an advanced conversational TTS model with streaming support. It uses the Marvis architecture combined with Mimi vocoder for high-quality speech synthesis.

Features:
 - Streaming audio generation for real-time TTS
 - Two conversational voices: conversational_a and conversational_b
 - Downloads model weights automatically on first use from Hugging Face
 - Optimized for Apple Silicon with MLX framework

The model runs at 24kHz sample rate and provides natural-sounding conversational speech.
