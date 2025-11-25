# MLX Audio TTS

A Swift library and example app for running local TTS models on Apple Silicon using MLX.

**Requirements:**
- macOS 15.4+ or iOS 18.4+

## Running the Example App

1. Open `MLXAudio.xcodeproj` in Xcode
2. Change project signing in "Signing and Capabilities" project settings
3. Select your target (macOS or iOS)
4. Run the app

All models download their weights automatically from Hugging Face Hub on first use.

eSpeak NG phoneme data is provided via the espeak-ng-spm Swift Package.

## Kokoro

Model weights are downloaded automatically from [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16)

Based on [Kokoro TTS for iOS](https://github.com/mlalma/kokoro-ios).

## Orpheus

Model weights are downloaded automatically from:
- [mlx-community/orpheus-3b-0.1-ft-4bit](https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-4bit)
- [mlx-community/snac_24khz](https://huggingface.co/mlx-community/snac_24khz)

The full Orpheus functionality is implemented including:
 - Voices: tara, leah, jess, leo, dan, mia, zac, zoe
 - Expressions: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>

## Marvis

Marvis is an advanced conversational TTS model with streaming support. It uses the Marvis architecture combined with Mimi vocoder for high-quality speech synthesis.

Features:
 - Streaming audio generation for real-time TTS
 - Two conversational voices: conversational_a and conversational_b

The model runs at 24kHz sample rate and provides natural-sounding conversational speech.
