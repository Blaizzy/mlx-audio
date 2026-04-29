# Quick Start: CLI

mlx-audio provides command-line tools for both text-to-speech generation and speech-to-text transcription.

## Text-to-Speech

!!! note
    These TTS quickstart examples use `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`.

### Basic Generation

Generate speech from text with a single command:

```bash
mlx_audio.tts.generate \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
    --text "Hello, world!" \
    --voice Chelsie \
    --lang_code English
```

### Play Audio Immediately

Add `--play` to hear the result without saving:

```bash
mlx_audio.tts.generate \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
    --text "Hello, world!" \
    --voice Chelsie \
    --lang_code English \
    --play
```

### Voice and Language Selection

Choose a voice preset and provide a language hint:

```bash
mlx_audio.tts.generate \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
    --text "Welcome to MLX-Audio!" \
    --voice Ethan \
    --lang_code English
```

### Save to a Directory

```bash
mlx_audio.tts.generate \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
    --text "Hello!" \
    --voice Chelsie \
    --lang_code English \
    --output_path ./my_audio
```

### Voice Cloning (CSM)

Clone a voice from a reference audio file:

```bash
mlx_audio.tts.generate \
    --model mlx-community/csm-1b \
    --text "Hello from Sesame." \
    --ref_audio ./reference_voice.wav \
    --play
```

## Speech-to-Text

### Transcribe with Whisper

```bash
python -m mlx_audio.stt.generate \
    --model mlx-community/whisper-large-v3-turbo-asr-fp16 \
    --audio audio.wav \
    --output-path output \
    --format json \
    --verbose
```

### Transcribe with Parakeet

```bash
python -m mlx_audio.stt.generate \
    --model mlx-community/parakeet-tdt-0.6b-v3 \
    --audio speech.wav \
    --output-path output \
    --format json \
    --verbose
```

### Transcribe with VibeVoice-ASR

```bash
python -m mlx_audio.stt.generate \
    --model mlx-community/VibeVoice-ASR-bf16 \
    --audio meeting.wav \
    --output-path output \
    --format json \
    --max-tokens 8192 \
    --verbose
```

Add context/hotwords for better accuracy on domain-specific terms:

```bash
python -m mlx_audio.stt.generate \
    --model mlx-community/VibeVoice-ASR-bf16 \
    --audio technical_talk.wav \
    --output-path output \
    --format json \
    --max-tokens 8192 \
    --context "MLX, Apple Silicon, PyTorch, Transformer" \
    --verbose
```

## Common Flags

| Flag | Description |
|------|-------------|
| `--model` | Hugging Face model ID or local path |
| `--text` | Input text for TTS generation |
| `--audio` | Input audio file for STT transcription |
| `--voice` | Voice preset name (e.g., `Chelsie`, `Ethan`, `casual_male`) |
| `--speed` | Speech speed multiplier (default: `1.0`) |
| `--lang_code` | Language hint (e.g., `English`, `Chinese`, or `auto`) |
| `--play` | Play audio immediately after generation |
| `--output_path` | Directory to save output files |
| `--verbose` | Show detailed generation info |
| `--format` | Output format for STT (`json`, etc.) |

!!! note
    Different models support different flags. Check the [Models](../models/index.md) section for model-specific options.
