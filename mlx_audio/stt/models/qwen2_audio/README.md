# Qwen2-Audio

[Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) is a multimodal audio-language model from Alibaba that goes beyond speech-to-text. It supports transcription, translation, audio captioning, emotion detection, sound classification, and general audio understanding.

## Quick Start

```python
from mlx_audio.stt.utils import load_model

model = load_model("mlx-community/Qwen2-Audio-7B-Instruct-4bit")

# Transcription
result = model.generate("audio.wav", prompt="Transcribe the audio.")
print(result.text)

# Audio understanding
result = model.generate("audio.wav", prompt="What emotion is the speaker expressing?")
print(result.text)

# Translation
result = model.generate("audio.wav", prompt="Translate the speech to French.")
print(result.text)
```

## CLI

```bash
mlx_audio.stt.generate --model mlx-community/Qwen2-Audio-7B-Instruct-4bit \
    --audio audio.wav \
    --prompt "Transcribe the audio."
```

## Models

| Model | Size | Quantization | HuggingFace |
|-------|------|-------------|-------------|
| Qwen2-Audio-7B-Instruct | ~15GB | bf16 | [Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) |
| Qwen2-Audio-7B-Instruct-4bit | ~4.2GB | 4-bit | [mlx-community/Qwen2-Audio-7B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2-Audio-7B-Instruct-4bit) |

## Architecture

- **Audio Encoder**: Whisper-large-v3 encoder (32 transformer layers, 1280-dim) + AvgPool1d
- **Projector**: Linear layer (1280 -> 4096)
- **LLM**: Qwen2-7B (32 layers, 4096 hidden)

Audio is converted to a 128-channel mel spectrogram at 16kHz, encoded, projected, and merged with text token embeddings before autoregressive generation.

## Capabilities

- Speech transcription (ASR)
- Speech translation
- Audio captioning
- Emotion / sentiment detection
- Environmental sound classification
- Music understanding
- Voice chat (audio-only, no text prompt needed)
