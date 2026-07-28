# Higgs-Audio v3 STT

[Higgs-Audio v3 STT](https://huggingface.co/bosonai/higgs-audio-v3-stt) is a speech-to-text model from Boson AI that pairs a Whisper-style audio encoder with a Qwen3 text backbone for transcription.

## Quick Start

```python
from mlx_audio.stt.utils import load_model

model = load_model("bosonai/higgs-audio-v3-stt")

result = model.generate("audio.wav")
print(result.text)
```

## CLI

```bash
mlx_audio.stt.generate --model bosonai/higgs-audio-v3-stt --audio audio.wav
```

## Models

| Model | Size | Quantization | HuggingFace |
|-------|------|-------------|-------------|
| higgs-audio-v3-stt | ~5.35GB | bf16 | [bosonai/higgs-audio-v3-stt](https://huggingface.co/bosonai/higgs-audio-v3-stt) |

## Architecture

- **Audio Encoder**: Whisper-style encoder (32 transformer layers, 1280-dim) + AvgPool1d
- **Projector**: depthwise temporal Conv1d (stride 2) + 2-layer MLP with ReLU (1280 -> 2048)
- **LLM**: Qwen3 (28 layers, 2048 hidden)

Audio is converted to a 128-channel mel spectrogram at 16kHz, segmented with a Silero VAD into chunks of at most `chunk_size_seconds`, encoded and projected per chunk, and merged into the `<|AUDIO|>` placeholder positions of the text prompt before autoregressive generation. When the VAD is unavailable the audio is split into fixed-length chunks.
