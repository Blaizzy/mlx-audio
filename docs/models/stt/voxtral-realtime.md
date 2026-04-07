---
title: Voxtral Realtime
---

# Voxtral Realtime

[Voxtral Realtime](https://huggingface.co/mistralai) is Mistral's 4B parameter streaming speech-to-text model, optimized for low-latency transcription.

## Available Variants

| Variant | Precision | Size | Use Case | Repo |
|---------|-----------|------|----------|------|
| **4bit** | 4-bit quantized | Smaller | Faster inference, lower memory | [mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit) |
| **fp16** | Full precision | Larger | Maximum accuracy | [mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16) |

## Python Usage

### Basic Transcription

```python
from mlx_audio.stt.utils import load

# Use 4bit for faster inference, fp16 for full precision
model = load("mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")

# Transcribe audio
result = model.generate("audio.wav")
print(result.text)
```

### Streaming Transcription

```python
for chunk in model.generate("audio.wav", stream=True):
    print(chunk, end="", flush=True)
```

### Adjusting Transcription Delay

Lower delay values produce faster output but may reduce accuracy:

```python
result = model.generate("audio.wav", transcription_delay_ms=240)
```

!!! tip "4bit vs fp16"
    The **4bit** variant offers significantly faster inference and lower memory usage, making it ideal for real-time applications. Use **fp16** when maximum transcription accuracy is required.

!!! info "Streaming-first design"
    Voxtral Realtime is specifically designed for streaming use cases. The `stream=True` mode delivers transcription chunks as they become available, enabling real-time applications like live captioning.
