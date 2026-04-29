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

### Live Input Streaming

When the audio is arriving in real time (mic, WebSocket, network), feed
samples as they become available instead of passing a completed array:

```python
import threading
import numpy as np
from mlx_audio.stt.models.voxtral_realtime.streaming import StreamingAudioSource

source = StreamingAudioSource()

def producer():
    for chunk in capture_microphone_chunks():  # 16 kHz float32 np.ndarray
        source.append(chunk)
    source.close()

threading.Thread(target=producer, daemon=True).start()

for delta in model.generate_streaming(source):
    print(delta, end="", flush=True)
```

For cooperative integrations (async servers, multiple concurrent
streams) use the lower-level session API. `feed()` is a cheap
thread-safe queue push; `step(max_decode_tokens=N)` runs a bounded
unit of MLX work and returns the deltas emitted during that call.
This is cooperative scheduling, not parallel MLX execution: a server
can alternate short `step()` calls across sessions on the same
executor, but the actual MLX work should still be serialized per
process/device. This works well when each session can process audio
faster than real time, leaving room for another stream — or an LLM
request — to run before the next audio chunk arrives:

```python
session = model.create_streaming_session()

# Producer side (any thread or asyncio task):
session.feed(samples)           # cheap, non-blocking
session.close()                 # end-of-stream signal

# Consumer side (MLX executor thread):
while not session.done:
    for delta in session.step(max_decode_tokens=4):
        emit(delta)
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
