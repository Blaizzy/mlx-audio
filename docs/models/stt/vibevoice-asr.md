---
title: VibeVoice ASR
---

# VibeVoice ASR

Microsoft publishes
[VibeVoice-ASR-Streaming-1.5B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B)
and
[VibeVoice-ASR-Streaming-7B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B)
streaming speech-to-text checkpoints. Both emit speaker-attributed text,
accept custom hotwords, and support Chinese, English, French, German, Italian,
Japanese, Korean, Portuguese, Russian, and Spanish.

## Python

```python
from mlx_audio.stt import load

model = load("microsoft/VibeVoice-ASR-Streaming-1.5B")

result = model.generate("meeting.wav", hotwords=["VibeVoice", "MLX"])
print(result.text)
```

For incremental file output:

```python
for chunk in model.stream_transcribe("meeting.wav", context="VibeVoice, MLX"):
    print(chunk, end="", flush=True)
```

## Native streaming geometry

The checkpoint's `preprocessor_config.json` defines the trained streaming
window:

- 22 new speech-token frames per step: 70,400 samples, or 2.933 seconds at 24 kHz
- 4 lookahead frames: 12,800 samples, or 0.533 seconds
- 83,200 samples, or 3.467 seconds, in each encoded window

MLX Audio reads these values at load time. `stream_transcribe()` advances by
the 22-frame chunk while retaining the four-frame lookahead, and keeps one
Qwen KV cache across all chunks.

## Live streams

Use the lower-level state API when audio arrives from a microphone, WebSocket,
or other producer:

```python
import mlx.core as mx

state = model.init_streaming_state(context_info="VibeVoice, MLX")

# Supply overlapping, mono float32 windows at 24 kHz.
features = model.encode_speech(mx.array(window)[None, :])
text, state = model.streaming_generate_step(features, state)
```

Buffer `model.streaming_window_samples` samples, then consume only
`model.streaming_chunk_samples` samples so the lookahead becomes the beginning
of the next window. Zero-pad the last partial window.

## CLI

```bash
python -m mlx_audio.stt.generate \
  --model microsoft/VibeVoice-ASR-Streaming-1.5B \
  --audio meeting.wav \
  --output-path transcript \
  --context "VibeVoice, MLX"
```

The upstream BF16 checkpoints are about 5.6 GB for the 1.5B variant and 17 GB
for the 7B variant. Both load directly because their tensor layouts match the
existing VibeVoice MLX architecture after the standard convolution
transpose/key sanitization pass.
