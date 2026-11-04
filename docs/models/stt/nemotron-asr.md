---
title: Nemotron ASR Streaming
---

# Nemotron ASR Streaming

NVIDIA's 0.6B streaming RNN-T models with language-prompt conditioning, in `mlx_audio.stt.models.nemotron_asr`:

- [Nemotron 3.5 ASR Streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b): cache-aware streaming, 40 language-locales.
- [Nemotron Unified ASR](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b): one checkpoint for offline and streaming transcription in 25 European languages, at 160 ms to 2.08 s latency. It streams buffered: each step re-encodes 5.6 s of left context with the new chunk.

Both transcribe with punctuation and capitalization. The [module README](https://github.com/Neywa-Labs/mlx-audio/blob/main/mlx_audio/stt/models/nemotron_asr/README.md) covers conversion, speaker-masked transcription and the latency rules.

## Available Models

| Model | Precision | Repo |
|-------|-----------|------|
| **Nemotron 3.5 ASR Streaming** | bf16 | [mlx-community/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b) |
| **Nemotron 3.5 ASR Streaming** | 8-bit | [mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit) |
| **Nemotron Unified ASR** | bf16 | [mlx-community/nemotron-labs-unified-asr-0.6b-bf16](https://huggingface.co/mlx-community/nemotron-labs-unified-asr-0.6b-bf16) |
| **Nemotron Unified ASR** | 8-bit | [mlx-community/nemotron-labs-unified-asr-0.6b-8bit](https://huggingface.co/mlx-community/nemotron-labs-unified-asr-0.6b-8bit) |
| **Nemotron Unified ASR** | 4-bit | [mlx-community/nemotron-labs-unified-asr-0.6b-4bit](https://huggingface.co/mlx-community/nemotron-labs-unified-asr-0.6b-4bit) |

## Python Usage

```python
from mlx_audio.stt import load

model = load("mlx-community/nemotron-labs-unified-asr-0.6b-8bit")

# Offline
print(model.generate("speech.wav", language="fr").text)

# Live: feed 16 kHz mono PCM, step to get text deltas
session = model.create_streaming_session(language="fr", transcription_delay_ms=320)
for pcm in audio_chunks:
    session.feed(pcm)
    print("".join(session.step()), end="", flush=True)
session.close()
while not session.done:
    print("".join(session.step()), end="", flush=True)
```

`language` takes a locale key (`en-US`), a bare code matching one key (`en`), or `auto`. For Nemotron Unified, `transcription_delay_ms` is rounded to an 80 ms frame within 160-2080 ms and mapped to NVIDIA's chunk table.

## CLI Usage

```bash
python -m mlx_audio.stt.generate \
  --model mlx-community/nemotron-labs-unified-asr-0.6b-8bit \
  --audio speech.wav --format txt
```
