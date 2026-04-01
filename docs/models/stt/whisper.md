---
title: Whisper
---

# Whisper

[Whisper](https://github.com/openai/whisper) is OpenAI's robust speech-to-text model supporting 99+ languages. The MLX Audio implementation also natively supports distilled variants like [Distil-Whisper](https://huggingface.co/distil-whisper/distil-large-v3).

## Available Models

| Model | Parameters | Description | Repo |
|-------|-----------|-------------|------|
| **whisper-large-v3-turbo** | ~809M | Fastest large model, multilingual | [mlx-community/whisper-large-v3-turbo-asr-fp16](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-fp16) |
| **whisper-large-v3** | ~1.5B | Best accuracy, multilingual | [mlx-community/whisper-large-v3](https://huggingface.co/mlx-community/whisper-large-v3) |
| **distil-large-v3** | ~756M | Distilled, English-focused | [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) |

## Python Usage

### Basic Transcription

```python
from mlx_audio.stt import load

# Standard Whisper
model = load("mlx-community/whisper-large-v3-turbo-asr-fp16")

# Distil-Whisper
# model = load("distil-whisper/distil-large-v3")

result = model.generate("audio.wav")
print(result.text)
```

### Segment-Level Timestamps

Segment-level timestamps are enabled by default:

```python
result = model.generate("audio.wav")
for segment in result.segments:
    print(f"[{segment['start']:.2f} -> {segment['end']:.2f}] {segment['text']}")
```

### Word-Level Timestamps

Word-level timestamps use cross-attention alignment heads via DTW:

```python
result = model.generate("audio.wav", word_timestamps=True)
for segment in result.segments:
    for word in segment["words"]:
        print(f"[{word['start']:.2f} -> {word['end']:.2f}] {word['word']} (p={word['probability']:.3f})")
```

### Disable Timestamps

```python
result = model.generate("audio.wav", return_timestamps=False)
```

## CLI Usage

=== "Basic transcription"

    ```bash
    mlx_audio.stt.generate \
      --model mlx-community/whisper-large-v3-turbo-asr-fp16 \
      --audio audio.wav \
      --verbose
    ```

=== "Word-level timestamps"

    ```bash
    mlx_audio.stt.generate \
      --model mlx-community/whisper-large-v3-turbo-asr-fp16 \
      --audio audio.wav \
      --verbose \
      --gen-kwargs '{"word_timestamps": true}'
    ```

=== "Export JSON with word timestamps"

    ```bash
    mlx_audio.stt.generate \
      --model mlx-community/whisper-large-v3-turbo-asr-fp16 \
      --audio audio.wav \
      --format json \
      --output-path output.json \
      --gen-kwargs '{"word_timestamps": true}'
    ```

!!! info "Language support"
    Whisper supports 99+ languages. The model automatically detects the spoken language, or you can specify it explicitly for better accuracy.

!!! tip "Distil-Whisper"
    Distil-Whisper variants are faster and smaller than the full Whisper models while maintaining strong accuracy for English transcription. Load them the same way -- just swap the model name.
