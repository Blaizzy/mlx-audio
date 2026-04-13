# Moonshine

MLX implementation of Useful Sensors' Moonshine, a lightweight ASR model that processes raw audio through a learned conv frontend rather than mel spectrograms.

## Available Models

| Model | Parameters | Description |
|-------|------------|-------------|
| [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) | 27M | Smallest variant |
| [UsefulSensors/moonshine-base](https://huggingface.co/UsefulSensors/moonshine-base) | 61M | Larger, more accurate |

## Python Usage

```python
from mlx_audio.stt import load

model = load("UsefulSensors/moonshine-tiny")

result = model.generate("audio.wav")
print(result.text)
```

## Architecture

- 3 layer conv frontend (strides 64, 3, 2) with GroupNorm
- Transformer encoder with RoPE (6 layers tiny, 8 layers base)
- Transformer decoder with cross attention and SwiGLU (6 layers tiny, 8 layers base)
- Byte level BPE tokenizer (32k vocab)
- 16kHz raw audio input (no mel spectrogram)
