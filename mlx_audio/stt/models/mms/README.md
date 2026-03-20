# MMS ASR

MLX implementation of Meta's Massively Multilingual Speech (MMS) ASR model, supporting 1000+ languages through language-specific adapter layers on top of a shared wav2vec2 backbone.

## Available Models

| Model | Parameters | Languages | Description |
|-------|------------|-----------|-------------|
| [facebook/mms-1b-fl102](https://huggingface.co/facebook/mms-1b-fl102) | 1B | 102 | Finetuned on FLEURS |
| [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | 1B | 1162 | All supported languages |

## Python Usage

```python
from mlx_audio.stt import load

model = load("facebook/mms-1b-fl102")

result = model.generate("audio.wav")
print(result.text)
```

## Architecture

- Wav2Vec2 encoder with convolutional feature extractor
- Per-layer attention adapter modules for language adaptation
- CTC head with language-specific vocabulary
- Adapter weights loaded automatically from `adapter.{lang}.safetensors`
- 16kHz audio input with zero-mean unit-variance normalization
