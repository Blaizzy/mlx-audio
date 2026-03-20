# Canary

MLX implementation of NVIDIA's Canary-1B-v2, a multilingual ASR model supporting transcription and translation across 25 EU languages plus Russian and Ukrainian.

## Python Usage

```python
from mlx_audio.stt import load

model = load("path/to/canary-1b-v2-mlx")

# Transcribe English audio
result = model.generate("audio.wav", source_lang="en", target_lang="en")
print(result.text)

# Transcribe German audio
result = model.generate("audio.wav", source_lang="de", target_lang="de")
print(result.text)

# Translate English audio to German
result = model.generate("audio.wav", source_lang="en", target_lang="de")
print(result.text)
```

## Supported Languages

Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish, Russian, Ukrainian.

## Architecture

- FastConformer encoder (32 layers, reused from Parakeet)
- Transformer decoder with cross-attention (8 layers)
- SentencePiece tokenizer (16,384 tokens)
- 16kHz audio input with per-feature normalized mel spectrogram

## Weight Conversion

Weights need to be converted from NVIDIA's `.nemo` format to MLX safetensors. See the [conversion scripts](https://github.com/mm65x/asr-model-conversions) for details.
