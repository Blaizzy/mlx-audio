# Canary

MLX implementation of NVIDIA's Canary-1B-v2, a multilingual ASR model supporting transcription and translation across 25 EU languages plus Russian and Ukrainian.

## Python Usage

```python
from mlx_audio.stt import load

# A local path or a HuggingFace repo id both work, e.g.:
#   load("qfuxa/canary-mlx")                  # full precision
#   load("Mediform/canary-1b-v2-mlx-q8")      # 8-bit quantized
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

### Supported checkpoint layouts

The loader auto-detects and supports two on-disk layouts:

- **NeMo-native** – raw NeMo weight names (`transf_decoder._decoder.layers.*`,
  `log_softmax.mlp.layer0.*`) with PyTorch conv layout. Conv kernels are
  transposed to MLX's `(out, *kernel, in)` order on load.
- **MLX-native** – conversions that already store MLX tensor layouts and use
  flattened names (`transf_decoder.layers.N.first_sub_layer.linear_q`,
  `head.classifier`, `transf_decoder.token_embedding`). Conv kernels are loaded
  as-is. Community checkpoints such as
  [`qfuxa/canary-mlx`](https://huggingface.co/qfuxa/canary-mlx) and
  [`Mediform/canary-1b-v2-mlx-q8`](https://huggingface.co/Mediform/canary-1b-v2-mlx-q8)
  use this layout.

The SentencePiece tokenizer is loaded from a `tokenizer.model` file, a
`tokens.txt` mapping, or — when neither is present — from a base64 blob embedded
in `config.json` under `tokenizer.model_base64`. 8-bit quantized checkpoints
(with a `"quantization"` block in `config.json`) are loaded transparently.
