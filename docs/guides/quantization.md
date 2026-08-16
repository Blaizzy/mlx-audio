# Quantization

Quantization reduces model size and can improve inference speed by storing weights at lower precision. MLX Audio supports several quantization formats through the `mlx_audio.convert` module.

## Why Quantize?

| Benefit | Details |
|---------|---------|
| **Smaller models** | A 4-bit model is roughly 4x smaller than float16 |
| **Faster inference** | Lower-precision weights move through memory faster on Apple Silicon |
| **Lower memory usage** | Fit larger models into unified memory |

The trade-off is a potential reduction in audio quality, especially at very low bit widths.

## Available Bit Widths

| Bits | Typical Use Case | Quality | Size Reduction |
|------|------------------|---------|----------------|
| **3-bit** | Maximum compression, acceptable for quick prototyping | Lower | ~5x vs fp16 |
| **4-bit** | Good balance of quality and size | Good | ~4x vs fp16 |
| **6-bit** | Near-lossless for most models | Very good | ~2.5x vs fp16 |
| **8-bit** | Minimal quality loss | Excellent | ~2x vs fp16 |

## Quantization Modes

MLX Audio supports several quantization modes:

| Mode | Description |
|------|-------------|
| `affine` | Standard affine quantization (default) |
| `mxfp4` | Microscaling FP4 format |
| `mxfp8` | Microscaling FP8 format |
| `nvfp4` | NVIDIA FP4 format |

## Converting and Quantizing Models

Use the `mlx_audio.convert` CLI to convert a HuggingFace model to MLX format with quantization:

### Basic 4-bit Quantization

```bash
python -m mlx_audio.convert \
    --hf-path prince-canuma/Kokoro-82M \
    --mlx-path ./Kokoro-82M-4bit \
    --quantize \
    --q-bits 4
```

### MXFP4 Quantization

```bash
python -m mlx_audio.convert \
    --hf-path prince-canuma/Kokoro-82M \
    --mlx-path ./Kokoro-82M-mxfp4 \
    --quantize \
    --q-mode mxfp4
```

### Convert to bfloat16 (No Quantization)

```bash
python -m mlx_audio.convert \
    --hf-path prince-canuma/Kokoro-82M \
    --mlx-path ./Kokoro-82M-bf16 \
    --dtype bfloat16
```

### Upload to Hugging Face Hub

Add `--upload-repo` to push the converted model directly:

```bash
python -m mlx_audio.convert \
    --hf-path prince-canuma/Kokoro-82M \
    --mlx-path ./Kokoro-82M-4bit \
    --quantize \
    --q-bits 4 \
    --upload-repo username/Kokoro-82M-4bit
```

## Conversion Options Reference

| Flag | Description |
|------|-------------|
| `--hf-path` | Source HuggingFace model or local path |
| `--mlx-path` | Output directory for the converted model |
| `-q`, `--quantize` | Enable quantization |
| `--q-bits` | Bits per weight (e.g., 3, 4, 6, 8) |
| `--q-group-size` | Group size for quantization (defaults depend on mode) |
| `--q-mode` | Quantization mode: `affine`, `mxfp4`, `mxfp8`, `nvfp4` |
| `--dtype` | Weight dtype when not quantizing: `float16`, `bfloat16`, `float32` |
| `--upload-repo` | Upload converted model to HuggingFace Hub |

## Using Pre-Quantized Models

The [mlx-community](https://huggingface.co/mlx-community) organization on Hugging Face hosts many pre-quantized models ready to use. For example:

```python
from mlx_audio.tts.utils import load_model

# Load a pre-quantized model -- no conversion needed
model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit")

results = list(model.generate(
    text="This is running from a 6-bit quantized model!",
    voice="serena",
))
```

```python
from mlx_audio.stt.utils import load

# 4-bit Voxtral for faster STT
model = load("mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")
result = model.generate("audio.wav")
print(result.text)
```

## Quality vs. Performance Trade-offs

!!! info "General guidance"
    Start with **4-bit** or **6-bit** quantization. Move to 8-bit only if you hear artifacts, or to 3-bit if you need the smallest possible model.

- **TTS models** tend to be sensitive to quantization at 3-bit. 4-bit is usually the sweet spot.
- **STT models** (e.g., Whisper, Voxtral Realtime) often tolerate 4-bit quantization with minimal accuracy loss.
- **Larger models** (1B+ parameters) generally tolerate lower bit widths better than smaller ones.
- Always **listen to output** when evaluating TTS quantization -- word error rate alone does not capture prosody degradation.

## Mixed Quantization Recipes

MLX Audio also supports mixed-precision quantization recipes that apply different bit widths to different layers:

| Recipe | Description |
|--------|-------------|
| `mixed_2_6` | 2-bit for some layers, 6-bit for others |
| `mixed_3_4` | 3-bit / 4-bit mix |
| `mixed_3_6` | 3-bit / 6-bit mix |
| `mixed_4_6` | 4-bit / 6-bit mix |

These can provide better quality-to-size ratios by keeping critical layers at higher precision.
