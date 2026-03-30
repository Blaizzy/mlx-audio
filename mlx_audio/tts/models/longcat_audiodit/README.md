# LongCat-AudioDiT

State-of-the-art diffusion-based text-to-speech that operates directly in the waveform latent space. Uses Conditional Flow Matching with a DiT (Diffusion Transformer) backbone and a WAV-VAE audio codec at 24kHz. Supports zero-shot voice cloning with SOTA speaker similarity on the Seed benchmark.

**Paper:** [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/LongCat-AudioDiT.pdf)

## Usage

Python API:

```python
from mlx_audio.tts.utils import load

model = load("meituan-longcat/LongCat-AudioDiT-1B")

result = next(model.generate("Hello, this is a test of AudioDiT."))
audio = result.audio  # mlx array, 24kHz
```

Play audio directly:

```python
import sounddevice as sd

result = next(model.generate("The quick brown fox jumps over the lazy dog."))
sd.play(result.audio, result.sample_rate)
sd.wait()
```

## Voice Cloning

Clone any voice using a reference audio sample and its transcript. Use `guidance_method="apg"` for best voice cloning quality:

```python
result = next(model.generate(
    text="Today is warm turning to rain, with good air quality.",
    ref_audio="reference.wav",
    ref_text="Transcript of the reference audio.",
    guidance_method="apg",
    cfg_strength=4.0,
    steps=16,
))
```

## Zero-Shot Generation (Chinese)

```python
result = next(model.generate(
    text="今天晴暖转阴雨，空气质量优至良，空气相对湿度较低。",
    steps=16,
    cfg_strength=4.0,
))
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps` | 16 | Euler ODE solver steps. Higher = better quality, slower |
| `cfg_strength` | 4.0 | Classifier-free guidance strength |
| `guidance_method` | `"cfg"` | `"cfg"` for TTS, `"apg"` for voice cloning |
| `seed` | 1024 | Random seed for reproducibility |
| `ref_audio` | `None` | Reference audio for voice cloning (24kHz) |
| `ref_text` | `None` | Transcript of the reference audio |

## CLI

```bash
# Zero-shot TTS
python -m mlx_audio.tts.generate \
  --model meituan-longcat/LongCat-AudioDiT-1B \
  --text "Hello, this is a test of AudioDiT." \
  --play

# Voice cloning
python -m mlx_audio.tts.generate \
  --model meituan-longcat/LongCat-AudioDiT-1B \
  --text "Today is warm turning to rain." \
  --ref_audio reference.wav \
  --ref_text "Transcript of the reference audio." \
  --play
```

## Available Models

| Model | Parameters | Languages |
|-------|-----------|-----------|
| `meituan-longcat/LongCat-AudioDiT-1B` | 1B | Chinese, English |
| `meituan-longcat/LongCat-AudioDiT-3.5B` | 3.5B | Chinese, English |

## Architecture

- **DiT backbone:** dim=1536, depth=24, heads=24 with RoPE and AdaLN
- **WAV-VAE codec:** latent_dim=64, 24kHz, runs in fp16
- **UMT5 text encoder:** 768-dim, 12 layers with per-layer relative position bias
- **Conditional Flow Matching** with Euler ODE solver

## License

LongCat-AudioDiT weights and code are released under the [MIT License](https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/LICENSE).
