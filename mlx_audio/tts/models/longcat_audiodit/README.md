# LongCat-AudioDiT

State-of-the-art diffusion-based text-to-speech that operates directly in the waveform latent space. Uses Conditional Flow Matching with a DiT (Diffusion Transformer) backbone and a WAV-VAE audio codec at 24kHz. Supports zero-shot voice cloning with SOTA speaker similarity on the Seed benchmark.

**Paper:** [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/LongCat-AudioDiT.pdf)

## Usage

Python API:

```python
from mlx_audio.tts.utils import load

model = load("mlx-community/LongCat-AudioDiT-1B-bf16")

result = next(model.generate("Hello, this is a test of AudioDiT."))
audio = result.audio  # mlx array, 24kHz
```

Play audio directly:

```python
from mlx_audio.tts.audio_player import AudioPlayer

player = AudioPlayer(sample_rate=24000)
result = next(model.generate("The quick brown fox jumps over the lazy dog."))
player.queue_audio(result.audio)
player.wait_for_drain()
player.stop()
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
  --model mlx-community/LongCat-AudioDiT-1B-bf16 \
  --text "Hello, this is a test of AudioDiT." \
  --play

# Voice cloning
python -m mlx_audio.tts.generate \
  --model mlx-community/LongCat-AudioDiT-1B-bf16 \
  --text "Today is warm turning to rain." \
  --ref_audio reference.wav \
  --ref_text "Transcript of the reference audio." \
  --play
```

## Available Models

| Model | Parameters | Format | Languages |
|-------|-----------|--------|-----------|
| `mlx-community/LongCat-AudioDiT-1B-bf16` | 1B | bf16 | Chinese, English |
| `mlx-community/LongCat-AudioDiT-1B-8bit` | 1B | 8-bit | Chinese, English |
| `mlx-community/LongCat-AudioDiT-1B-6bit` | 1B | 6-bit | Chinese, English |
| `mlx-community/LongCat-AudioDiT-1B-5bit` | 1B | 5-bit | Chinese, English |
| `mlx-community/LongCat-AudioDiT-1B-4bit` | 1B | 4-bit | Chinese, English |
| `mlx-community/LongCat-AudioDiT-1B-mxfp8` | 1B | MXFP8 | Chinese, English |
| `mlx-community/LongCat-AudioDiT-1B-mxfp4` | 1B | MXFP4 | Chinese, English |
| `mlx-community/LongCat-AudioDiT-1B-nvfp4` | 1B | NVFP4 | Chinese, English |
| `mlx-community/LongCat-AudioDiT-3.5B-bf16` | 3.5B | bf16 | Chinese, English |
| `mlx-community/LongCat-AudioDiT-3.5B-8bit` | 3.5B | 8-bit | Chinese, English |
| `mlx-community/LongCat-AudioDiT-3.5B-6bit` | 3.5B | 6-bit | Chinese, English |
| `mlx-community/LongCat-AudioDiT-3.5B-5bit` | 3.5B | 5-bit | Chinese, English |
| `mlx-community/LongCat-AudioDiT-3.5B-4bit` | 3.5B | 4-bit | Chinese, English |
| `mlx-community/LongCat-AudioDiT-3.5B-mxfp8` | 3.5B | MXFP8 | Chinese, English |
| `mlx-community/LongCat-AudioDiT-3.5B-mxfp4` | 3.5B | MXFP4 | Chinese, English |
| `mlx-community/LongCat-AudioDiT-3.5B-nvfp4` | 3.5B | NVFP4 | Chinese, English |

## Architecture

- **DiT backbone:** dim=1536, depth=24, heads=24 with RoPE and AdaLN
- **WAV-VAE codec:** latent_dim=64, 24kHz, runs in fp16
- **UMT5 text encoder:** 768-dim, 12 layers with per-layer relative position bias
- **Conditional Flow Matching** with Euler ODE solver

## License

LongCat-AudioDiT weights and code are released under the [MIT License](https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/LICENSE).
