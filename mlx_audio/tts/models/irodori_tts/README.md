# Irodori TTS

Japanese text-to-speech model based on Echo TTS architecture, ported to MLX.
Uses Rectified Flow diffusion with a DiT (Diffusion Transformer) and DACVAE codec (48kHz).

## Model

Original: [Aratako/Irodori-TTS-500M](https://huggingface.co/Aratako/Irodori-TTS-500M) (500M parameters)

## Usage

Python API:

```python
from mlx_audio.tts import load

model = load("mlx-community/Irodori-TTS-500M-fp16")
result = next(model.generate("こんにちは、音声合成のテストです。"))
audio = result.audio
```

With reference audio for voice cloning:

```python
result = next(model.generate(
    "こんにちは、音声合成のテストです。",
    ref_audio="speaker.wav",
))
```

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Irodori-TTS-500M-fp16 \
  --text "こんにちは、音声合成のテストです。"
```

## Memory requirements

The default `sequence_length=750` requires approximately 24GB of unified memory.
On 16GB machines, use reduced settings:

```python
result = next(model.generate(
    "こんにちは。",
    sequence_length=300,       # ~9GB
    cfg_guidance_mode="alternating",  # ~1/3 of independent mode memory
))
```

Approximate memory usage with `cfg_guidance_mode="alternating"`:

| sequence_length | Memory | Audio length |
|---|---|---|
| 100 | ~2GB | ~4s |
| 300 | ~2GB | ~12s |
| 400 | ~3GB | ~16s |

With `cfg_guidance_mode="independent"` (default), multiply memory by ~3.

## Notes

- Input language: Japanese. Latin characters may not be pronounced correctly;
  convert them to katakana beforehand (e.g. "MLX" → "エムエルエックス").
- The DACVAE codec weights (`facebook/dacvae-watermarked`) are automatically
  downloaded on first use.

## License

Irodori-TTS weights are released under the [MIT License](https://opensource.org/licenses/MIT).
See [Aratako/Irodori-TTS-500M](https://huggingface.co/Aratako/Irodori-TTS-500M) for details.
