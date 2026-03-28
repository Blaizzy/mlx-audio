# KugelAudio

Open-source 7B TTS model for 24 European languages, based on Microsoft VibeVoice.
Uses a hybrid AR + Diffusion architecture (Qwen2.5 LM + SDE-DPM-Solver++ + VAE decoder).

Original: [kugelaudio/kugelaudio-0-open](https://huggingface.co/kugelaudio/kugelaudio-0-open)

## Model

| Repo ID | Parameters | Description |
|---------|-----------|-------------|
| `kugelaudio/kugelaudio-0-open` | 7B | Original weights (bfloat16, loaded directly) |

## Usage

Python API:

```python
from mlx_audio.tts import load

model = load("kugelaudio/kugelaudio-0-open")

result = next(model.generate(
    text="Hello, this is KugelAudio running on Apple Silicon.",
    cfg_scale=3.0,
    ddpm_steps=10,
))
audio = result.audio  # mx.array, 24kHz
```

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model kugelaudio/kugelaudio-0-open \
  --text "Hello, this is KugelAudio." \
  --cfg_scale 3.0
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cfg_scale` | 3.0 | Classifier-free guidance strength (1.0 = no CFG, faster) |
| `ddpm_steps` | 10 | Diffusion sampling steps (5 = fast, 10 = balanced, 20 = max quality) |
| `max_tokens` | 2048 | Maximum speech tokens to generate |

## Supported languages (24)

English, German, French, Spanish, Italian, Portuguese, Dutch, Polish, Russian,
Ukrainian, Czech, Romanian, Hungarian, Swedish, Danish, Finnish, Norwegian,
Greek, Bulgarian, Slovak, Croatian, Serbian, Turkish

Quality varies by language. English, German, French, and Spanish have the
strongest training data coverage.

## Conversion

The model loads original PyTorch safetensors directly (weights are remapped
via `sanitize()`). To quantize or save in a pre-converted format:

```bash
python -m mlx_audio.convert \
  --hf-path kugelaudio/kugelaudio-0-open \
  --mlx-path ./kugelaudio-0-open-bf16 \
  --dtype bfloat16
```

## Memory requirements

Requires approximately 17GB of unified memory (7B parameters in bfloat16).
Tested on M4 Max 36GB.

## Notes

- Pre-encoded voice presets are not yet available in the upstream model;
  the model generates with a default voice.
- RTF is approximately 5-7x with `cfg_scale=3.0` and `ddpm_steps=10` on M4 Max.

## License

KugelAudio is released under the [MIT License](https://opensource.org/licenses/MIT).
