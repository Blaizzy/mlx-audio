# Irodori TTS

Flow Matching-based Japanese TTS model, ported to MLX.
Uses a Rectified Flow DiT over continuous DACVAE latents (48kHz).
Architecture and training follow [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/).

Original: [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)

## Models

### v3 (recommended)

| Model | HuggingFace | Conditioning |
|---|---|---|
| `mlx-community/Irodori-TTS-500M-v3-fp16` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-v3-fp16) | Voice cloning + automatic duration |
| `mlx-community/Irodori-TTS-500M-v3-8bit` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-v3-8bit) | Voice cloning + automatic duration |
| `mlx-community/Irodori-TTS-500M-v3-4bit` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-v3-4bit) | Voice cloning + automatic duration |

### v2

| Model | HuggingFace | Conditioning |
|---|---|---|
| `mlx-community/Irodori-TTS-500M-v2-fp16` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-v2-fp16) | Voice cloning (reference audio) |
| `mlx-community/Irodori-TTS-500M-v2-8bit` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-v2-8bit) | Voice cloning (reference audio) |
| `mlx-community/Irodori-TTS-500M-v2-4bit` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-v2-4bit) | Voice cloning (reference audio) |
| `mlx-community/Irodori-TTS-500M-v2-VoiceDesign-fp16` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-v2-VoiceDesign-fp16) | Voice design (text description) |
| `mlx-community/Irodori-TTS-500M-v2-VoiceDesign-8bit` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-v2-VoiceDesign-8bit) | Voice design (text description) |
| `mlx-community/Irodori-TTS-500M-v2-VoiceDesign-4bit` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-v2-VoiceDesign-4bit) | Voice design (text description) |

### v1

| Model | HuggingFace |
|---|---|
| `mlx-community/Irodori-TTS-500M-fp16` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-fp16) |

## Usage

### Voice cloning

```python
from mlx_audio.tts.generate import generate_audio

generate_audio(
    model="mlx-community/Irodori-TTS-500M-v3-fp16",
    text="今日はいい天気ですね。",
    ref_audio="speaker.wav",
    file_prefix="output",
)
```

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Irodori-TTS-500M-v3-fp16 \
  --text "今日はいい天気ですね。" \
  --ref_audio speaker.wav
```

### VoiceDesign

Describe the desired voice in text instead of providing reference audio:

```python
generate_audio(
    model="mlx-community/Irodori-TTS-500M-v2-VoiceDesign-fp16",
    text="今日はいい天気ですね。",
    instruct="落ち着いた、近い距離感の女性話者",
    file_prefix="output",
)
```

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Irodori-TTS-500M-v2-VoiceDesign-fp16 \
  --text "今日はいい天気ですね。" \
  --instruct "落ち着いた、近い距離感の女性話者"
```

## v3 Features

### Automatic Duration Prediction

v3 base models include an integrated duration predictor that automatically estimates
output length from the input text and reference audio. When `--seconds` is omitted,
the duration is predicted automatically:

```python
generate_audio(
    model="mlx-community/Irodori-TTS-500M-v3-fp16",
    text="今日はいい天気ですね。",
    ref_audio="speaker.wav",
    file_prefix="output",
    # seconds is auto-predicted; use duration_scale to adjust
    duration_scale=1.0,  # >1 longer, <1 shorter
)
```

### Sway Sampling

For faster inference, Sway Sampling can be combined with fewer Euler steps:

```python
generate_audio(
    model="mlx-community/Irodori-TTS-500M-v3-fp16",
    text="今日はいい天気ですね。",
    ref_audio="speaker.wav",
    file_prefix="output",
    num_steps=6,
    t_schedule_mode="sway",
    sway_coeff=-1.0,
)
```

## Memory requirements

The default `sequence_length=750` requires approximately 24GB of unified memory.
On 16GB machines, use reduced settings:

```python
generate_audio(
    model="mlx-community/Irodori-TTS-500M-v3-fp16",
    text="こんにちは。",
    ref_audio="speaker.wav",
    sequence_length=300,
    cfg_guidance_mode="alternating",
    file_prefix="output",
)
```

Approximate memory usage with `cfg_guidance_mode="alternating"`:

| sequence_length | Memory | Audio length |
|---|---|---|
| 100 | ~2GB | ~4s |
| 300 | ~2GB | ~12s |
| 400 | ~3GB | ~16s |

With `cfg_guidance_mode="independent"` (default), multiply memory by ~3.

## Notes

- v3 uses [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)
  and includes an integrated duration predictor for automatic output length estimation.
- v2 uses [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)
  and is bundled in the converted model weights.
- v1 uses `facebook/dacvae-watermarked`, downloaded automatically on first use.

## License

MIT License. See [Aratako/Irodori-TTS-500M-v3](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) for details.
