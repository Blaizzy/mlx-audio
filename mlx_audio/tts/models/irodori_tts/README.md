# Irodori TTS

Flow Matching-based Japanese TTS model, ported to MLX.
Uses a Rectified Flow DiT over continuous DACVAE latents (48kHz).
Architecture and training follow [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/).

Original: [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)

## Models

### v4.1 (recommended)

v4.1-Small is a single unified model: voice cloning, VoiceDesign and automatic
duration prediction in one checkpoint.

| Model | HuggingFace | Conditioning |
|---|---|---|
| `mlx-community/Irodori-TTS-v4.1-Small-fp16` | [link](https://huggingface.co/mlx-community/Irodori-TTS-v4.1-Small-fp16) | Voice cloning + VoiceDesign + automatic duration |
| `mlx-community/Irodori-TTS-v4.1-Small-8bit` | [link](https://huggingface.co/mlx-community/Irodori-TTS-v4.1-Small-8bit) | Voice cloning + VoiceDesign + automatic duration |

### v4

| Model | HuggingFace | Conditioning |
|---|---|---|
| `mlx-community/Irodori-TTS-v4-Small-fp16` | [link](https://huggingface.co/mlx-community/Irodori-TTS-v4-Small-fp16) | Voice cloning + VoiceDesign + automatic duration |
| `mlx-community/Irodori-TTS-v4-Small-8bit` | [link](https://huggingface.co/mlx-community/Irodori-TTS-v4-Small-8bit) | Voice cloning + VoiceDesign + automatic duration |

### v3

| Model | HuggingFace | Conditioning |
|---|---|---|
| `mlx-community/Irodori-TTS-500M-v3-fp16` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-v3-fp16) | Voice cloning + automatic duration |
| `mlx-community/Irodori-TTS-500M-v3-8bit` | [link](https://huggingface.co/mlx-community/Irodori-TTS-500M-v3-8bit) | Voice cloning + automatic duration |
| `mlx-community/Irodori-TTS-600M-v3-VoiceDesign-fp16` | [link](https://huggingface.co/mlx-community/Irodori-TTS-600M-v3-VoiceDesign-fp16) | Voice cloning + VoiceDesign (dual conditioning) |
| `mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit` | [link](https://huggingface.co/mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit) | Voice cloning + VoiceDesign (dual conditioning) |

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
    model="mlx-community/Irodori-TTS-v4.1-Small-fp16",
    text="今日はいい天気ですね。",
    ref_audio="speaker.wav",
    file_prefix="output",
)
```

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Irodori-TTS-v4.1-Small-fp16 \
  --text "今日はいい天気ですね。" \
  --ref_audio speaker.wav
```

### VoiceDesign

#### v4 / v4.1 VoiceDesign

Caption only:

```python
generate_audio(
    model="mlx-community/Irodori-TTS-v4.1-Small-fp16",
    text="今日はいい天気ですね。",
    instruct="落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。",
    file_prefix="output",
)
```

Style-controlled voice cloning with reference speech + caption:

```python
generate_audio(
    model="mlx-community/Irodori-TTS-v4.1-Small-fp16",
    text="今日はいい天気ですね。",
    ref_audio="speaker.wav",
    instruct="深く傷つき、今にも泣き出しそうな様子。声が震えており、悲痛なトーンで弱々しく話す。",
    file_prefix="output",
)
```

#### v3 VoiceDesign

Caption only:

```python
generate_audio(
    model="mlx-community/Irodori-TTS-600M-v3-VoiceDesign-fp16",
    text="今日はいい天気ですね。",
    instruct="落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。",
    file_prefix="output",
)
```

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Irodori-TTS-600M-v3-VoiceDesign-fp16 \
  --text "今日はいい天気ですね。" \
  --instruct "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。"
```

Style-controlled voice cloning with reference speech + caption:

```python
generate_audio(
    model="mlx-community/Irodori-TTS-600M-v3-VoiceDesign-fp16",
    text="今日はいい天気ですね。",
    ref_audio="speaker.wav",
    instruct="深く傷つき、今にも泣き出しそうな様子。声が震えており、悲痛なトーンで弱々しく話す。",
    file_prefix="output",
)
```

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Irodori-TTS-600M-v3-VoiceDesign-fp16 \
  --text "今日はいい天気ですね。" \
  --ref_audio speaker.wav \
  --instruct "深く傷つき、今にも泣き出しそうな様子。声が震えており、悲痛なトーンで弱々しく話す。"
```

#### v2 VoiceDesign

Caption only (reference audio is not supported):

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

## v4 / v4.1 Features

### Shared pretrained text encoder

v4 replaces the two scratch-trained text/caption encoders with a single
pretrained [ModernBERT-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m)
backbone feeding separate projectors. The backbone weights and its tokenizer are
bundled in the converted model, so no extra download happens at inference time.

### Multi-clip reference audio (up to 120s)

v4 was trained with up to 120 seconds of reference audio. Passing a list encodes
each clip separately and concatenates them, which matches training better than
one long uninterrupted recording:

```python
generate_audio(
    model="mlx-community/Irodori-TTS-v4.1-Small-fp16",
    text="今日はいい天気ですね。",
    ref_audio=["speaker_1.wav", "speaker_2.wav", "speaker_3.wav"],
    file_prefix="output",
)
```

`max_ref_seconds` overrides the checkpoint's 120s budget; the reference is
trimmed to it after concatenation.

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

- v4 and v4.1 use [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)
  and bundle a ModernBERT-ja-310m text encoder, so their weights are roughly
  1 GB larger than v3 at the same precision.
- v3 uses [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)
  and includes an integrated duration predictor for automatic output length estimation.
- v2 uses [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)
  and is bundled in the converted model weights.
- v1 uses `facebook/dacvae-watermarked`, downloaded automatically on first use.

## License

MIT License. See [Aratako/Irodori-TTS-500M-v3](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) for details.
