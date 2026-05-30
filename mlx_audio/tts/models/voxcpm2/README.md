# VoxCPM2

2B-parameter multilingual tokenizer-free TTS model with 48kHz studio-quality output. Supports zero-shot generation, voice design (create voices from text descriptions), voice cloning, and continuation for long-form speech. 30 languages including English, Chinese, Indonesian, Japanese, Korean, and more.

**Original model:** [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)

## Usage

Python API:

```python
from mlx_audio.tts.utils import load

model = load("mlx-community/VoxCPM2-8bit")

result = next(model.generate("Hello, this is VoxCPM2 on Apple Silicon."))
audio = result.audio  # mlx array, 48kHz
```

## Voice Design

Create voices from text descriptions without reference audio:

```python
result = next(model.generate(
    text="Hello, welcome to VoxCPM2.",
    instruct="A young woman, warm and gentle voice",
))
```

## Voice Cloning

Clone any voice from an audio sample:

```python
result = next(model.generate(
    text="This text will be spoken in the reference voice.",
    ref_audio="speaker.wav",
))
```

## Continuation

Continue generating speech from a previous audio clip (for long-form content like audiobooks):

```python
result = next(model.generate(
    text=" and this continues seamlessly from the previous sentence.",
    prompt_text="The previous sentence spoken aloud",
    prompt_audio="previous.wav",
))
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference_timesteps` | 10 | CFM diffusion steps. Lower = faster, higher = better quality |
| `cfg_value` | 2.0 | Classifier-free guidance strength |
| `instruct` | `None` | Voice description for voice design mode |
| `ref_audio` | `None` | Reference audio path or array for voice cloning |
| `prompt_text` | `None` | Text of the prompt audio for continuation mode |
| `prompt_audio` | `None` | Prompt audio path or array for continuation mode |
| `warmup_patches` | 0 | Extra patches to generate before output (for onset stability) |
| `max_tokens` | 2000 | Maximum number of audio patches to generate |

## CLI

```bash
# Zero-shot
python -m mlx_audio.tts.generate \
  --model mlx-community/VoxCPM2-8bit \
  --text "Hello world" \
  --play

# Voice design
python -m mlx_audio.tts.generate \
  --model mlx-community/VoxCPM2-8bit \
  --text "Hello world" \
  --instruct "A young woman, gentle voice" \
  --play

# Voice cloning
python -m mlx_audio.tts.generate \
  --model mlx-community/VoxCPM2-8bit \
  --text "Hello world" \
  --ref_audio speaker.wav \
  --ref_text "placeholder" \
  --play
```

## Available Models

| Model | Parameters | Format | Size |
|-------|-----------|--------|------|
| `mlx-community/VoxCPM2-bf16` | 2B | bf16 | 4.96 GB |
| `mlx-community/VoxCPM2-8bit` | 2B | 8-bit | 3.23 GB |
| `mlx-community/VoxCPM2-4bit` | 2B | 4-bit | 2.30 GB |

## Architecture

- **MiniCPM4 backbone:** 2048 hidden, 28 layers, GQA with kv_channels=128
- **Residual LM:** 8 layers, no RoPE
- **VoxCPMLocDiTV2:** 1024 hidden, 12 layers, multi-token mu with CFM sampling
- **AudioVAE V2:** asymmetric 16kHz encode / 48kHz decode with sample-rate conditioning
- **Scalar quantization** for audio token compression

## License

VoxCPM2 is released under the [Apache License 2.0](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE).
