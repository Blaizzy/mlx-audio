# Qwen3-TTS

Alibaba's state-of-the-art multilingual TTS with three model variants.

## Voice Cloning

Clone any voice using a reference audio sample. Provide the wav file and its transcript:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
results = list(model.generate(
    text="Hello from Sesame.",
    ref_audio="sample_audio.wav",
    ref_text="This is what my voice sounds like.",
))

audio = results[0].audio  # mx.array
```

## CustomVoice (Emotion Control)

Use predefined voices with emotion/style instructions:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")
results = list(model.generate_custom_voice(
    text="I'm so excited to meet you!",
    speaker="Vivian",
    language="English",
    instruct="Very happy and excited.",
))

audio = results[0].audio  # mx.array
```

## VoiceDesign (Create Any Voice)

Create any voice from a text description:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
results = list(model.generate_voice_design(
    text="Big brother, you're back!",
    language="English",
    instruct="A cheerful young female voice with high pitch and energetic tone.",
))

audio = results[0].audio  # mx.array
```

## Available Models

| Model | Method | Description |
|-------|--------|-------------|
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | `generate()` | Fast, predefined voices |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | `generate()` | Higher quality |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16` | `generate_custom_voice()` | Voices + emotion |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | `generate_custom_voice()` | Better emotion control |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | `generate_voice_design()` | Create any voice |

## Speakers (Base/CustomVoice)

**Chinese:** `Vivian`, `Serena`, `Uncle_Fu`, `Dylan` (Beijing Dialect), `Eric` (Sichuan Dialect)

**English:** `Ryan`, `Aiden`
