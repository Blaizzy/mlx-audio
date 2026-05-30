# Voice Cloning

Several MLX Audio models can clone a speaker's voice from a short reference audio sample. This guide covers the supported models and how to use them.

## Overview

| Model | Method | Reference Audio Required | Notes |
|-------|--------|--------------------------|-------|
| **CSM** | `--ref_audio` CLI / `ref_audio` kwarg | Yes (WAV) | Conversational Speech Model from Sesame |
| **Qwen3-TTS Base** | `ref_audio` + `ref_text` kwargs | Yes (WAV) + transcript | Alibaba multilingual TTS |
| **OmniVoice** | `ref_audio` + `ref_text` kwargs | Yes (WAV) + transcript recommended | 646+ language zero-shot cloning, best with prompt preprocessing |
| **Spark** | `ref_audio` kwarg | Yes | SparkTTS voice cloning |
| **Chatterbox** | `ref_audio` kwarg | Yes | Expressive multilingual TTS |
| **OuteTTS** | `ref_audio` kwarg | Yes | Efficient TTS with cloning |
| **Ming Omni TTS** | `ref_audio` kwarg | Yes | Multimodal with voice cloning |

## CSM (Conversational Speech Model)

CSM is the simplest path to voice cloning. Provide a WAV file of the target voice and CSM will match it:

### CLI

```bash
mlx_audio.tts.generate \
    --model mlx-community/csm-1b \
    --text "Hello from Sesame." \
    --ref_audio ./reference_voice.wav \
    --play
```

### Python

```python
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio

model = load_model("mlx-community/csm-1b")

for result in model.generate(
    text="Hello from Sesame.",
    ref_audio="./reference_voice.wav",
):
    audio = result.audio
```

!!! tip "Automatic transcription"
    If you do not provide `--ref_text`, MLX Audio will automatically transcribe the reference audio using a Whisper model. You can specify which STT model to use with `--stt_model`.

## Qwen3-TTS

Qwen3-TTS Base models support voice cloning by providing both a reference audio file and its transcript:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")

results = list(model.generate(
    text="Hello, welcome to MLX-Audio!",
    ref_audio="sample_audio.wav",
    ref_text="This is what my voice sounds like.",
))

audio = results[0].audio  # mx.array
```

### CustomVoice (Emotion Control)

The CustomVoice variant lets you combine a predefined voice with emotion and style instructions:

```python
model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")

results = list(model.generate_custom_voice(
    text="I'm so excited to meet you!",
    speaker="Vivian",
    language="English",
    instruct="Very happy and excited.",
))
```

### VoiceDesign (Create Any Voice)

The VoiceDesign variant creates a voice from a text description -- no reference audio needed:

```python
model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")

results = list(model.generate_voice_design(
    text="Big brother, you're back!",
    language="English",
    instruct="A cheerful young female voice with high pitch and energetic tone.",
))
```

### Available Qwen3-TTS Models

| Model | Method | Description |
|-------|--------|-------------|
| `Qwen3-TTS-12Hz-0.6B-Base-bf16` | `generate()` | Fast, predefined voices + cloning |
| `Qwen3-TTS-12Hz-1.7B-Base-bf16` | `generate()` | Higher quality |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16` | `generate_custom_voice()` | Voices + emotion |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | `generate_custom_voice()` | Better emotion control |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | `generate_voice_design()` | Create any voice from a description |

### Available Speakers (Base / CustomVoice)

- **Chinese:** `Vivian`, `Serena`, `Uncle_Fu`, `Dylan` (Beijing Dialect), `Eric` (Sichuan Dialect)
- **English:** `Ryan`, `Aiden`

## Ming Omni TTS

Ming Omni TTS supports voice cloning and style control:

```bash
mlx_audio.tts.generate \
    --model mlx-community/Ming-omni-tts-16.8B-A3B-bf16 \
    --text "This is a Ming Omni voice cloning test." \
    --ref_audio ./reference_voice.wav \
    --lang_code en \
    --verbose
```

See the [Ming Omni TTS model page](../models/tts/index.md) for detailed cookbook examples.

## Best Practices

### Preparing Reference Audio

!!! warning "Quality matters"
    The quality of your cloned voice depends heavily on the reference audio.

- **Duration** -- 5 to 15 seconds of clean speech works best. Very short clips lack enough speaker information; very long clips may confuse the model.
- **Format** -- Use WAV at 16 kHz or higher. The library will resample automatically, but starting with a good sample rate avoids artifacts.
- **Noise** -- Record in a quiet environment. Background noise will be cloned along with the voice. Consider using MossFormer2 or DeepFilterNet to enhance noisy recordings first.
- **Content** -- Natural, conversational speech produces the best results. Avoid whispering or shouting unless that is the target style.

### Providing Reference Text

Some models (Qwen3-TTS) require a transcript of the reference audio (`ref_text`). If you omit it, MLX Audio will transcribe the clip automatically using Whisper:

```bash
mlx_audio.tts.generate \
    --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 \
    --text "Cloned speech output." \
    --ref_audio reference.wav \
    --stt_model mlx-community/whisper-large-v3-turbo-asr-fp16
```

Providing the transcript yourself avoids loading the STT model and speeds up generation.

## OmniVoice

OmniVoice supports multilingual zero-shot voice cloning with a HiggsAudioV2 acoustic tokenizer and iterative masked generation.

!!! warning "ref_text must match preprocessed audio"
    OmniVoice preprocessing removes silence and trims the reference clip. If you transcribe the **original** file, the ASR transcript may be longer than the preprocessed audio, causing the extra text to leak into generation. Always transcribe the **preprocessed** audio, not the raw recording. See `examples/omnivoice_clone_demo.py` for the correct workflow.

```python
from mlx_audio.tts.utils import load_model as load_tts
from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt
from mlx_audio.stt.utils import load_model as load_stt
from mlx_audio.audio_io import write as audio_write
import mlx.core as mx, numpy as np, tempfile, os

tts = load_tts("mlx-community/OmniVoice-bf16")

# Preprocess → encode → decode → transcribe (matches original pipeline)
ref_tokens = create_voice_clone_prompt("reference.wav", tokenizer=tts.audio_tokenizer)
mx.eval(ref_tokens)

preprocessed = np.array(tts.audio_tokenizer.decode(ref_tokens).astype(mx.float32))
tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
audio_write(tmp.name, preprocessed, 24000); tmp.close()

ref_text = load_stt("mlx-community/Qwen3-ASR-0.6B-8bit").generate(tmp.name).text
os.unlink(tmp.name)

results = list(tts.generate(
    text="Hello from OmniVoice.",
    language="english",
    ref_tokens=ref_tokens,
    ref_text=ref_text,
))

audio_write("output.wav", np.array(results[0].audio), results[0].sample_rate)
```

### OmniVoice-specific notes

- **Reference text is required for stable cloning.** Without it, output quality degrades significantly — garbled speech, wrong language, or missing words.
- **Transcribe after preprocessing, not before.** The original k2-fsa/OmniVoice runs Whisper on audio after silence removal. This demo replicates that approach.
- **Prompt preprocessing matters.** MLX Audio mirrors the original Python pipeline with RMS normalization, silence removal, trimming at silence gaps, and torchaudio-compatible resampling before reference encoding.
- **Best reference length:** roughly 5–15 seconds of actual speech after silence trimming.
- **Supported inline controls:** nonverbal tags such as `[laughter]`, `[sigh]`, and pronunciation overrides for English CMU dictionary forms and Chinese pinyin forms.

### Example: English CMU pronunciation control

```python
results = list(model.generate(
    text="He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.",
    language="english",
))
```

### Example: Nonverbal tags

```python
results = list(model.generate(
    text="I just heard the funniest joke [laughter] that was incredible.",
    language="english",
))
```

### Combining Cloning with Streaming

Voice cloning and streaming work together. Add `--stream` to any cloning command:

```bash
mlx_audio.tts.generate \
    --model mlx-community/csm-1b \
    --text "Streaming with a cloned voice." \
    --ref_audio reference.wav \
    --stream
```
