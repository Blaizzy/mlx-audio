# OmniVoice

OmniVoice is a zero-shot multilingual TTS model with voice cloning support built on a bidirectional Qwen backbone and a HiggsAudioV2 acoustic tokenizer.

## Highlights

- 646+ languages
- zero-shot voice cloning
- batch generation via `generate_batch()`
- nonverbal tags like `[laughter]` and `[sigh]`
- English CMU pronunciation overrides
- Chinese pinyin pronunciation overrides

## Voice cloning

OmniVoice cloning requires two inputs:

- `ref_audio`: a clean reference speech clip (5–15 seconds recommended)
- `ref_text`: the transcript of the reference clip

### Why ref_text matters

Without `ref_text`, the model cannot properly align the reference voice with the target text. This causes artifacts such as garbled output or wrong language at the beginning of the generated audio.

The original Python OmniVoice solves this by auto-transcribing the reference audio with a built-in Whisper model. The MLX port intentionally does **not** bundle an ASR model to avoid coupling. Instead, you can use any `mlx-audio` STT model to obtain the transcript before calling `generate()`.

### Obtaining ref_text with Qwen3 ASR

The transcript must match the **preprocessed** reference audio (after silence removal), not the original recording. If you transcribe the raw file, the ASR may return more text than the preprocessed clip contains, and the excess will leak into the generated speech.

The correct workflow mirrors the original k2-fsa/OmniVoice: preprocess first, then transcribe.

```python
from mlx_audio.stt.utils import load_model as load_stt
from mlx_audio.tts.utils import load_model as load_tts
from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt
from mlx_audio.audio_io import write as audio_write
import mlx.core as mx
import numpy as np
import tempfile, os

tts = load_tts("mlx-community/OmniVoice-bf16")
tokenizer = tts.audio_tokenizer

# Step 1: Preprocess reference audio (silence removal, RMS norm)
ref_tokens = create_voice_clone_prompt("reference.wav", tokenizer=tokenizer)
mx.eval(ref_tokens)

# Step 2: Decode preprocessed tokens back to audio, then transcribe
preprocessed = np.array(tokenizer.decode(ref_tokens).astype(mx.float32))
tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
audio_write(tmp.name, preprocessed, 24000)
tmp.close()

stt = load_stt("mlx-community/Qwen3-ASR-0.6B-8bit")
ref_text = stt.generate(tmp.name).text
os.unlink(tmp.name)

# Step 3: Generate with ref_tokens + ref_text from the same source
results = list(tts.generate(
    text="Hello from OmniVoice.",
    language="english",
    ref_tokens=ref_tokens,
    ref_text=ref_text,
))

audio_write("output.wav", np.array(results[0].audio), results[0].sample_rate)
```

Any STT model works — Whisper, Qwen3-ASR, SenseVoice, etc. The key is to transcribe the **preprocessed** audio, not the original file.

See `examples/omnivoice_clone_demo.py` for a complete CLI script with this workflow.

### Reference preprocessing

The MLX port matches the original Python pipeline for reference audio preprocessing:

- torchaudio-compatible Hann-windowed sinc resampling
- RMS normalization for quiet references
- fixed-threshold silence removal with built-in preprocessing
- long-audio trimming at silence gaps

Best results come from 5–15 seconds of clean speech after silence trimming.

## Batch generation

Generate multiple utterances in one call:

```python
from mlx_audio.audio_io import write as audio_write
import numpy as np

model = load_tts("mlx-community/OmniVoice-bf16")

results = model.generate_batch(
    text=["Hello world.", "Bonjour le monde."],
    language=["english", "french"],
    num_steps=32,
)

for i, r in enumerate(results):
    audio_write(f"output_{i}.wav", np.array(r.audio), r.sample_rate)
```

Batch generation supports per-item language, ref_audio, ref_text, and duration. Default `max_batch_size=8` with automatic chunking for larger inputs.

## Inline pronunciation control

### English: CMU dictionary

```python
results = list(model.generate(
    text="He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.",
    language="english",
))
```

### Chinese: pinyin with tones

```python
results = list(model.generate(
    text="今天天气很好，我想去打ZHE2出售后的商店买东西。",
    language="chinese",
))
```

### Nonverbal tags

Supported tags: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]`.

```python
results = list(model.generate(
    text="I just heard the funniest joke [laughter] that was incredible.",
    language="english",
))
```

## Notes

- `ref_text` is strongly recommended for stable voice cloning. Without it, output quality degrades significantly.
- The MLX HiggsAudio encode path achieves full token parity with the Python/CUDA reference implementation.
- The encode pipeline uses a torchaudio-compatible sinc resampler (not scipy) to match upstream precision exactly.

## References

- Original repo: <https://github.com/k2-fsa/OmniVoice>
- Paper: <https://arxiv.org/abs/2604.00688>
- HF model: <https://huggingface.co/mlx-community/OmniVoice-bf16>
