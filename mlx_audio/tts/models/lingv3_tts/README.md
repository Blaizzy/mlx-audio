# Ming-3.0 TTS Tiny

Native MLX support for [mlx-community/Ming-3.0-tts-tiny-bf16](https://huggingface.co/mlx-community/Ming-3.0-tts-tiny-bf16), the 7.9B MoE language backbone with a flow-matching speech decoder. Output is 44.1 kHz mono audio. BF16, 8-bit and 4-bit MLX checkpoints use the same loader.

| Precision | Repository |
|-----------|------------|
| BF16 | [mlx-community/Ming-3.0-tts-tiny-bf16](https://huggingface.co/mlx-community/Ming-3.0-tts-tiny-bf16) |
| 8-bit | [mlx-community/Ming-3.0-tts-tiny-8bit](https://huggingface.co/mlx-community/Ming-3.0-tts-tiny-8bit) |
| 4-bit | [mlx-community/Ming-3.0-tts-tiny-4bit](https://huggingface.co/mlx-community/Ming-3.0-tts-tiny-4bit) |

Use any of these repositories with the Python and CLI examples below.

```python
from mlx_audio.tts import load
from mlx_audio.audio_io import write

model = load("mlx-community/Ming-3.0-tts-tiny-bf16", strict=True)
result = next(model.generate(
    "The evening breeze moved through the trees and turned the clouds gold.",
    voice="US_James", seed=42,
))
write("speech.wav", result.audio, result.sample_rate)
```

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Ming-3.0-tts-tiny-bf16 \
  --text "Hello from Ming three." --voice US_James
```

Other built-in voices include `灵小柔` and `灵楷`. The multilingual cloning task supports Chinese, English, German, Spanish, French, Korean and Japanese. The remaining task controls follow the upstream training contract:

| Task | Conditioning and controls |
|------|---------------------------|
| `ip` (default) | Built-in `voice`; empty speaker slot |
| `seed` / `clone` | `ref_audio`, `ref_text`, `lang_code="en"` or `"zh"` |
| `multilingual` | Reference audio/transcript and `lang_code="de"`, `"es"`, `"fr"`, `"ko"` or `"ja"` |
| `paralinguistic` | Reference audio/transcript, EN or ZH, inline vocal-event tags in text |
| `basic` | Reference speaker and `speed="慢速"/"中速"/"快速"`, `pitch="低"/"中"/"高"`, `volume="低"/"中"/"高"` |
| `emotion` | Reference speaker and `emotion="happy"`, `"sad"`, `"angry"`, `"neutral"`, `"surprised"`, `"fearful"` or `"disgusted"` |
| `dialect` | Reference speaker and `dialect="cantonese"` or `"sichuanese"` |
| `voice_design` | `style` description; empty speaker slot |
| `freeform` | Natural-language `instruct`; optional reference speaker |
| `podcast` | Ordered reference audio/transcript lists, text lines labeled `speaker_1:`, `speaker_2:`, etc. |

Reference-speaker extraction runs natively in MLX. All listed checkpoints include the CAMPPlus weights in `speaker_encoder/campplus.safetensors`, so no separate model downloads are required. Neither ONNX nor ONNX Runtime is required. A precomputed `spk_emb` with 192 values per speaker can replace speaker extraction; cloning and podcast continuation still require the reference waveform and transcript. Array references use the model's 44.1 kHz sample rate; file paths are resampled automatically.

```python
# Voice cloning, including acoustic continuation from the reference.
result = next(model.generate(
    "This sentence uses the reference speaker.",
    ref_audio="reference.wav", ref_text="Words spoken in the reference.",
    lang_code="en", seed=42,
))

# Freeform voice control.
result = next(model.generate(
    "Today feels like a wonderful day.",
    instruct="A cheerful young woman speaking clear English.", seed=42,
))

# Two-speaker continuation. Each speaker must appear in the target dialogue.
result = next(model.generate(
    " speaker_1:How are you?\n speaker_2:I am doing well.\n",
    task="podcast", ref_audio=["first.wav", "second.wav"],
    ref_text=["First reference transcript.", "Second reference transcript."],
    max_tokens=500,
))
```

Generation is offline and returns one audio result. The default limit is 250 acoustic steps (about 20 seconds); pass `max_tokens` or `max_decode_steps` for longer speech, within the 4096-position context budget. Text is passed through as supplied, so spell out numbers or abbreviations when a particular reading is required. Structured controls are mutually exclusive by task. Quantization covers the language backbone while preserving the acoustic modules and MoE routers. Runtime inference does not require PyTorch or FLA.
