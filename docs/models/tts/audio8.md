# Audio8 TTS

[Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b)
generates mono 44.1 kHz speech in 23 languages on Apple Silicon, with optional
zero-shot voice cloning.

Ready-to-use MLX checkpoints include the codec and require no PyTorch:

| Precision | Repository |
|-----------|------------|
| BF16 | [nativ-community/Audio8-TTS-Realistic-0.6b-bf16](https://huggingface.co/nativ-community/Audio8-TTS-Realistic-0.6b-bf16) |
| 4-bit | [nativ-community/Audio8-TTS-Realistic-0.6b-4bit](https://huggingface.co/nativ-community/Audio8-TTS-Realistic-0.6b-4bit) |
| 8-bit | [nativ-community/Audio8-TTS-Realistic-0.6b-8bit](https://huggingface.co/nativ-community/Audio8-TTS-Realistic-0.6b-8bit) |

Use any of these repository IDs in the examples below.

## Generate speech

```python
from mlx_audio.audio_io import write
from mlx_audio.tts.utils import load

model = load("nativ-community/Audio8-TTS-Realistic-0.6b-bf16")
result = next(model.generate(
    text="Welcome to Audio8, running on Apple Silicon.",
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
))
write("speech.wav", result.audio, result.sample_rate)
```

```bash
python -m mlx_audio.tts.generate \
  --model nativ-community/Audio8-TTS-Realistic-0.6b-bf16 \
  --text "Welcome to Audio8, running on Apple Silicon." \
  --temperature 0.8 --top_p 0.95 --max_tokens 1024
```

## Clone a voice

```python
result = next(model.generate(
    text="This sentence uses the reference speaker.",
    ref_audio="reference.wav",
    ref_text="The exact words spoken in the reference recording.",
))
write("cloned.wav", result.audio, result.sample_rate)
```

For the CLI, add `--ref_audio reference.wav --ref_text "The exact reference transcript."`.
Reference paths are resampled automatically; Python audio arrays must already
be mono at 44.1 kHz. Language follows the input text. Realistic supports Arabic,
Cantonese, Chinese, Dutch, English, Finnish, French, German, Hindi, Indonesian,
Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish,
Swedish, Thai, Turkish, and Vietnamese. Quality varies by language.

Generation produces one complete waveform per call. Text tokens, reference
audio frames, and generated frames share a 2,048-position context budget.
`max_tokens` limits audio frames, each approximately 46 milliseconds long.
Long input text or reference recordings leave less room for output; split long
passages into shorter utterances. See the [model README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/arktts/README.md)
for generation options and the supported Preview models.

Realistic-0.6b is released under Apache-2.0.
