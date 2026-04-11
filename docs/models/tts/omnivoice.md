# OmniVoice

OmniVoice is a zero-shot multilingual TTS model with voice cloning support built on a bidirectional Qwen backbone and a HiggsAudioV2 acoustic tokenizer.

## Highlights

- 646+ languages
- zero-shot voice cloning
- nonverbal tags like `[laughter]` and `[sigh]`
- English CMU pronunciation overrides
- Chinese pinyin pronunciation overrides

## Voice cloning

OmniVoice works best when both of the following are provided:

- `ref_audio`: a clean reference speech clip
- `ref_text`: the transcript of the reference clip

The MLX port now matches the original Python pipeline for reference preprocessing:

- torchaudio-compatible Hann-windowed sinc resampling
- RMS normalization for quiet references
- silence removal
- long-audio trimming at silence gaps

## Python example

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/OmniVoice-bf16")

results = list(model.generate(
    text="Hello from OmniVoice.",
    language="english",
    ref_audio="reference.wav",
    ref_text="This is what my voice sounds like.",
))

audio = results[0].audio
```

## Notes

- Reference text is strongly recommended for stable cloning.
- Best results come from a short clean reference clip after silence trimming.
- The MLX HiggsAudio encode path was debugged against the Python/CUDA reference implementation and brought to full parity for the tested cloning path.

## References

- Original repo: <https://github.com/k2-fsa/OmniVoice>
- Model source README: <https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/omnivoice>
