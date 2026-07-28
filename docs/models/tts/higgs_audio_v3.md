# Higgs Audio v3 TTS

Higgs Audio v3 TTS is a Qwen3-backed conversational TTS model with fused
multi-codebook audio token generation, inline control tokens, multilingual
speech, and zero-shot voice cloning.

```bash
python -m mlx_audio.tts.generate \
  --model bosonai/higgs-audio-v3-tts-4b \
  --text "Hello from Higgs Audio v3 on MLX."
```

## Voice cloning

Pass one or more reference clips with matching transcripts:

```bash
python -m mlx_audio.tts.generate \
  --model bosonai/higgs-audio-v3-tts-4b \
  --text "Have a nice day and enjoy the sunshine." \
  --ref_audio reference.wav \
  --ref_text "Reference transcript."
```

Multiple references use repeated CLI flags:

```bash
python -m mlx_audio.tts.generate \
  --model bosonai/higgs-audio-v3-tts-4b \
  --text "Let's keep the same voice across this line." \
  --ref_audio speaker_1.wav \
  --ref_text "First reference transcript." \
  --ref_audio speaker_2.wav \
  --ref_text "Second reference transcript."
```

## Python

```python
from mlx_audio.tts.utils import load
from mlx_audio.audio_io import write as audio_write

model = load("bosonai/higgs-audio-v3-tts-4b")

for result in model.generate(
    text="Hello from Higgs Audio v3 on MLX.",
    ref_audio="reference.wav",
    ref_text="Reference transcript.",
    temperature=1.0,
    max_new_tokens=2048,
):
    audio_write("output.wav", result.audio, result.sample_rate)
```

If you reuse the same reference voice across multiple generations, encode it
once and pass the pre-encoded reference codes:

```python
reference_codes = model.encode_reference_audio("reference.wav")

for result in model.generate(
    text="This skips reference audio encoding.",
    ref_audio_codes=reference_codes,
    ref_text="Reference transcript.",
    temperature=1.0,
):
    audio_write("output.wav", result.audio, result.sample_rate)
```

Batch generation can reuse the same pre-encoded reference across multiple
texts:

```python
reference_codes = model.encode_reference_audio("reference.wav")
texts = [
    "The first line uses the cloned voice.",
    "The second line is generated in the same batch.",
]

for result in model.batch_generate(
    texts=texts,
    ref_audio_codes=reference_codes,
    ref_text="Reference transcript.",
    temperature=1.0,
):
    audio_write(f"output_{result.sequence_idx}.wav", result.audio, result.sample_rate)
```

## Controls

Inline control tokens from the upstream model can be placed directly in the
input text, for example:

```text
<|emotion:amusement|><|prosody:expressive_high|>That was unexpected. <|sfx:laughter|>Hehe.
```

For sound-effect tags, follow the upstream guidance and include matching written
onomatopoeia after the tag.

## Notes

- The model is released under the Boson Higgs Audio v3 Research and
  Non-Commercial License. See the original model card and license:
  <https://huggingface.co/bosonai/higgs-audio-v3-tts-4b>
