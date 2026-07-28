# Higgs Audio v3 TTS

Higgs Audio v3 TTS is a Qwen3-backed conversational TTS model with fused
multi-codebook audio token generation, inline control tokens, multilingual
speech, and zero-shot voice cloning.

## Supported Models

- `bosonai/higgs-audio-v3-tts-4b`

## Usage

Python API:

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load

model = load("bosonai/higgs-audio-v3-tts-4b")

result = next(model.generate(
    text="Hello from Higgs Audio v3 on MLX.",
    temperature=1.0,
    max_new_tokens=2048,
))
audio_write("output.wav", result.audio, result.sample_rate)
```

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model bosonai/higgs-audio-v3-tts-4b \
  --text "Hello from Higgs Audio v3 on MLX." \
  --output_path outputs
```

## Voice Cloning

Pass a reference clip and matching transcript:

```python
result = next(model.generate(
    text="Have a nice day and enjoy the sunshine.",
    ref_audio="reference.wav",
    ref_text="Reference transcript.",
    temperature=1.0,
    max_new_tokens=2048,
))
audio_write("cloned.wav", result.audio, result.sample_rate)
```

For repeated use of the same reference, encode it once and pass the returned
codes to generation:

```python
reference_codes = model.encode_reference_audio("reference.wav")

result = next(model.generate(
    text="This skips reference audio encoding.",
    ref_audio_codes=reference_codes,
    ref_text="Reference transcript.",
    temperature=1.0,
    max_new_tokens=2048,
))
audio_write("cloned.wav", result.audio, result.sample_rate)
```

Batch generation can reuse the same pre-encoded reference across multiple texts:

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
    max_new_tokens=2048,
):
    audio_write(f"cloned_{result.sequence_idx}.wav", result.audio, result.sample_rate)
```

Reference audio and text can be repeated:

```bash
python -m mlx_audio.tts.generate \
  --model bosonai/higgs-audio-v3-tts-4b \
  --text "Let's keep the same voice across this line." \
  --ref_audio speaker_1.wav \
  --ref_text "First reference transcript." \
  --ref_audio speaker_2.wav \
  --ref_text "Second reference transcript." \
  --output_path outputs
```

## Inline Controls

Control tokens from the upstream model can be placed directly in the input text:

```text
<|emotion:amusement|><|prosody:expressive_high|>That was unexpected. <|sfx:laughter|>Hehe.
```

For sound-effect tags, include matching written onomatopoeia after the tag.

## License

The upstream model is released under the Boson Higgs Audio v3 Research and
Non-Commercial License. See the original model card and license:
<https://huggingface.co/bosonai/higgs-audio-v3-tts-4b>
