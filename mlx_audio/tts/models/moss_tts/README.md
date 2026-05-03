# MOSS-TTS

MOSS-TTS covers the 8B `MossTTSDelay` model, the `MOSS-TTSD-v1.0` dialogue model, and the 1.7B local-transformer variant from OpenMOSS. It uses a Qwen3 backbone with the matching multi-codebook audio generation path and the full MOSS Audio Tokenizer.

## Supported Models

- `OpenMOSS-Team/MOSS-TTS`
- `OpenMOSS-Team/MOSS-TTSD-v1.0`
- `OpenMOSS-Team/MOSS-TTS-Local-Transformer`

## Usage

Python API:

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load

model = load("OpenMOSS-Team/MOSS-TTS", lazy=True)

result = next(model.generate(
    text="Hello, this is MOSS-TTS running on MLX.",
    max_tokens=120,
))
audio_write("output.wav", result.audio, result.sample_rate)
```

Voice cloning:

```python
result = next(model.generate(
    text="This is a short cloned voice sample.",
    ref_audio="speaker.wav",
    max_tokens=120,
))
audio_write("cloned.wav", result.audio, result.sample_rate)
```

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS \
  --text "Hello, this is MOSS-TTS running on MLX." \
  --output_path outputs
```

Dialogue generation with `MOSS-TTSD-v1.0`:

```bash
python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --text "[S1] Hello. [S2] Hi, this is MOSS-TTSD running on MLX." \
  --output_path outputs
```

Reference audio and text can be repeated for multiple speakers:

```bash
python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --text "[S1] This uses the first reference. [S2] This uses the second." \
  --ref_audio speaker_1.wav \
  --ref_text "Reference transcript for speaker one." \
  --ref_audio speaker_2.wav \
  --ref_text "Reference transcript for speaker two." \
  --output_path outputs
```
