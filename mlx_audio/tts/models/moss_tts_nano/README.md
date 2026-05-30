# MOSS-TTS-Nano

MOSS-TTS-Nano is a compact voice-cloning TTS model from OpenMOSS.

## Supported Models

- `mlx-community/MOSS-TTS-Nano-100M`

## Usage

Python API:

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load

model = load("mlx-community/MOSS-TTS-Nano-100M")

result = next(model.generate(
    text="Hello, this is a cloned voice.",
    ref_audio="speaker.wav",
))
audio_write("output.wav", result.audio, result.sample_rate)
```

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/MOSS-TTS-Nano-100M \
  --text "Hello, this is a cloned voice." \
  --ref_audio speaker.wav
```
