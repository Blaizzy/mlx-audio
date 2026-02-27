# Echo TTS

Diffusion-based text-to-speech with fast, high-fidelity voice cloning.

## Usage

Python API:

```python
from mlx_audio.tts import load

model = load("mlx-community/echo-tts-base")
result = next(model.generate("Hello from Echo TTS.", ref_audio="speaker.wav"))
audio = result.audio
```

CLI:

```bash
python -m mlx_audio.tts.generate --model mmlx-community/echo-tts-base --text "Hello from Echo TTS." --ref-audio speaker.wav
```

## License

Echo-TTS and Fish S1 weights are released under `CC-BY-NC-SA-4.0`.
Use is non-commercial unless you have separate permission from the model authors.
