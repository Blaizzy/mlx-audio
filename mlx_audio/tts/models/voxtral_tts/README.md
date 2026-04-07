# Voxtral TTS

Expressive multilingual text-to-speech with 20 voice presets across 9 languages. Based on [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603).

## Supported Models

- `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16`

## Usage

Python API:

```python
from mlx_audio.tts.utils import load
import sounddevice as sd

model = load("mlx-community/Voxtral-4B-TTS-2603-mlx-bf16")

for result in model.generate(text="Hello, how are you today?", voice="casual_male"):
    sd.play(result.audio, result.sample_rate)
    sd.wait()
```

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Voxtral-4B-TTS-2603-mlx-bf16 \
  --text "Hello, how are you today?" \
  --voice casual_male
```

## Voices

**English:** `casual_male`, `casual_female`, `cheerful_female`, `neutral_male`, `neutral_female`

**Multilingual:** `fr_male`, `fr_female`, `es_male`, `es_female`, `de_male`, `de_female`, `it_male`, `it_female`, `pt_male`, `pt_female`, `nl_male`, `nl_female`, `ar_male`, `hi_male`, `hi_female`

## Languages

English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi.

## License

Voxtral TTS weights are released under `CC-BY-NC`.
