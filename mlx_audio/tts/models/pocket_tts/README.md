# Phonon

Phonon generates 24 kHz speech on Apple Silicon and supports streaming audio generation.

## Supported Models

| Model | Parameters | Languages |
|---|---|---|
| [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en) | 31M | English |
| [gradium/phonon-40m-en](https://huggingface.co/gradium/phonon-40m-en) | 40M | English |
| [gradium/phonon-40m-fr](https://huggingface.co/gradium/phonon-40m-fr) | 40M | French |
| [gradium/phonon-40m-de](https://huggingface.co/gradium/phonon-40m-de) | 40M | German |
| [gradium/phonon-40m-es](https://huggingface.co/gradium/phonon-40m-es) | 40M | Spanish |
| [gradium/phonon-40m-pt](https://huggingface.co/gradium/phonon-40m-pt) | 40M | Portuguese |
| [gradium/phonon-40m-multilingual](https://huggingface.co/gradium/phonon-40m-multilingual) | 40M | English, French, German, Spanish, Portuguese |

For the 40M models, accept their access terms on Hugging Face and sign in before loading.

## Usage

Python API:

```python
import numpy as np

from mlx_audio.audio_io import write
from mlx_audio.tts.utils import load

model = load("gradium/phonon-31m-en")

result = next(model.generate(
    text="Hello, this is Phonon running on Apple Silicon.",
))
write("phonon.wav", np.array(result.audio), result.sample_rate)
```

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model gradium/phonon-31m-en \
  --text "Hello, this is Phonon running on Apple Silicon." \
  --play
```

Add `--stream` to the CLI command for streaming playback, or pass `stream=True`
to `model.generate()` to receive audio chunks in Python.

## Voices

The official Gradium repositories include a default voice, selected when `voice`
is omitted or set to `"default"`. To use another voice embedding, pass its local
`.safetensors` path or an `hf://` URL as `voice`.
