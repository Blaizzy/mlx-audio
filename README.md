# MLX-Audio

A text-to-speech (TTS) and Speech-to-Speech (STS) library built on Apple's MLX framework, providing efficient speech synthesis on Apple Silicon.

## Features

- Fast inference on Apple Silicon (M series chips)
- Multiple language support
- Voice customization options
- Quantization support for optimized performance

## Installation

```bash
pip install mlx-audio
```

### Quick Start

To generate audio with an LLM use:

```bash
# Basic usage
mlx_audio.tts.generate --text "Hello, world"

# Specify prefix for output file
mlx_audio.tts.generate --text "Hello, world" --file_prefix hello

# Adjust speaking speed (0.5-2.0)
mlx_audio.tts.generate --text "Hello, world" --speed 1.4
```



## Models

### Kokoro

Kokoro is a multilingual TTS model that supports various languages and voice styles.

#### Example Usage

```python
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model
from IPython.display import Audio
import soundfile as sf

# Initialize the model
model_id = 'prince-canuma/Kokoro-82M'
model = load_model(model_id)

# Create a pipeline with American English
pipeline = KokoroPipeline(lang_code='a', model=model, repo_id=model_id)

# Generate audio
text = "The MLX King lives. Let him cook!"
for _, _, audio in pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+'):
    # Display audio in notebook (if applicable)
    display(Audio(data=audio, rate=24000, autoplay=0))

    # Save audio to file
    sf.write('audio.wav', audio[0], 24000)
```

#### Language Options

- 🇺🇸 `'a'` - American English
- 🇬🇧 `'b'` - British English
- 🇯🇵 `'j'` - Japanese (requires `pip install misaki[ja]`)
- 🇨🇳 `'z'` - Mandarin Chinese (requires `pip install misaki[zh]`)

## Advanced Features

### Quantization

You can quantize models for improved performance:

```python
from mlx_audio.tts.utils import quantize_model, load_model
import json
import mlx.core as mx

model = load_model(repo_id='prince-canuma/Kokoro-82M')
config = model.config

# Quantize to 8-bit
group_size = 64
bits = 8
weights, config = quantize_model(model, config, group_size, bits)

# Save quantized model
with open('./8bit/config.json', 'w') as f:
    json.dump(config, f)

mx.save_safetensors("./8bit/kokoro-v1_0.safetensors", weights, metadata={"format": "mlx"})
```

## Requirements

- MLX
- Python 3.8+
- Apple Silicon Mac (for optimal performance)

## License

[MIT License](LICENSE)

## Acknowledgements

- Thanks to the Apple MLX team for providing a great framework for building TTS and STS models.
- This project uses the Kokoro model architecture for text-to-speech synthesis.
