# Usage

## Installation

```bash
pip install mlx-audio

# For web interface and API dependencies
pip install -r requirements.txt
```

## Command Line Example

```bash
# Basic usage
mlx_audio.tts.generate --text "Hello, world"

# Specify prefix for output file
mlx_audio.tts.generate --text "Hello, world" --file_prefix hello

# Adjust speaking speed
mlx_audio.tts.generate --text "Hello, world" --speed 1.4
```

## Python Example

```python
from mlx_audio.tts.generate import generate_audio

text = "The MLX King lives. Let him cook!"
generate_audio(text=text)
```

