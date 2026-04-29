# Moshi

MLX implementation of [Moshi](https://github.com/kyutai-labs/moshi) from Kyutai Labs. Moshi is a full-duplex speech-to-speech foundation model that can listen and talk at the same time in real-time.

```python
from mlx_audio.sts import load

model = load("kyutai/moshiko-mlx-q4")
```

## Features

- Full-duplex: Handles concurrent input and output audio streams
- Multi-modal: Simultaneously generates text ("inner monologue") and audio tokens
- Ultra-low latency: ~200ms theoretical latency
- Streaming generation natively via MLX

## Usage

```python
from mlx_audio.sts.models.moshi import MoshiSTSModel
import sounddevice as sd
import numpy as np

# Load quantized model directly from HuggingFace
model = MoshiSTSModel.from_pretrained("kyutai/moshiko-mlx-q4", quantized=4)

# Open audio output stream
stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.float32)
stream.start()

print("Generating...")
# Generate 150 steps (~12 seconds of audio)
for word, pcm_frame in model.generate(steps=150):
    if word:
        # Print inner monologue as it generates
        print(word, end="", flush=True)
        
    if pcm_frame is not None:
        # Stream audio directly to speakers
        stream.write(np.array(pcm_frame))

stream.stop()
stream.close()
```

## Available Models

- `kyutai/moshiko-mlx-bf16`: Full precision (requires ~16GB unified memory)
- `kyutai/moshiko-mlx-q8`: 8-bit quantization
- `kyutai/moshiko-mlx-q4`: 4-bit quantization (fastest, fits in ~8GB unified memory)
