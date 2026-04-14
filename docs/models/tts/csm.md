# CSM (Conversational Speech Model)

CSM is Sesame's 1B parameter conversational speech model with voice cloning support. It generates natural-sounding speech and supports multi-turn conversational context, making it well-suited for dialogue applications.

## Model Variants

| Model | Format | HuggingFace |
|-------|--------|-------------|
| `mlx-community/csm-1b` | -- | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/csm-1b) |

## Usage

### Basic Generation

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/csm-1b \
        --text "Hello from Sesame." \
        --voice conversational_a
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/csm-1b")

    for result in model.generate(
        text="Hello from Sesame.",
        voice="conversational_a",
    ):
        audio = result.audio  # mx.array waveform
    ```

### Voice Cloning

Clone any voice using a reference audio sample and its transcript:

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/csm-1b \
        --text "Hello from Sesame." \
        --ref_audio ./reference_voice.wav \
        --ref_text "This is what my voice sounds like." \
        --play
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/csm-1b")

    for result in model.generate(
        text="Hello from Sesame.",
        ref_audio="reference_voice.wav",
        ref_text="This is what my voice sounds like.",
    ):
        audio = result.audio
    ```

### Streaming

CSM supports streaming for low-latency audio output:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/csm-1b")

for result in model.generate(
    text="This is a streaming example.",
    voice="conversational_a",
    stream=True,
    streaming_interval=0.5,
):
    # Process each audio chunk as it arrives
    audio_chunk = result.audio
```

### Multi-Turn Context

CSM can take conversational context (previous turns) to maintain speaker consistency:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/csm-1b")

# The model accepts a list of Segment objects as context
# Each segment has a speaker ID, text, and audio
for result in model.generate(
    text="That sounds great, let's do it!",
    speaker=0,
    voice="conversational_a",
):
    audio = result.audio
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `voice` | `conversational_a` | Default voice preset (used when no `ref_audio` is provided) |
| `speaker` | `0` | Speaker ID for multi-speaker context |
| `max_audio_length_ms` | `90000` | Maximum audio length in milliseconds |
| `stream` | `False` | Enable streaming output |
| `streaming_interval` | `0.5` | Interval (seconds) between streamed chunks |
| `voice_match` | `True` | Enable voice matching |

## Links

- [:octicons-mark-github-16: Source code](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/sesame)
- [:octicons-link-external-16: mlx-community/csm-1b](https://huggingface.co/mlx-community/csm-1b)
