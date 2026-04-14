# Dia

Dia is a 1.6B parameter dialogue-focused TTS model. It natively supports multi-speaker conversations using `[S1]` and `[S2]` speaker tags, making it ideal for generating realistic dialogue audio.

## Model Variants

| Model | Format | HuggingFace |
|-------|--------|-------------|
| `mlx-community/Dia-1.6B-fp16` | float16 | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/Dia-1.6B-fp16) |

## Usage

### Basic Dialogue Generation

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/Dia-1.6B-fp16 \
        --text "[S1] Hey, have you tried MLX-Audio? [S2] Yes, it runs great on my Mac!"
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/Dia-1.6B-fp16")

    for result in model.generate(
        text="[S1] Hey, have you tried MLX-Audio? [S2] Yes, it runs great on my Mac!",
    ):
        audio = result.audio  # mx.array waveform
    ```

### Multi-Turn Dialogue

Dia automatically splits text on `[S1]`/`[S2]` tags and generates each turn separately:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Dia-1.6B-fp16")

dialogue = """[S1] Welcome to the show! Today we're talking about AI on Apple Silicon.
[S2] Thanks for having me. It's an exciting time for on-device inference.
[S1] Absolutely. What's been the biggest breakthrough?
[S2] I'd say the combination of unified memory and optimized frameworks like MLX."""

for result in model.generate(text=dialogue):
    audio = result.audio
```

### With Reference Audio

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Dia-1.6B-fp16")

for result in model.generate(
    text="[S1] Hello, this is a voice cloning test.",
    ref_audio="reference.wav",
    ref_text="This is a sample of my voice.",
):
    audio = result.audio
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | `1.3` | Sampling temperature |
| `top_p` | `0.95` | Top-p (nucleus) sampling threshold |
| `split_pattern` | `"\n"` | Pattern to split text into segments |
| `max_tokens` | `None` | Maximum number of tokens to generate |

!!! tip "Dialogue format"
    Use `[S1]` and `[S2]` tags at the start of each speaker's line. Dia will automatically separate turns and generate distinct voices for each speaker.

## Links

- [:octicons-mark-github-16: Source code](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/dia)
- [:octicons-link-external-16: mlx-community/Dia-1.6B-fp16](https://huggingface.co/mlx-community/Dia-1.6B-fp16)
