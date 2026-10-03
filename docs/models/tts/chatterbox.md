# Chatterbox

Chatterbox is an expressive 0.5B TTS model by [ResembleAI](https://huggingface.co/ResembleAI/chatterbox) with voice cloning and fine-grained emotion control. MLX-Audio supports the English model and the 23-language multilingual v2 and v3 checkpoints. V3 improves speaker similarity, stability, and conversational speech while retaining the v2 architecture.

## Model Variants

| Model | Variant | HuggingFace |
|-------|---------|-------------|
| `mlx-community/chatterbox-multilingual-v3` | Multilingual v3 | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/chatterbox-multilingual-v3) |
| `mlx-community/chatterbox-fp16` | Multilingual v2 | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/chatterbox-fp16) |

!!! note
    Chatterbox requires the S3Tokenizer weights from [mlx-community/S3TokenizerV2](https://huggingface.co/mlx-community/S3TokenizerV2), which are downloaded automatically on first use.

To convert the official multilingual v3 checkpoint locally:

```bash
python -m mlx_audio.tts.models.chatterbox.scripts.convert \
    --variant v3 \
    --output-dir Chatterbox-Multilingual-v3-MLX
```

## Usage

### Basic Generation with Voice Cloning

Chatterbox requires a reference audio for voice cloning:

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/chatterbox-multilingual-v3 \
        --text "Hello, this is Chatterbox on MLX!" \
        --ref_audio reference.wav
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/chatterbox-multilingual-v3")

    for result in model.generate(
        text="Hello, this is Chatterbox on MLX!",
        ref_audio="reference.wav",
    ):
        audio = result.audio  # mx.array waveform
    ```

### Emotion Exaggeration

Control expressiveness with the `exaggeration` parameter (0 to 1):

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/chatterbox-multilingual-v3")

# Subtle expression
for result in model.generate(
    text="That's really interesting.",
    ref_audio="reference.wav",
    exaggeration=0.1,
):
    audio = result.audio

# Highly expressive
for result in model.generate(
    text="That's really interesting!",
    ref_audio="reference.wav",
    exaggeration=0.9,
):
    audio = result.audio
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exaggeration` | `0.1` | Emotion exaggeration factor (0-1) |
| `cfg_weight` | `0.5` | Classifier-free guidance weight |
| `temperature` | `0.8` | Sampling temperature |
| `repetition_penalty` | `1.2` | Penalty for repeated tokens |
| `min_p` | `0.05` | Minimum probability threshold |
| `top_p` | `1.0` | Top-p (nucleus) sampling threshold |
| `max_new_tokens` | `1000` | Maximum number of tokens to generate |

## Supported Languages

Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, and Chinese.

## Links

- [:octicons-mark-github-16: Source code](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/chatterbox)
- [:octicons-link-external-16: mlx-community/chatterbox-multilingual-v3](https://huggingface.co/mlx-community/chatterbox-multilingual-v3)
- [:octicons-link-external-16: mlx-community/chatterbox-fp16](https://huggingface.co/mlx-community/chatterbox-fp16)
- [:octicons-link-external-16: ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) (original model)
