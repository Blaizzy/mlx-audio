# Kokoro

Kokoro is a fast, lightweight (82M parameter) multilingual TTS model with 54 built-in voice presets. It delivers high-quality speech synthesis with minimal resource usage, making it ideal for quick generation tasks on Apple Silicon.

## Model Variants

| Model | Format | HuggingFace |
|-------|--------|-------------|
| `mlx-community/Kokoro-82M-bf16` | bfloat16 | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/Kokoro-82M-bf16) |

## Installation

Kokoro requires `misaki` for text processing in all languages:

```bash
# Base dependency
pip install misaki

# Optional language support
pip install misaki[ja]
pip install misaki[zh]
```

## Usage

=== "CLI"

    ```bash
    # Basic generation (American English)
    mlx_audio.tts.generate \
        --model mlx-community/Kokoro-82M-bf16 \
        --text "Hello, world!" \
        --lang_code a

    # Choose a voice and adjust speed
    mlx_audio.tts.generate \
        --model mlx-community/Kokoro-82M-bf16 \
        --text "Welcome to MLX-Audio!" \
        --voice af_heart \
        --speed 1.2 \
        --lang_code a

    # Play audio immediately
    mlx_audio.tts.generate \
        --model mlx-community/Kokoro-82M-bf16 \
        --text "Hello!" \
        --play \
        --lang_code a
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/Kokoro-82M-bf16")

    for result in model.generate(
        text="Welcome to MLX-Audio!",
        voice="af_heart",  # American female
        speed=1.0,
        lang_code="a",     # American English
    ):
        audio = result.audio  # mx.array waveform
    ```

## Language Codes

| Code | Language | Extra Dependency |
|------|----------|-----------------|
| `a` | American English | `pip install misaki` |
| `b` | British English | `pip install misaki` |
| `j` | Japanese | `pip install misaki[ja]` |
| `z` | Mandarin Chinese | `pip install misaki[zh]` |
| `e` | Spanish | `pip install misaki` |
| `f` | French | `pip install misaki` |

## Available Voices

### American English

| Female | Male |
|--------|------|
| `af_heart` | `am_adam` |
| `af_bella` | `am_echo` |
| `af_nova` | |
| `af_sky` | |

### British English

| Female | Male |
|--------|------|
| `bf_alice` | `bm_daniel` |
| `bf_emma` | `bm_george` |

### Japanese

| Female | Male |
|--------|------|
| `jf_alpha` | `jm_kumo` |

### Chinese

| Female | Male |
|--------|------|
| `zf_xiaobei` | `zm_yunxi` |

!!! note
    Kokoro ships with 54 voice presets total. The voices listed above are a representative sample. Check the [model card](https://huggingface.co/mlx-community/Kokoro-82M-bf16) for the full list.

## Links

- [:octicons-mark-github-16: Source code](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/kokoro)
- [🤗 mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16)
