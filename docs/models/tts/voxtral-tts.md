# Voxtral TTS

Mistral's 4B parameter multilingual text-to-speech model with 20 expressive voice presets across 9 languages. Based on [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603).

## Model Variants

| Model | Format | HuggingFace |
|-------|--------|-------------|
| `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16` | bfloat16 | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-bf16) |

## Usage

=== "CLI"

    ```bash
    python -m mlx_audio.tts.generate \
        --model mlx-community/Voxtral-4B-TTS-2603-mlx-bf16 \
        --text "Hello, how are you today?" \
        --voice casual_male
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load

    model = load("mlx-community/Voxtral-4B-TTS-2603-mlx-bf16")

    for result in model.generate(text="Hello, how are you today?", voice="casual_male"):
        print(result.audio_duration)
    ```

## Available Voices

### English

| Voice | Style |
|-------|-------|
| `casual_male` | Casual |
| `casual_female` | Casual |
| `cheerful_female` | Cheerful |
| `neutral_male` | Neutral |
| `neutral_female` | Neutral |

### Multilingual

| Voice | Language |
|-------|----------|
| `fr_male`, `fr_female` | French |
| `es_male`, `es_female` | Spanish |
| `de_male`, `de_female` | German |
| `it_male`, `it_female` | Italian |
| `pt_male`, `pt_female` | Portuguese |
| `nl_male`, `nl_female` | Dutch |
| `ar_male` | Arabic |
| `hi_male`, `hi_female` | Hindi |

## Supported Languages

English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi.

!!! warning "License"
    Voxtral TTS weights are released under **CC-BY-NC** (non-commercial use). Check the [model card](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) for full licensing details.

## Links

- [:octicons-mark-github-16: Source code](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/voxtral_tts)
- [:octicons-mark-github-16: In-repo README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/voxtral_tts/README.md)
- [:octicons-link-external-16: mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
