# VoxCPM2

[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) is OpenBMB's 2B-parameter,
multilingual tokenizer-free text-to-speech model. The MLX implementation produces
48 kHz audio and supports standard synthesis, voice design from a text description,
voice cloning from reference audio, and speech continuation.

## Model Variants

| Model | Format | Size | HuggingFace |
|-------|--------|------|-------------|
| `mlx-community/VoxCPM2-bf16` | bfloat16 | 4.96 GB | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/VoxCPM2-bf16) |
| `mlx-community/VoxCPM2-8bit` | 8-bit | 3.23 GB | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/VoxCPM2-8bit) |
| `mlx-community/VoxCPM2-4bit` | 4-bit | 2.30 GB | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/VoxCPM2-4bit) |

## Quick Start

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/VoxCPM2-8bit \
        --text "Hello, this is VoxCPM2 running on Apple Silicon." \
        --output_path outputs
    ```

=== "Python"

    ```python
    from mlx_audio.tts import load

    model = load("mlx-community/VoxCPM2-8bit")

    result = next(
        model.generate("Hello, this is VoxCPM2 running on Apple Silicon.")
    )
    audio = result.audio  # 48 kHz mono waveform as an MLX array
    ```

## Voice Design

Describe the desired speaker and delivery with `instruct`; no reference audio is
required.

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/VoxCPM2-8bit \
        --text "Welcome. I hope you enjoy the presentation." \
        --instruct "A warm, confident young woman speaking at a relaxed pace" \
        --output_path outputs
    ```

=== "Python"

    ```python
    result = next(
        model.generate(
            text="Welcome. I hope you enjoy the presentation.",
            instruct="A warm, confident young woman speaking at a relaxed pace",
        )
    )
    ```

## Voice Cloning

Pass a clean reference recording to synthesize the target text in the reference
speaker's voice.

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/VoxCPM2-8bit \
        --text "This sentence uses the voice from the reference recording." \
        --ref_audio speaker.wav \
        --ref_text "The exact transcript of speaker.wav." \
        --output_path outputs
    ```

=== "Python"

    ```python
    result = next(
        model.generate(
            text="This sentence uses the voice from the reference recording.",
            ref_audio="speaker.wav",
        )
    )
    ```

`ref_text` is part of the common MLX-Audio CLI voice-cloning interface. Supplying
it prevents the CLI from running automatic reference-audio transcription;
VoxCPM2 conditions directly on the reference audio.

## Speech Continuation

For long-form generation, provide an earlier audio clip and its transcript. The
new text should begin where `prompt_text` ends.

```python
result = next(
    model.generate(
        text=" The story continues from here.",
        prompt_text="This is the opening of the story.",
        prompt_audio="opening.wav",
    )
)
```

Continuation is currently available through the Python API. Reference voice
cloning can be combined with continuation by also passing `ref_audio`.

## Generation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference_timesteps` | `10` | Diffusion steps; higher values trade speed for quality |
| `cfg_value` | `2.0` | Classifier-free guidance strength |
| `instruct` | `None` | Natural-language voice description for voice design |
| `ref_audio` | `None` | Reference audio path or waveform for voice cloning |
| `prompt_text` | `None` | Transcript of `prompt_audio` for continuation |
| `prompt_audio` | `None` | Previous audio clip used as continuation context |
| `warmup_patches` | `0` | Extra conditioning patches excluded from the output |
| `max_tokens` | `2000` | Maximum number of audio patches to generate |

The CLI exposes `--ddpm_steps` as an alias for `inference_timesteps` and
`--cfg_scale` as an alias for `cfg_value`. VoxCPM2 enforces a minimum guidance
value of `2.0`.

## License

VoxCPM2 is released under the
[Apache License 2.0](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE).

## Links

- [:octicons-mark-github-16: Source code](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/voxcpm2)
- [:octicons-mark-github-16: In-repo README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/voxcpm2/README.md)
- [:octicons-link-external-16: Original model](https://huggingface.co/openbmb/VoxCPM2)
