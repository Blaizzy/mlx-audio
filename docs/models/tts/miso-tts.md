# Miso TTS

Miso TTS 8B is Miso Labs' conversational text-to-speech model. It follows the
Sesame/CSM-style RVQ transformer design with a Llama 3.2-style 8B backbone, a
300M audio depth decoder, Mimi audio tokens, and optional audio context for
voice cloning or multi-turn dialogue.

## Model Variants

| Model | Format | HuggingFace |
|-------|--------|-------------|
| `mlx-community/MisoTTS-bf16` | MLX bf16 | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/MisoTTS-bf16) |
| `mlx-community/MisoTTS-8bit` | MLX 8-bit | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/MisoTTS-8bit) |
| `MisoLabs/MisoTTS` | upstream F32 safetensors | [:octicons-link-external-16: Original Model](https://huggingface.co/MisoLabs/MisoTTS) |

The MLX checkpoints include the `config.json` needed by the standard loader.
Miso uses the `meta-llama/Llama-3.2-1B` tokenizer with the same Llama BOS/EOS
template processing as the upstream PyTorch implementation.

## Usage

### Basic Generation

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/MisoTTS-bf16 \
        --text "Hello from Miso."
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/MisoTTS-bf16")

    for result in model.generate(
        text="Hello from Miso.",
        speaker=0,
        max_audio_length_ms=10_000,
    ):
        audio = result.audio
    ```

### Prompted Generation

Provide reference audio plus its transcript to condition the generated voice:

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/MisoTTS-bf16 \
        --text "This is the next sentence to synthesize." \
        --ref_audio ./prompt.wav \
        --ref_text "This is the transcript for the prompt audio."
    ```

=== "Python"

    ```python
    from mlx_audio.tts.models.miso_tts import Segment
    from mlx_audio.tts.utils import load_model
    from mlx_audio.utils import load_audio

    model = load_model("mlx-community/MisoTTS-bf16")
    prompt_audio = load_audio("prompt.wav", sample_rate=model.sample_rate)

    context = [
        Segment(
            speaker=0,
            text="This is the transcript for the prompt audio.",
            audio=prompt_audio,
        )
    ]

    for result in model.generate(
        text="This is the next sentence to synthesize.",
        speaker=0,
        context=context,
        max_audio_length_ms=10_000,
    ):
        audio = result.audio
    ```

### Quantize With The Standard Converter

The published MLX checkpoints are ready to use. To build a quantized variant
from the bf16 MLX checkpoint, use the standard converter:

```bash
python -m mlx_audio.convert \
    --hf-path mlx-community/MisoTTS-bf16 \
    --mlx-path ./MisoTTS-8bit \
    --model-domain tts \
    --quantize \
    --q-bits 8 \
    --upload-repo mlx-community/MisoTTS-8bit
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speaker` | `0` | Speaker ID inserted into the text frame as `[speaker]` |
| `context` | `None` | Previous `Segment` objects for voice/audio context |
| `ref_audio` | `None` | Reference audio path or array for voice cloning |
| `ref_text` | `None` | Transcript for `ref_audio` |
| `temperature` | `0.9` | Sampling temperature |
| `top_k` / `topk` | `50` | Top-k sampling limit |
| `max_audio_length_ms` | `90000` | Maximum generated audio length |
| `stream` | `False` | Yield audio chunks while generating |
| `streaming_interval` | `0.5` | Seconds of audio tokens between streaming chunks |

## Links

- [:octicons-mark-github-16: Source code](https://github.com/MisoLabsAI/MisoTTS)
- [:octicons-link-external-16: MLX bf16](https://huggingface.co/mlx-community/MisoTTS-bf16)
- [:octicons-link-external-16: MLX 8-bit](https://huggingface.co/mlx-community/MisoTTS-8bit)
- [:octicons-link-external-16: Upstream MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS)
