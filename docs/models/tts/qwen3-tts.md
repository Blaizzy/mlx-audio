# Qwen3-TTS

Alibaba's state-of-the-art multilingual TTS with three model variants covering voice cloning, emotion control, and voice design from text descriptions. Supports streaming and batched generation.

## Model Variants

| Model | Method | Description | HuggingFace |
|-------|--------|-------------|-------------|
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | `generate()` | Fast, predefined voices | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16) |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | `generate()` | Higher quality | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16) |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16` | `generate_custom_voice()` | Voices + emotion | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16) |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | `generate_custom_voice()` | Better emotion control | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16) |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | `generate_voice_design()` | Create any voice from description | [:octicons-link-external-16: Model Card](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16) |

## Usage

### Basic Generation

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 \
        --text "Hello, welcome to MLX-Audio!" \
        --voice Chelsie
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
    results = list(model.generate(
        text="Hello, welcome to MLX-Audio!",
        voice="Chelsie",
        language="English",
    ))

    audio = results[0].audio  # mx.array
    ```

### Voice Cloning

Clone any voice by providing a reference audio sample and its transcript:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
results = list(model.generate(
    text="Hello from Sesame.",
    ref_audio="sample_audio.wav",
    ref_text="This is what my voice sounds like.",
))

audio = results[0].audio  # mx.array
```

### CustomVoice (Emotion Control)

Use predefined voices with emotion and style instructions:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")
results = list(model.generate_custom_voice(
    text="I'm so excited to meet you!",
    speaker="Vivian",
    language="English",
    instruct="Very happy and excited.",
))

audio = results[0].audio  # mx.array
```

### VoiceDesign (Create Any Voice)

Create a voice from a free-form text description:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
results = list(model.generate_voice_design(
    text="Big brother, you're back!",
    language="English",
    instruct="A cheerful young female voice with high pitch and energetic tone.",
))

audio = results[0].audio  # mx.array
```

### Streaming

All generation methods support `stream=True` for low-latency playback:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit")

audio_chunks = []
for result in model.generate(
    text="Hello, how are you today?",
    voice="serena",
    stream=True,
    streaming_interval=0.32,  # ~4 tokens at 12.5Hz
):
    audio_chunks.append(result.audio)
    # Play or process each chunk for low-latency output
```

### Batch Generation

Generate multiple texts with different voices in a single batched forward pass:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit")

texts = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Good morning, welcome to the show!",
]
voices = ["serena", "vivian", "ryan", "aiden"]

for result in model.batch_generate(
    texts=texts,
    voices=voices,
    stream=True,
    streaming_interval=0.32,
):
    audio_chunk = result.audio       # mx.array [samples]
    seq_idx = result.sequence_idx    # which sequence (0-3)
    is_done = result.is_final_chunk  # True on last chunk
```

!!! info "Batch throughput (6-bit, short prompt)"
    | Batch | TPS | Throughput | Avg TTFB | Memory |
    |-------|-----|------------|----------|--------|
    | 1 | 20.8 | 1.67x | 84.8ms | 3.88GB |
    | 2 | 34.7 | 2.78x | 78.0ms | 3.92GB |
    | 4 | 53.2 | 4.26x | 99.9ms | 3.98GB |
    | 8 | 68.1 | 5.45x | 140.5ms | 4.10GB |

## Available Speakers

**Chinese:** `Vivian`, `Serena`, `Uncle_Fu`, `Dylan` (Beijing Dialect), `Eric` (Sichuan Dialect)

**English:** `Ryan`, `Aiden`

## Links

- [:octicons-mark-github-16: Source code](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/qwen3_tts)
- [:octicons-mark-github-16: In-repo README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/qwen3_tts/README.md)
