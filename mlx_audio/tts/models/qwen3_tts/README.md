# Qwen3-TTS

Alibaba's state-of-the-art multilingual TTS with three model variants.

## Voice Cloning

Clone any voice using a reference audio sample. Provide the wav file and its transcript:

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

## CustomVoice (Emotion Control)

Use predefined voices with emotion/style instructions:

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

## VoiceDesign (Create Any Voice)

Create any voice from a text description:

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

## Streaming

All generation methods support streaming via `stream=True`. Audio chunks are yielded as they're produced, enabling low-latency playback:

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
    # Play or process each chunk here for low-latency output
```

The `streaming_interval` controls how frequently chunks are emitted (in seconds). Smaller values give lower latency but more overhead per chunk.

## Batch Generation

Generate multiple texts with different voices in a single batched forward pass. Near-linear throughput scaling with minimal memory overhead:

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

# Streaming: yields audio chunks per sequence as they're generated
for result in model.batch_generate(
    texts=texts,
    voices=voices,
    stream=True,
    streaming_interval=0.32,
):
    audio_chunk = result.audio       # mx.array [samples]
    seq_idx = result.sequence_idx    # which sequence (0-3)
    is_done = result.is_final_chunk  # True on last chunk for this sequence

# Non-streaming: yields one complete audio per sequence
for result in model.batch_generate(
    texts=texts,
    voices=voices,
    stream=False,
):
    audio = result.audio             # mx.array [samples]
    seq_idx = result.sequence_idx
```

**Benchmark (6-bit, short prompt):**

| Batch | TPS | Throughput | Avg TTFB | Memory |
|-------|-----|------------|----------|--------|
| 1 | 20.8 | 1.67x | 84.8ms | 3.88GB |
| 2 | 34.7 | 2.78x | 78.0ms | 3.92GB |
| 4 | 53.2 | 4.26x | 99.9ms | 3.98GB |
| 8 | 68.1 | 5.45x | 140.5ms | 4.10GB |

## Available Models

| Model | Method | Description |
|-------|--------|-------------|
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | `generate()` | Fast, predefined voices |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | `generate()` | Higher quality |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16` | `generate_custom_voice()` | Voices + emotion |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | `generate_custom_voice()` | Better emotion control |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | `generate_voice_design()` | Create any voice |

## Speakers (Base/CustomVoice)

**Chinese:** `Vivian`, `Serena`, `Uncle_Fu`, `Dylan` (Beijing Dialect), `Eric` (Sichuan Dialect)

**English:** `Ryan`, `Aiden`
