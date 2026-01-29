# Qwen3-ASR & Qwen3-ForcedAligner

MLX implementations of Qwen3 speech models for transcription and word-level alignment.

## Available Models

| Model | Parameters | Description |
|-------|------------|-------------|
| [mlx-community/Qwen3-ASR-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit) | 0.6B | Speech recognition (8-bit quantized) |
| [mlx-community/Qwen3-ASR-1.7B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit) | 1.7B | Speech recognition (8-bit quantized) |
| [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) | 0.6B | Word-level alignment (8-bit quantized) |

**Supported Languages:** Chinese, Cantonese, English, German, Spanish, French, Italian, Portuguese, Russian, Korean, Japanese

## CLI Usage

### Speech Recognition (ASR)

```bash
# Basic transcription
uv run mlx_audio.stt.generate --model mlx-community/Qwen3-ASR-0.6B-8bit --audio audio.wav

# With language specification
uv run mlx_audio.stt.generate --model mlx-community/Qwen3-ASR-1.7B-8bit --audio audio.wav --language English

# Streaming output
uv run mlx_audio.stt.generate --model mlx-community/Qwen3-ASR-0.6B-8bit --audio audio.wav --stream
```

### Forced Alignment

```bash
# Word-level alignment (requires text input)
uv run mlx_audio.stt.generate --model mlx-community/Qwen3-ForcedAligner-0.6B-8bit \
    --audio audio.wav \
    --text "The transcript to align" \
    --language English
```

## Python Usage

### Speech Recognition

```python
from mlx_audio.stt import load

# Load model
model = load("mlx-community/Qwen3-ASR-0.6B-8bit")

# Transcribe audio
result = model.generate("audio.wav", language="English")
print(result.text)

# With timing info
for segment in result.segments:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
```

### Streaming Transcription

```python
from mlx_audio.stt import load

model = load("mlx-community/Qwen3-ASR-0.6B-8bit")

# Stream tokens as they're generated
for text in model.stream_transcribe("audio.wav", language="English"):
    print(text, end="", flush=True)
```

### Forced Alignment

```python
from mlx_audio.stt import load

# Load forced aligner
model = load("mlx-community/Qwen3-ForcedAligner-0.6B-8bit")

# Align text to audio (model.align is also available as an alias)
result = model.generate(
    audio="audio.wav",
    text="I have a dream that one day",
    language="English"
)

# Print word-level timestamps
for item in result:
    print(f"[{item.start_time:.2f}s - {item.end_time:.2f}s] {item.text}")
```

### Batch Alignment

```python
from mlx_audio.stt import load

model = load("mlx-community/Qwen3-ForcedAligner-0.6B-8bit")

# Align multiple audio-text pairs
results = model.generate(
    audio=["audio1.wav", "audio2.wav"],
    text=["First transcript", "Second transcript"],
    language="English"
)

for i, result in enumerate(results):
    print(f"\nSample {i + 1}:")
    for item in result:
        print(f"  [{item.start_time:.2f}s - {item.end_time:.2f}s] {item.text}")
```

### Check Supported Languages

```python
from mlx_audio.stt import load

model = load("mlx-community/Qwen3-ASR-0.6B-8bit")
print(model.get_supported_languages())
# ['cantonese', 'chinese', 'english', 'french', 'german', 'italian',
#  'japanese', 'korean', 'portuguese', 'russian', 'spanish']
```

## Output Format

### ASR Output

```python
STTOutput(
    text="Full transcription text",
    segments=[
        {"text": "segment text", "start": 0.0, "end": 2.5},
        ...
    ],
    prompt_tokens=1234,
    generation_tokens=56,
    total_tokens=1290,
    total_time=3.2,
    prompt_tps=385.6,
    generation_tps=17.5
)
```

### Forced Alignment Output

```python
ForcedAlignResult(
    items=[
        ForcedAlignItem(text="word", start_time=0.12, end_time=0.45),
        ...
    ]
)
```
