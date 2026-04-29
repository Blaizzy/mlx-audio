---
title: Voice Activity Detection Models
---

# Voice Activity Detection (VAD) Models

MLX Audio provides voice activity detection and speaker diarization models for identifying who is speaking when, as well as endpoint detection for conversational turn-taking.

| Model | Provider | Use Case | Speakers | Streaming | Repo |
|-------|----------|----------|----------|-----------|------|
| [**Sortformer v1**](#sortformer-v1) | NVIDIA | Speaker diarization | Up to 4 | Basic | [mlx-community/diar_sortformer_4spk-v1-fp16](https://huggingface.co/mlx-community/diar_sortformer_4spk-v1-fp16) |
| [**Sortformer v2.1**](#sortformer-v21) | NVIDIA | Streaming speaker diarization | Up to 4 | AOSC | [mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16](https://huggingface.co/mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16) |
| [**Smart Turn**](#smart-turn) | Pipecat AI | Turn endpoint detection | -- | -- | [mlx-community/smart-turn-v3](https://huggingface.co/mlx-community/smart-turn-v3) |

---

## Sortformer v1

NVIDIA's end-to-end speaker diarization model. Sortformer predicts "who spoke when" by outputting per-frame speaker activity probabilities for up to 4 speakers.

### Architecture

1. **FastConformer Encoder** -- Conv subsampling (8x) + Conformer layers with relative positional attention
2. **Transformer Encoder** -- BART-style post-LN encoder layers with positional embeddings
3. **Sortformer Modules** -- Linear projection + feedforward + sigmoid output for 4 speakers

### Quick Start

```python
from mlx_audio.vad import load

model = load("mlx-community/diar_sortformer_4spk-v1-fp32")
result = model.generate("meeting.wav", threshold=0.5, verbose=True)
print(result.text)
```

### With Post-Processing

```python
result = model.generate(
    "meeting.wav",
    threshold=0.4,
    min_duration=0.25,   # Ignore segments shorter than 250ms
    merge_gap=0.5,       # Merge segments within 500ms of each other
)

for seg in result.segments:
    print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

### Output Format

The `generate()` method returns a `DiarizationOutput` with:

| Field | Type | Description |
|-------|------|-------------|
| `segments` | `List[DiarizationSegment]` | Speaker segments with `start`, `end`, `speaker` |
| `speaker_probs` | `mx.array` | Per-frame speaker probabilities `(num_frames, 4)` |
| `num_speakers` | `int` | Number of detected active speakers |
| `total_time` | `float` | Processing time in seconds |
| `text` | `str` (property) | RTTM-formatted output |

### RTTM Output

```
SPEAKER audio 1 0.000 3.200 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER audio 1 3.520 5.120 <NA> <NA> speaker_1 <NA> <NA>
```

---

## Sortformer v2.1

An improved streaming variant of Sortformer with AOSC (Arrival-Order Speaker Cache) compression for intelligent context management during long sessions.

### Improvements over v1

- **128 mel bins** (vs 80) for richer spectral representation
- **AOSC compression** for intelligent streaming context management
- **Left/right context** for chunk boundary handling
- **Silence profiling** to maintain speaker cache quality over long sessions
- **No per-feature normalization** in streaming mode for lower-latency processing

!!! warning "Conversion required"
    The v2.1 model is distributed as a `.nemo` archive and must be converted to MLX format before use.

### Converting from NeMo

```bash
# From HuggingFace repo ID (downloads automatically)
python -m mlx_audio.vad.models.sortformer.convert \
    --nemo-path nvidia/diar_streaming_sortformer_4spk-v2.1 \
    --output-dir ./sortformer-v2.1-mlx

# From a local .nemo file
python -m mlx_audio.vad.models.sortformer.convert \
    --nemo-path /path/to/model.nemo \
    --output-dir ./sortformer-v2.1-mlx

# Convert and upload to HuggingFace
python -m mlx_audio.vad.models.sortformer.convert \
    --nemo-path nvidia/diar_streaming_sortformer_4spk-v2.1 \
    --output-dir ./sortformer-v2.1-mlx \
    --upload your-username/sortformer-v2.1-mlx
```

!!! info "Conversion requirements"
    Conversion requires `torch`, `pyyaml`, and `huggingface_hub`.

### Streaming from a File

```python
from mlx_audio.vad import load

model = load("./sortformer-v2.1-mlx")

for result in model.generate_stream("meeting.wav", chunk_duration=5.0, verbose=True):
    for seg in result.segments:
        print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

### Streaming from Audio Chunks

```python
from mlx_audio.audio_io import read as audio_read

audio, sr = audio_read("meeting.wav")
chunk_size = int(5.0 * sr)
chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

for result in model.generate_stream(chunks, sample_rate=sr):
    for seg in result.segments:
        print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

### Real-Time Streaming (Microphone)

=== "generate_stream API"

    ```python
    state = model.init_streaming_state()
    for chunk in mic_stream():
        for result in model.generate_stream(chunk, state=state, sample_rate=16000):
            state = result.state
            for seg in result.segments:
                print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
    ```

=== "Low-level feed API"

    ```python
    state = model.init_streaming_state()
    for chunk in mic_stream():
        result, state = model.feed(chunk, state, sample_rate=16000)
        for seg in result.segments:
            print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
    ```

### Streaming Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_duration` | `5.0` | Seconds per chunk (file/array mode) |
| `state` | `None` | Streaming state for single-chunk mode |
| `spkcache_max` | `188` | Max speaker cache size (diarization frames) |
| `fifo_max` | `188` | Max FIFO buffer size (diarization frames) |

### Memory Considerations

The v2.1 model was trained on chunks of up to 90 seconds. Memory scales quadratically with chunk duration due to self-attention:

| Chunk Duration | Attention Memory (~36 layers) |
|----------------|-------------------------------|
| 5 seconds | ~4 MB |
| 30 seconds | ~155 MB |
| 90 seconds | ~1.4 GB |
| 120 seconds | ~2.5 GB |

!!! tip "Use small chunks"
    Use 5-10 second chunks to keep memory usage low. The streaming state object carries context across chunks, so results remain accurate without large chunks.

---

## Smart Turn

[Smart Turn](https://huggingface.co/pipecat-ai/smart-turn-v3) by Pipecat AI predicts whether a user's conversational turn is complete or incomplete from audio. Useful for building conversational AI systems that need to determine when the user has finished speaking.

### How It Works

- Accepts up to 8 seconds of 16kHz mono audio
- Audio shorter than 8 seconds is left-padded with zeros
- Audio longer than 8 seconds uses the last 8 seconds
- Returns a binary prediction: `1` (complete) or `0` (incomplete) with a probability score

### Usage

```python
from mlx_audio.vad import load

model = load("mlx-community/smart-turn-v3", strict=True)

result = model.predict_endpoint("audio.wav")
print("prediction:", result.prediction)   # 1 = complete, 0 = incomplete
print("probability:", result.probability)
```

### With an MLX Array

```python
import mlx.core as mx
import numpy as np
from mlx_audio.vad import load

model = load("mlx-community/smart-turn-v3")
audio = mx.zeros(16000, dtype=np.float32)  # 1 second at 16kHz
result = model.predict_endpoint(audio, sample_rate=16000, threshold=0.5)
```

!!! tip "Typical usage"
    Run Smart Turn after silence is detected by a lightweight VAD. Use the full current-turn context instead of very short snippets for best results.

---

## Visualization

Sortformer diarization results can be visualized with matplotlib:

```python
import matplotlib.pyplot as plt
from mlx_audio.vad import load

model = load("mlx-community/diar_sortformer_4spk-v1-fp32")
result = model.generate("meeting.wav", threshold=0.5, verbose=True)

SPEAKER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=(12, 3))

for seg in result.segments:
    ax.barh(
        y=f"Speaker {seg.speaker}",
        width=seg.end - seg.start,
        left=seg.start,
        height=0.6,
        color=SPEAKER_COLORS[seg.speaker % len(SPEAKER_COLORS)],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

ax.set_xlabel("Time (s)")
ax.set_title("Speaker Diarization")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()
```

## Notes

- Input audio is automatically resampled to 16kHz and converted to mono
- Sortformer supports up to 4 simultaneous speakers
- Lower `threshold` values detect more speaker activity (more sensitive, possibly noisier)
- Use `min_duration` and `merge_gap` to clean up fragmented segments
- Ported from [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) `SortformerEncLabelModel`
