# Sortformer Speaker Diarization

MLX port of NVIDIA's Sortformer speaker diarization models. Sortformer predicts "who spoke when" by outputting per-frame speaker activity probabilities for up to 4 speakers.

## Supported Models

| Model | Mel Bins | FC Layers | Streaming | Repo |
|-------|----------|-----------|-----------|------|
| **Sortformer v1** | 80 | 18 | Basic | [mlx-community/diar_sortformer_4spk-v1-fp16](https://huggingface.co/mlx-community/diar_sortformer_4spk-v1-fp16) |
| **Sortformer v2.1** | 128 | 17 | AOSC | [mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16](https://huggingface.co/mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16) |

**v1** is available directly on HuggingFace as safetensors. **v2.1** is distributed as a `.nemo` archive and must be converted first (see [Converting v2.1](#converting-v21-from-nemo)).

## Architecture

1. **FastConformer Encoder** - Conv subsampling (8x) + Conformer layers with relative positional attention
2. **Transformer Encoder** - BART-style post-LN encoder layers with positional embeddings
3. **Sortformer Modules** - Linear projection + feedforward + sigmoid output for 4 speakers

### v2.1 Differences

v2.1 introduces several improvements over v1:

- **128 mel bins** (vs 80) for richer spectral representation
- **17 Conformer layers** (vs 18), slightly lighter
- **AOSC (Arrival-Order Speaker Cache)** compression for intelligent streaming context management
- **Left/right context** for chunk boundary handling
- **Silence profiling** to maintain speaker cache quality over long sessions
- **No per-feature normalization** in streaming mode for lower-latency processing

## Quick Start

### v1 (Direct Load)

```python
from mlx_audio.vad import load

model = load("mlx-community/diar_sortformer_4spk-v1-fp32")
result = model.generate("audio.wav", threshold=0.5, verbose=True)
print(result.text)
```

### v2.1 (Requires Conversion)

```bash
# Convert from NeMo format
python -m mlx_audio.vad.models.sortformer.convert \
    --nemo-path nvidia/diar_streaming_sortformer_4spk-v2.1 \
    --output-dir ./sortformer-v2.1-mlx
```

```python
from mlx_audio.vad import load

model = load("./sortformer-v2.1-mlx")

# Streaming inference (recommended for v2.1)
for result in model.generate_stream("meeting.wav", chunk_duration=5.0, verbose=True):
    for seg in result.segments:
        print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

## Converting v2.1 from NeMo

The v2.1 model is distributed as a `.nemo` tar archive. The conversion script extracts weights, remaps keys, transposes convolution weights, and generates a `config.json` + `model.safetensors`.

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

Conversion requires `torch`, `pyyaml`, and `huggingface_hub`.

## API

### `model.generate()`

Offline inference on a full audio file.

```python
result = model.generate(
    audio,                    # str (file path), np.ndarray, or mx.array
    sample_rate=16000,        # sample rate of input audio
    threshold=0.5,            # speaker activity threshold (0-1)
    min_duration=0.0,         # minimum segment duration in seconds
    merge_gap=0.0,            # max gap (seconds) to merge consecutive segments
    verbose=False,            # print progress info
)
```

**Returns** a `DiarizationOutput` with:

| Field | Type | Description |
|-------|------|-------------|
| `segments` | `List[DiarizationSegment]` | Speaker segments with `start`, `end`, `speaker` |
| `speaker_probs` | `mx.array` | Per-frame speaker probabilities `(num_frames, 4)` |
| `num_speakers` | `int` | Number of detected active speakers |
| `total_time` | `float` | Processing time in seconds |
| `text` | `str` (property) | RTTM-formatted output |

### `model.generate_stream()`

Streaming inference that processes audio in chunks.

```python
for result in model.generate_stream(
    audio,                    # str, np.ndarray, mx.array, or Iterable[np.ndarray]
    state=None,               # StreamingState for single-chunk mode
    chunk_duration=5.0,       # seconds per chunk (file/array mode)
    threshold=0.5,
    min_duration=0.0,
    merge_gap=0.0,
    spkcache_max=188,         # max speaker cache size (diarization frames)
    fifo_max=188,             # max FIFO buffer size (diarization frames)
    verbose=False,
):
    ...
```

### `model.feed()`

Low-level single-chunk API for real-time streaming.

```python
state = model.init_streaming_state()
result, state = model.feed(
    chunk,                    # np.ndarray or mx.array (1-D audio samples)
    state,                    # StreamingState
    sample_rate=16000,
    threshold=0.5,
    spkcache_max=188,
    fifo_max=188,
)
```

### RTTM Output

```
SPEAKER audio 1 0.000 3.200 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER audio 1 3.520 5.120 <NA> <NA> speaker_1 <NA> <NA>
```

## Examples

### Basic diarization

```python
from mlx_audio.vad import load

model = load("mlx-community/diar_sortformer_4spk-v1-fp32")
result = model.generate("meeting.wav", threshold=0.5)

for seg in result.segments:
    print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

### With post-processing

```python
result = model.generate(
    "meeting.wav",
    threshold=0.4,
    min_duration=0.25,   # ignore segments shorter than 250ms
    merge_gap=0.5,       # merge segments within 500ms of each other
)
```

### Streaming from a file (v2.1)

```python
from mlx_audio.vad import load

model = load("./sortformer-v2.1-mlx")

for result in model.generate_stream("meeting.wav", chunk_duration=5.0, verbose=True):
    for seg in result.segments:
        print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

### Streaming from a list of chunks

```python
import soundfile as sf

audio, sr = sf.read("meeting.wav")
chunk_size = int(5.0 * sr)
chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

for result in model.generate_stream(chunks, sample_rate=sr):
    for seg in result.segments:
        print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

### Real-time streaming (e.g. microphone)

```python
state = model.init_streaming_state()
for chunk in mic_stream():  # your audio source
    for result in model.generate_stream(chunk, state=state, sample_rate=16000):
        state = result.state
        for seg in result.segments:
            print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

Or using the lower-level `feed()` API:

```python
state = model.init_streaming_state()
for chunk in mic_stream():
    result, state = model.feed(chunk, state, sample_rate=16000)
    for seg in result.segments:
        print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

### Visualization

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

## Streaming Architecture

The streaming pipeline maintains two buffers of pre-encoded embeddings:

```
[spkcache | fifo | left_ctx | new_chunk | right_ctx]
     ^         ^        ^          ^            ^
  long-term  recent  overlap    current      look-ahead
  context    context  from fifo  audio       (file mode)
```

- **Speaker Cache (spkcache)**: Long-term context, compressed when full to retain the most informative frames
- **FIFO**: Recent context buffer. Oldest frames roll into the speaker cache when the FIFO overflows
- **Left/Right Context** (v2.1): Overlap frames from adjacent chunks for better boundary handling

Each streaming step encodes the full assembled sequence through the Conformer + Transformer encoders, but only emits predictions for the new chunk.

### AOSC Compression (v2.1)

When the speaker cache overflows, v2.1 uses AOSC (Arrival-Order Speaker Cache) to intelligently select which frames to keep:

1. **Score** each frame per speaker using a log-likelihood ratio (high for confident non-overlapped speech)
2. **Filter** non-speech and overlapped-speech frames
3. **Boost** recent frames to add a recency bias
4. **Strong boost** top frames per speaker to guarantee minimum representation
5. **Weak boost** additional frames to prevent single-speaker dominance
6. **Pad** with silence slots to ensure silence is represented in the cache
7. **Select** top-K frames globally across all speakers
8. **Gather** selected embeddings, filling disabled slots with the running mean silence embedding

This produces a compressed cache that preserves the most informative frames from each speaker while maintaining temporal order.

### Streaming Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_duration` | `5.0` | Seconds per chunk (file/array mode) |
| `state` | `None` | Streaming state for single-chunk mode |
| `spkcache_max` | `188` | Max speaker cache size (diarization frames) |
| `fifo_max` | `188` | Max FIFO buffer size (diarization frames) |

For v2.1, `spkcache_max` and `fifo_max` are automatically set from the model config when using AOSC.

## Memory Considerations (v2.1 Streaming)

The v2.1 model was trained on chunks of up to **90 seconds** — using chunks larger than this will produce unreliable results and consume excessive memory. Each chunk is processed through **36 self-attention layers** (18 FastConformer + 18 Transformer), each creating a `(seq_len × seq_len)` attention score matrix, so memory scales **quadratically** with chunk duration:

| Chunk Duration | Frames after Subsampling | Attention Memory (per layer) | Total (~36 layers) |
|----------------|--------------------------|------------------------------|---------------------|
| 5 seconds      | ~63                      | ~0.12 MB                     | **~4 MB**           |
| 30 seconds     | ~375                     | ~4.3 MB                      | **~155 MB**         |
| 90 seconds     | ~1,125                   | ~38.6 MB                     | **~1.4 GB**         |
| 120 seconds    | ~1,500                   | ~68.7 MB                     | **~2.5 GB**         |

**Use small chunks (5-10 seconds)** to keep memory usage low. Using large chunks defeats the purpose of streaming — the streaming state object (`state`) already carries context across chunks, so results remain accurate without needing large chunks:

```python
# Good: small chunks, low memory
chunk_size = 5 * sr
chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

state = model.init_streaming_state()
for chunk in chunks:
    for result in model.generate_stream(chunk, state=state, sample_rate=sr):
        state = result.state
        ...

# Bad: large chunks cause memory explosion
chunk_size = 120 * sr  # ~2.5 GB in attention matrices alone
```

## Notes

- Input audio is automatically resampled to 16kHz and converted to mono
- The model supports up to 4 simultaneous speakers
- Lower `threshold` values detect more speaker activity (more sensitive, possibly noisier)
- Use `min_duration` and `merge_gap` to clean up fragmented segments
- v1 uses per-feature normalization and peak normalization; v2.1 streaming skips both for lower latency
- Ported from [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) `SortformerEncLabelModel`
