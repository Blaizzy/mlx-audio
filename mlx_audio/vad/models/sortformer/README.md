# Sortformer Speaker Diarization

MLX port of NVIDIA's [Sortformer](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) speaker diarization model. Sortformer predicts "who spoke when" by outputting per-frame speaker activity probabilities for up to 4 speakers.

## Architecture

1. **FastConformer Encoder** - Conv subsampling (8x) + 18 Conformer layers with relative positional attention
2. **Transformer Encoder** - 18 BART-style post-LN encoder layers with learned positional embeddings
3. **Sortformer Modules** - Linear projection + feedforward + sigmoid output for 4 speakers

## Quick Start

```python
from mlx_audio.vad import load

model = load("nvidia/diar_sortformer_4spk-v1")
result = model.generate("audio.wav", threshold=0.5, verbose=True)

# Print RTTM-formatted output
print(result.text)
```

## API

### `model.generate()`

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

### RTTM Output

```
SPEAKER audio 1 0.000 3.200 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER audio 1 3.520 5.120 <NA> <NA> speaker_1 <NA> <NA>
```

## Examples

### Basic diarization

```python
from mlx_audio.vad import load

model = load("nvidia/diar_sortformer_4spk-v1")
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

### From mlx array

```python
from mlx_audio.audio_io import read as audio_read

audio, sr = audio_read("meeting.wav")
result = model.generate(audio, sample_rate=sr)
```


### Streaming diarization

`generate_stream` processes audio incrementally, yielding results per chunk.
The model maintains a speaker cache and FIFO buffer for long-range context.

**From a file path** (global normalization, auto-chunked):

```python
from mlx_audio.vad import load

model = load("nvidia/diar_sortformer_4spk-v1")

for result in model.generate_stream("meeting.wav", chunk_duration=5.0, verbose=True):
    for seg in result.segments:
        print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

**From a list of chunks** (per-chunk normalization):

```python
import soundfile as sf

audio, sr = sf.read("meeting.wav")
chunk_size = int(5.0 * sr)
chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

for result in model.generate_stream(chunks, sample_rate=sr):
    for seg in result.segments:
        print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

**One chunk at a time** (real-time streaming, e.g. from a microphone):

```python
state = model.init_streaming_state()
for chunk in mic_stream():  # your audio source
    for result in model.generate_stream(chunk, state=state, sample_rate=16000):
        state = result.state
        for seg in result.segments:
            print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

Streaming parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_duration` | `5.0` | Seconds per chunk (file/array mode) |
| `state` | `None` | Streaming state for single-chunk mode |
| `spkcache_max` | `188` | Max speaker cache size (diarization frames) |
| `fifo_max` | `188` | Max FIFO buffer size (diarization frames) |

### Visualization

```python
import urllib.request
import matplotlib.pyplot as plt
from mlx_audio.vad import load

# Load model and perform diarization
model = load("nvidia/diar_sortformer_4spk-v1")
result = model.generate("meeting.wav", threshold=0.5, verbose=True)

# Print segments
for seg in result.segments:
    print(f"Speaker {seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")

# Visualization
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
- The model supports up to 4 simultaneous speakers
- Lower `threshold` values detect more speaker activity (more sensitive, possibly noisier)
- Use `min_duration` and `merge_gap` to clean up fragmented segments
- Ported from [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) `SortformerEncLabelModel`
