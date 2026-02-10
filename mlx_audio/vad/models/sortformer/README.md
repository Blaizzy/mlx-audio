# Sortformer Speaker Diarization

MLX port of NVIDIA's [Sortformer](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) speaker diarization model. Sortformer predicts "who spoke when" by outputting per-frame speaker activity probabilities for up to 4 speakers.

## Architecture

1. **FastConformer Encoder** - Conv subsampling (8x) + 18 Conformer layers with relative positional attention
2. **Transformer Encoder** - 18 BART-style post-LN encoder layers with learned positional embeddings
3. **Sortformer Modules** - Linear projection + feedforward + sigmoid output for 4 speakers

## Quick Start

```python
from mlx_audio.stt import load

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
from mlx_audio.stt import load

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

## Notes

- Input audio is automatically resampled to 16kHz and converted to mono
- The model supports up to 4 simultaneous speakers
- Lower `threshold` values detect more speaker activity (more sensitive, possibly noisier)
- Use `min_duration` and `merge_gap` to clean up fragmented segments
- Ported from [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) `SortformerEncLabelModel`
