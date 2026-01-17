# SAM-Audio Usage Guide

MLX implementation of [SAM-Audio](https://github.com/facebookresearch/sam-audio) (Segment Anything Model for Audio) - a foundation model for audio source separation using text prompts.


## Installation

```bash
pip install mlx-audio
```

## Quick Start

```python
from mlx_audio.sts import SAMAudio, save_audio

# Load model
model = SAMAudio.from_pretrained("facebook/sam-audio-large")

# Separate audio - pass file paths directly!
result = model.separate(
    audios=["path/to/audio.mp3"],  # File paths or mx.array
    descriptions=["speech"],
)

# Save output
save_audio(result.target[0], "separated.wav", sample_rate=model.sample_rate)
print(f"Peak memory: {result.peak_memory:.2f} GB")
```

### With Temporal Anchors

Anchors tell the model which time spans contain the target sound:

```python
from mlx_audio.sts import SAMAudio, save_audio

model = SAMAudio.from_pretrained("facebook/sam-audio-large")

# Use anchors directly with file paths
result = model.separate(
    audios=["path/to/audio.mp3"],
    descriptions=["speech"],
    anchors=[[("+", 1.5, 3.0)]],  # Target speech from 1.5s to 3.0s
)

save_audio(result.target[0], "separated.wav", sample_rate=model.sample_rate)
```

**Note:** Anchors are only supported with `separate()`, not with chunked methods (`separate_long()`, `separate_streaming()`).

## Methods

### `separate()` - Standard Separation

Best for short audio files (< 30 seconds) or when you have enough memory.

```python
result = model.separate(
    audios=["audio.mp3"],             # File paths or (B, 1, T) tensor
    descriptions=["speech"],          # List of text prompts
    anchors=[[("+", 1.0, 5.0)]],      # Optional: temporal anchors (with file paths)
    ode_opt=None,                     # ODE solver options (see below)
    ode_decode_chunk_size=50,         # Chunked decoding (reduces memory)
)
```

### `separate_long()` - Chunked Processing

Best for long audio files or limited memory. Processes audio in chunks with crossfade blending.

**Note:** Temporal anchors are not supported with chunked processing.

```python
result = model.separate_long(
    audios=["long_audio.mp3"],    # File paths or (B, 1, T) tensor
    descriptions=["speech"],
    chunk_seconds=10.0,           # Chunk size (default: 10s)
    overlap_seconds=3.0,          # Overlap for crossfade (default: 3s)
    ode_opt=None,                 # ODE solver options
    seed=42,                      # Random seed for reproducibility
    verbose=True,                 # Print progress
)
```

## Temporal Anchors

Anchors are **temporal prompts** that tell the model which time spans contain the target sound. Use them when the text description alone isn't specific enough (e.g., multiple speakers in audio).

### Anchor Tokens

| Token | Meaning |
|-------|---------|
| `"+"` | Positive - "the target sound IS here" |
| `"-"` | Negative - "the target sound is NOT here" |

### Format

```python
anchors=[[(token, start_time, end_time), ...]]  # times in seconds
```

### Examples

```python
# Extract speech occurring between 1.5s and 3.0s
batch = processor(
    descriptions=["speech"],
    audios=["audio.wav"],
    anchors=[[("+", 1.5, 3.0)]],
)

# Multiple anchors - target is here but NOT there
batch = processor(
    descriptions=["speech"],
    audios=["audio.wav"],
    anchors=[[("+", 1.5, 3.0), ("-", 5.0, 7.0)]],
)

# Batch processing with different anchors per sample
batch = processor(
    descriptions=["speech", "music"],
    audios=["audio1.wav", "audio2.wav"],
    anchors=[
        [("+", 0.0, 2.0)],           # First sample
        [("+", 1.0, 4.0), ("-", 6.0, 8.0)],  # Second sample
    ],
)
```

### Use Cases

- **Multiple speakers**: Use `+` anchor to specify which speaker's time range
- **Intermittent sounds**: Mark where the target appears and doesn't appear
- **Fine-grained control**: When text prompts are ambiguous

## ODE Solver Options

The separation quality vs speed tradeoff is controlled by `ode_opt`.

**Reference default** (from official SAM-Audio):
```python
DFLT_ODE_OPT = {"method": "midpoint", "step_size": 2/32}  # 16 midpoint steps
```

| Method | Steps | Speed | Quality | Use Case |
|--------|-------|-------|---------|----------|
| `midpoint` | 2 / 64 (32 steps) | 0.125x | Maximum | Studio quality, no artifacts |
| `midpoint` | 2 / 32 (16 steps) | 0.25x | Best | **Official default**, quality priority |
| `midpoint` | 2 / 16 (8 steps) | 0.5x | Good | Real-time, speed priority |
| `midpoint` | 2 / 8 (4 steps) | 1x | Good | Real-time, speed priority |
| `midpoint` | 2 / 4 (2 steps) | 2x | Ok | Real-time, speed priority |
| `euler` | 2 / 64 (32 steps) | ~0.5x | Very Good | Long audio, balanced |
| `euler` | 2 / 32 (16 steps) | ~1x | Good | Real-time, speed priority |
| `euler` | 2 / 16 (8 steps) | ~2x | Good | Real-time, speed priority |
| `euler` | 2 / 8 (4 steps) | ~4x | Good | Real-time, speed priority |
| `euler` | 2 / 4 (2 steps) | ~8x | OK | Real-time, speed priority |

Note: The step size must be between 0 and 1 (exclusive). For instance, use step_size (2 / 32) = 0.0625 for 16 steps.

### Configuration Examples

```python
# Maximum Quality - 32 midpoint steps (slowest, cleanest)
ode_opt = {"method": "midpoint", "step_size": 2/64}  # 32 midpoint steps

# Official Default - 16 midpoint steps
ode_opt = {"method": "midpoint", "step_size": 2/32}  # 16 midpoint steps

# Balanced - Good for separate_long()
ode_opt = {"method": "euler", "step_size": 2/64}     # 32 euler steps

# Fastest - Real-time, may have artifacts
ode_opt = {"method": "euler", "step_size": 2/32}     # 16 euler steps
```

## Inference Recommendations

**Recommended audio length**: ~10 seconds (training data was around 10s). For longer audio, use `separate_long()` with chunked processing.

### Short Audio (< 30s)

Use `separate()` with default settings:

```python
result = model.separate(
    audios=batch.audios,
    descriptions=batch.descriptions,
    sizes=batch.sizes,
)
```

### Long Audio (> 30s)

Use `separate_long()` with euler method:

```python
# Good quality, faster than realtime on M-series Macs
result = model.separate_long(
    batch.audios,
    batch.descriptions,
    chunk_seconds=10.0,
    overlap_seconds=3.0,
    ode_opt={"method": "euler", "step_size": 2/64},  # 32 steps
)
```

### Very Long Audio (> 5 min) or Limited Memory

Use smaller chunks:

```python
result = model.separate_long(
    batch.audios,
    batch.descriptions,
    chunk_seconds=5.0,       # Smaller chunks
    overlap_seconds=1.5,     # 30% overlap
    ode_opt={"method": "euler", "step_size": 2/32},
)
```

### Maximum Quality (Studio)

Use 32 midpoint steps (4x slower than euler/16):

```python
result = model.separate_long(
    batch.audios,
    batch.descriptions,
    ode_opt={"method": "midpoint", "step_size": 2/64},  # 32 midpoint steps
)
```

## Performance Benchmarks

Tested on Apple M-series with float16:

| Audio Length | Method | Settings | Time | Realtime Factor |
|--------------|--------|----------|------|-----------------|
| 12s | `separate` | midpoint/16 | 18s | 0.7x |
| 12s | `separate_long` | euler/16 | 12s | 1.0x |
| 2 min | `separate_long` | euler/16 | ~100s | 1.2x |
| 2 min | `separate_long` | euler/32 | ~180s | 0.7x |
| 2 min | `separate_long` | midpoint/32 | ~360s | 0.3x |

## Tips

### Reducing Background Music/Noise

1. Use more ODE steps: `step_size: 2/64` instead of `2/32`
2. Use midpoint method for cleaner separation
3. For maximum quality use 32 midpoint steps: `{"method": "midpoint", "step_size": 2/64}`
4. Be specific in your text prompt: "A man speaking clearly" vs "speech"

### Smoother Chunk Transitions

1. Increase overlap: `overlap_seconds=3.0` or higher
2. Use longer chunks if memory allows: `chunk_seconds=15.0`

### Memory Management

The model automatically:
- Clears GPU cache between chunks
- Uses `wired_limit` context for optimal Metal memory

For very large files, reduce chunk size to 5s.

### Reproducibility

Use the `seed` parameter for reproducible results:

```python
result = model.separate_long(..., seed=42)
```

## Output Format

Both methods return a `SeparationResult`:

```python
result.target    # List[mx.array] - Separated target audio
result.residual  # List[mx.array] - Background/residual audio
result.noise     # mx.array - Initial noise (for reproducibility)
```

Save outputs:

```python
from mlx_audio.sts import save_audio

save_audio(result.target[0], "target.wav", sample_rate=model.sample_rate)
save_audio(result.residual[0], "residual.wav", sample_rate=model.sample_rate)
```

## Streaming Separation

For real-time streaming output, use `separate_streaming()` which processes audio in chunks and streams output immediately. **First audio arrives in ~10-15 seconds** instead of waiting for the entire file to process.

### Generator Mode (Recommended)

Returns `SeparationResult` objects with streaming-specific fields:

```python
from mlx_audio.audio_io import write as audio_write
import numpy as np

target_chunks = []
residual_chunks = []

for result in model.separate_streaming(
    audios=["audio.mp3"],         # File paths or mx.array
    descriptions=["speech"],
    chunk_seconds=10.0,
    overlap_seconds=3.0,
    verbose=True,
):
    target_chunks.append(np.array(result.target[:, 0]))
    residual_chunks.append(np.array(result.residual[:, 0]))
    print(f"Chunk {result.chunk_idx} processed, is_last={result.is_last}")

    # On last chunk, access metadata
    if result.is_last:
        print(f"Peak memory: {result.peak_memory:.2f} GB")
        print(f"Noise shape: {result.noise.shape}")

audio_write('target.wav', np.concatenate(target_chunks), 48000)
audio_write('residual.wav', np.concatenate(residual_chunks), 48000)
```

### Callback Mode

```python
from mlx_audio.audio_io import write as audio_write
import numpy as np

target_chunks = []
residual_chunks = []

def collect_target(chunk, idx, is_last):
    target_chunks.append(np.array(chunk[:, 0]))

def collect_residual(chunk, idx, is_last):
    residual_chunks.append(np.array(chunk[:, 0]))

samples = model.separate_streaming(
    audios=["audio.mp3"],         # File paths or mx.array
    descriptions=["speech"],
    target_callback=collect_target,
    residual_callback=collect_residual,
    chunk_seconds=10.0,
    verbose=True,
)
print(f"Processed {samples} samples")

audio_write('target.wav', np.concatenate(target_chunks), 48000)
audio_write('residual.wav', np.concatenate(residual_chunks), 48000)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_seconds` | 10.0 | Size of each audio chunk in seconds |
| `overlap_seconds` | 3.0 | Overlap between chunks for smooth crossfade |
| `seed` | 42 | Random seed for reproducibility |
| `verbose` | False | Print progress during processing |

### SeparationResult Fields (Streaming Mode)

In streaming mode, `SeparationResult` includes these fields:

| Field | Type | Description |
|-------|------|-------------|
| `target` | `mx.array` | Target audio chunk (samples, 1) |
| `residual` | `mx.array` | Residual audio chunk (samples, 1) |
| `chunk_idx` | `int` | Index of this chunk |
| `is_last` | `bool` | True if this is the final chunk |
| `peak_memory` | `float` | Peak memory in GB (only on last chunk) |
| `noise` | `mx.array` | Accumulated noise for reproducibility (only on last chunk) |

### Performance Comparison

| Method | Time to First Audio | Use Case |
|--------|---------------------|----------|
| `separate()` | Full processing time | Short audio (< 30s) |
| `separate_long()` | Full processing time | Long audio, accumulated |
| `separate_streaming()` | **~10-15 seconds** | Real-time streaming |

**Note:** Call `flush()` after each write to ensure data is written to disk immediately.

### Codec-level Streaming

For direct codec access (advanced use):

```python
# Stream decode encoded features
for chunk, is_last in model.audio_codec.decode_streaming(
    encoded_features,
    chunk_size=50,
    overlap=4,
):
    process(chunk)
```

## Model Weights

SAM-Audio models are gated on HuggingFace. Request access at:
https://huggingface.co/facebook/sam-audio-large

