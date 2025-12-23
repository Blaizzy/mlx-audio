# SAM-Audio MLX

MLX implementation of [SAM-Audio](https://github.com/facebookresearch/sam-audio) (Segment Anything Model for Audio) - a foundation model for audio source separation using text prompts.

## Installation

```bash
pip install mlx-audio
```

## Quick Start

```python
from mlx_audio.ss import SAMAudio, SAMAudioProcessor, save_audio

# Load model and processor
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
model = SAMAudio.from_pretrained("facebook/sam-audio-large")

# Process inputs
batch = processor(
    descriptions=["A person speaking"],
    audios=["path/to/audio.wav"],
)

# Separate audio
result = model.separate(
    audios=batch.audios,
    descriptions=batch.descriptions,
    sizes=batch.sizes,
)

# Save output
save_audio(result.target[0], "separated.wav", sample_rate=model.sample_rate)
```

## Methods

### `separate()` - Standard Separation

Best for short audio files (< 30 seconds) or when you have enough memory.

```python
result = model.separate(
    audios=batch.audios,           # (B, 1, T) audio tensor
    descriptions=batch.descriptions, # List of text prompts
    sizes=batch.sizes,             # Optional: sequence lengths
    ode_opt=None,                  # ODE solver options (see below)
)
```

### `separate_long()` - Chunked Processing

Best for long audio files or limited memory. Processes audio in chunks with crossfade blending.

```python
result = model.separate_long(
    audios=batch.audios,
    descriptions=batch.descriptions,
    chunk_seconds=10.0,      # Chunk size (default: 10s)
    overlap_seconds=3.0,     # Overlap for crossfade (default: 3s)
    ode_opt=None,            # ODE solver options
    seed=42,                 # Random seed for reproducibility
    verbose=True,            # Print progress
)
```

## ODE Solver Options

The separation quality vs speed tradeoff is controlled by `ode_opt`:

| Method | Steps | Speed | Quality | Use Case |
|--------|-------|-------|---------|----------|
| `midpoint` | 32 | 0.5x | Maximum | Studio quality, no artifacts |
| `midpoint` | 16 | 1x | Best | Short audio, quality priority |
| `euler` | 32 | ~1.5x | Very Good | Long audio, balanced |
| `euler` | 16 | ~2x | Good | Real-time, speed priority |

### Configuration Examples

```python
# Maximum Quality - 32 midpoint steps (slowest, cleanest)
ode_opt = {"method": "midpoint", "step_size": 2/64}  # 32 midpoint steps

# Best Quality - 16 midpoint steps (default for separate())
ode_opt = {"method": "midpoint", "step_size": 2/32}  # 16 midpoint steps

# Balanced - Recommended for separate_long()
ode_opt = {"method": "euler", "step_size": 2/64}     # 32 euler steps

# Fastest - May have artifacts on complex audio
ode_opt = {"method": "euler", "step_size": 2/32}     # 16 euler steps
```

## Inference Recommendations

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
from mlx_audio.ss import save_audio

save_audio(result.target[0], "target.wav", sample_rate=model.sample_rate)
save_audio(result.residual[0], "residual.wav", sample_rate=model.sample_rate)
```

## Model Weights

SAM-Audio models are gated on HuggingFace. Request access at:
https://huggingface.co/facebook/sam-audio-large

