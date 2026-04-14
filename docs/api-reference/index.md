# API Reference

This section documents the public Python API for MLX Audio. The library is organized into several top-level modules:

| Module | Description |
|--------|-------------|
| [`mlx_audio.tts`](tts.md) | Text-to-Speech -- model loading, generation, and utilities |
| [`mlx_audio.stt`](stt.md) | Speech-to-Text -- model loading and transcription |
| [`mlx_audio.audio_io`](audio-io.md) | Audio I/O -- reading and writing audio files |

## Quick Links

### Loading Models

```python
# TTS
from mlx_audio.tts.utils import load, load_model

# STT
from mlx_audio.stt.utils import load, load_model
```

### Generating Audio

```python
from mlx_audio.tts.generate import generate_audio
```

### Reading and Writing Files

```python
from mlx_audio.audio_io import read, write
```

## Conventions

- All models are loaded from HuggingFace repos or local paths via `load()` / `load_model()`.
- TTS `generate()` methods return iterators of `GenerationResult` dataclass objects.
- Audio waveforms are represented as `mx.array` (MLX) or `np.ndarray` (NumPy) depending on the context.
- Sample rates are always in Hz and accompany the audio data in result objects.
