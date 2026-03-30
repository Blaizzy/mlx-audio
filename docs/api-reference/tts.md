# TTS API Reference

## Model Loading

The primary entry points for loading TTS models.

### `mlx_audio.tts.utils`

Example:

```python
from mlx_audio.tts import load

model = load("mlx-community/outetts-0.3-500M-bf16")
audio = model.generate("Hello world!")
```

::: mlx_audio.tts.utils.load
    options:
      show_source: false
      heading_level: 4

::: mlx_audio.tts.utils.load_model
    options:
      show_source: false
      heading_level: 4

::: mlx_audio.tts.utils.get_available_models
    options:
      show_source: false
      heading_level: 4

::: mlx_audio.tts.utils.get_model_and_args
    options:
      show_source: false
      heading_level: 4

::: mlx_audio.tts.utils.fetch_from_hub
    options:
      show_source: false
      heading_level: 4

::: mlx_audio.tts.utils.upload_to_hub
    options:
      show_source: false
      heading_level: 4

## Audio Generation

### `mlx_audio.tts.generate`

::: mlx_audio.tts.generate.generate_audio
    options:
      show_source: false
      heading_level: 4

## Data Classes

### `mlx_audio.tts.models.base`

::: mlx_audio.tts.models.base.GenerationResult
    options:
      show_source: false
      heading_level: 4
      members:
        - audio
        - samples
        - sample_rate
        - segment_idx
        - token_count
        - audio_duration
        - real_time_factor
        - processing_time_seconds
        - peak_memory_usage
        - is_streaming_chunk
        - is_final_chunk

::: mlx_audio.tts.models.base.BatchGenerationResult
    options:
      show_source: false
      heading_level: 4

::: mlx_audio.tts.models.base.BaseModelArgs
    options:
      show_source: false
      heading_level: 4
