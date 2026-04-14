# STT API Reference

## Model Loading

### `mlx_audio.stt.utils`

Example:

```python
from mlx_audio.stt import load

model = load("mlx-community/whisper-tiny-asr-fp16")
result = model.generate(audio)
```

::: mlx_audio.stt.utils.load
    options:
      show_source: false
      heading_level: 4

::: mlx_audio.stt.utils.load_model
    options:
      show_source: false
      heading_level: 4

::: mlx_audio.stt.utils.load_audio
    options:
      show_source: false
      heading_level: 4

::: mlx_audio.stt.utils.resample_audio
    options:
      show_source: false
      heading_level: 4

## Transcription CLI

The `mlx_audio.stt.generate` module provides a command-line interface for transcription:

```bash
python -m mlx_audio.stt.generate \
    --model mlx-community/whisper-large-v3-turbo-asr-fp16 \
    --audio speech.wav \
    --output-path output \
    --format json \
    --verbose
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `whisper-large-v3-turbo` | Model path or HuggingFace repo |
| `--audio` | required | Path to the audio file |
| `--output-path` | required | Directory to save output |
| `--format` | `txt` | Output format: `txt`, `srt`, `vtt`, `json` |
| `--language` | `en` | Language code |
| `--max-tokens` | `8192` | Maximum output tokens |
| `--chunk-duration` | `30.0` | Chunk duration in seconds |
| `--stream` | `false` | Stream transcription output |
| `--context` | `null` | Hotwords or metadata string |
| `--prefill-step-size` | `2048` | Prefill step size |
| `--verbose` | `false` | Print detailed output |
