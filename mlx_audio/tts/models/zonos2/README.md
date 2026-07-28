# ZONOS2

ZONOS2 is Zyphra's autoregressive text-to-speech model with multi-codebook audio
generation, 44.1 kHz DAC decode, and speaker conditioning from reference audio.

**Original model:** [Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2)

## Supported Models

| Repository | Format | Notes |
|------------|--------|-------|
| `mlx-community/Zyphra-ZONOS2` | BF16 | Official MLX conversion with bundled speaker encoder |


## Usage

Python API:

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load

model = load("mlx-community/Zyphra-ZONOS2", lazy=True)

result = next(model.generate(
    text="Hello, this is ZONOS two running locally with MLX audio.",
    max_tokens=1024,
))

audio_write("zonos2.wav", result.audio, result.sample_rate)
```

## Voice Cloning

Pass a short reference clip with `ref_audio`. Clean speech-only clips work best.

```python
result = next(model.generate(
    text="This text will be spoken with the reference speaker.",
    ref_audio="speaker.wav",
    max_tokens=1024,
))

audio_write("zonos2_clone.wav", result.audio, result.sample_rate)
```

You can also compute the speaker embedding once and reuse it from the Python
API:

```python
speaker = model.extract_speaker_embedding("speaker.wav")

result = next(model.generate(
    text="This reuses a precomputed speaker embedding.",
    speaker_embedding=speaker,
    max_tokens=1024,
))
```

## Batch Generation

Use `batch_generate` from Python for non-streaming batches. A shared
`ref_audio` or `speaker_embedding` applies to every row; use `ref_audios` or
`speaker_embeddings` for per-row speaker conditioning.

```python
texts = [
    "This is the first ZONOS two batch sample.",
    "This is the second sample generated in the same batch.",
]

for result in model.batch_generate(texts, max_tokens=1024):
    audio_write(
        f"zonos2_batch_{result.sequence_idx}.wav",
        result.audio,
        result.sample_rate,
    )
```

Batch streaming is not implemented yet; use `generate(..., stream=True)` for
single-sequence streaming.

## CLI

Text-to-speech:

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Zyphra-ZONOS2 \
  --text "Hello, this is ZONOS two running with MLX audio." \
  --output_path outputs \
  --file_prefix zonos2
```

Voice cloning:

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Zyphra-ZONOS2 \
  --text "This text will use the voice from the reference clip." \
  --ref_audio speaker.wav \
  --output_path outputs \
  --file_prefix zonos2_clone
```

Streaming playback:

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Zyphra-ZONOS2 \
  --text "This streams ZONOS two audio chunks as they are generated." \
  --stream \
  --streaming_interval 0.5
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ref_audio` | `None` | Reference audio path or array for voice cloning |
| `ref_audios` | `None` | Per-row reference audio list for batch generation, Python API only |
| `speaker_embedding` | `None` | Precomputed 2048-D speaker embedding, Python API only |
| `speaker_embeddings` | `None` | Per-row 2048-D speaker embeddings for batch generation, Python API only |
| `max_tokens` | 1024 | Maximum number of audio token frames |
| `temperature` | 1.15 | Sampling temperature |
| `top_k` | 106 | Top-k sampling filter |
| `top_p` | 0.0 | Nucleus sampling filter, disabled at 0 |
| `min_p` | 0.18 | Minimum-probability sampling filter |
| `repetition_penalty` | 1.2 | Repetition penalty applied to recent audio tokens |
| `seed` | `None` | Seed for deterministic sampling |
| `stream` | `False` | Yield audio chunks during generation |
| `streaming_interval` | 2.0 | Approximate seconds of audio per streaming chunk |
| `text_normalization` | `True` | English text normalization toggle, Python API only |

The CLI exposes common generation controls. Model-specific conditioning options
such as `speaker_embedding`, `speaking_rate_bucket`, `quality_buckets`,
`accurate_mode`, and `text_normalization` are available through the Python API.

## Notes

- Output audio is mono 44.1 kHz.
- Reference-audio speaker extraction uses the bundled
  `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B` speaker encoder.
- Audio decode uses `mlx-community/descript-audio-codec-44khz`.
- English text normalization handles common written forms; unsupported languages
  fall back to raw UTF-8 byte prompting.
- Streaming decodes completed delayed-codebook frames as they become available;
  final audio is emitted in the last chunk.
- Batch generation currently supports non-streaming Python API usage.

## Architecture

- 28-layer, 2048-hidden transformer backbone with grouped-query attention
- Sonic-style MoE feed-forward layers with 16 experts
- 9 generated audio codebooks plus a text token column
- Speaker conditioning through a projected 2048-D speaker embedding
- Descript Audio Codec decode at 44.1 kHz

## License

See the [Zyphra/ZONOS2 model card](https://huggingface.co/Zyphra/ZONOS2) for
upstream license and usage details.
