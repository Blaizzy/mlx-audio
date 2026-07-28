# MOSS-TTS

MOSS-TTS covers the 8B `MossTTSDelay` v1.5 and v1.0 models, the `MOSS-TTSD-v1.0` dialogue model, and both local-transformer variants from OpenMOSS. It uses a Qwen3 backbone with the matching multi-codebook audio generation path and the full MOSS Audio Tokenizer.

## Supported Models

- `OpenMOSS-Team/MOSS-TTS-v1.5`
- `OpenMOSS-Team/MOSS-TTS`
- `OpenMOSS-Team/MOSS-TTSD-v1.0`
- `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`
- `OpenMOSS-Team/MOSS-TTS-Local-Transformer`

## Usage

Python API:

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load

model = load("OpenMOSS-Team/MOSS-TTS-v1.5", lazy=True)

result = next(model.generate(
    text="Hello, this is MOSS-TTS running on MLX.",
    max_tokens=120,
))
audio_write("output.wav", result.audio, result.sample_rate)
```

Voice cloning:

```python
result = next(model.generate(
    text="This is a short cloned voice sample.",
    ref_audio="speaker.wav",
    max_tokens=120,
))
audio_write("cloned.wav", result.audio, result.sample_rate)
```

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-v1.5 \
  --text "Hello, this is MOSS-TTS running on MLX." \
  --output_path outputs
```

For non-English multilingual prompts on v1.5, pass `language` when known:

```python
result = next(model.generate(
    text="Bonjour, je voudrais essayer une voix francaise naturelle.",
    language="French",
    max_tokens=120,
))
```

Inline pause markers such as `[pause 3.2s]` are passed through to the v1.5 prompt.

For the v1.5 local-transformer checkpoint:

```python
model = load("OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5", lazy=True)
```

This variant uses `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`, 48 kHz output, and a fixed 12-codebook RVQ depth. Do not pass `n_vq_for_inference` with a value different from `12`.

Streaming is supported for `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`:

```python
chunks = model.generate(
    text="Hello, this is streamed MOSS local transformer audio.",
    language="English",
    stream=True,
    streaming_interval=2.0,
)

for chunk in chunks:
    play_or_buffer(chunk.audio, chunk.sample_rate)
```

The first streamed chunk decodes after 4 generated RVQ frames by default. Later
chunks use `streaming_interval` seconds of generated frames. The MLX path keeps a
single-stream MOSS-Audio-Tokenizer-v2 decoder session for each generation call;
it does not require the SGLang server used by the upstream streaming demo.

Dialogue generation with `MOSS-TTSD-v1.0`:

```bash
python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --text "[S1] Hello. [S2] Hi, this is MOSS-TTSD running on MLX." \
  --output_path outputs
```

Reference audio and text can be repeated for multiple speakers:

```bash
python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --text "[S1] This uses the first reference. [S2] This uses the second." \
  --ref_audio speaker_1.wav \
  --ref_text "Reference transcript for speaker one." \
  --ref_audio speaker_2.wav \
  --ref_text "Reference transcript for speaker two." \
  --output_path outputs
```
