# Quick Start: Python

Use the mlx-audio Python API to generate speech, transcribe audio, and process audio programmatically.

## Text-to-Speech

!!! note
    These TTS quickstart examples use `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`.

### Basic Generation

```python
from mlx_audio.tts.utils import load_model

# Load a TTS model
model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")

# Generate speech
for result in model.generate(
    "Hello from MLX-Audio!",
    voice="Chelsie",
    lang_code="English",
):
    print(f"Generated {result.audio.shape[0]} samples")
    # result.audio contains the waveform as mx.array
```

### Voice and Language Options

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")

for result in model.generate(
    text="Welcome to MLX-Audio!",
    voice="Ethan",
    lang_code="English",
):
    audio = result.audio
```

!!! tip "Qwen3-TTS Base Model"
    Pass `voice` to pick a preset speaker and `lang_code` to hint the language, for example `voice="Chelsie"` and `lang_code="English"`.

### Voxtral TTS

```python
from mlx_audio.tts.utils import load

model = load("mlx-community/Voxtral-4B-TTS-2603-mlx-bf16")

for result in model.generate(text="Hello, how are you today?", voice="casual_male"):
    print(result.audio_duration)
```

### Qwen3-TTS

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
results = list(model.generate(
    text="Hello, welcome to MLX-Audio!",
    voice="Chelsie",
    lang_code="English",
))

audio = results[0].audio  # mx.array
```

## Speech-to-Text

### Whisper

```python
from mlx_audio.stt.generate import generate_transcription

result = generate_transcription(
    model="mlx-community/whisper-large-v3-turbo-asr-fp16",
    audio="audio.wav",
)
print(result.text)
```

### Qwen3-ASR

```python
from mlx_audio.stt import load

# Speech recognition
model = load("mlx-community/Qwen3-ASR-0.6B-8bit")
result = model.generate("audio.wav", language="English")
print(result.text)
```

### Word-Level Alignment (Qwen3-ForcedAligner)

```python
from mlx_audio.stt import load

aligner = load("mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
result = aligner.generate("audio.wav", text="I have a dream", language="English")

for item in result:
    print(f"[{item.start_time:.2f}s - {item.end_time:.2f}s] {item.text}")
```

### Parakeet (with Timestamps)

```python
from mlx_audio.stt.utils import load

model = load("mlx-community/parakeet-tdt-0.6b-v3")
result = model.generate("audio.wav")

print(f"Text: {result.text}")

# Word-level timestamps
for sentence in result.sentences:
    print(f"[{sentence.start:.2f}s - {sentence.end:.2f}s] {sentence.text}")
```

### Streaming Transcription

Several STT models support streaming for low-latency output:

=== "Voxtral Realtime"

    ```python
    from mlx_audio.stt.utils import load

    model = load("mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")

    for chunk in model.generate("audio.wav", stream=True):
        print(chunk, end="", flush=True)
    ```

=== "Parakeet"

    ```python
    from mlx_audio.stt.utils import load

    model = load("mlx-community/parakeet-tdt-0.6b-v3")

    for chunk in model.generate("long_audio.wav", stream=True):
        print(chunk.text, end="", flush=True)
    ```

=== "VibeVoice"

    ```python
    from mlx_audio.stt.utils import load

    model = load("mlx-community/VibeVoice-ASR-bf16")

    for text in model.stream_transcribe(audio="speech.wav", max_tokens=4096):
        print(text, end="", flush=True)
    ```

## Saving Audio

### Save with STS Utilities

```python
from mlx_audio.sts import save_audio

# Save an audio array to a WAV file
save_audio(audio_array, "output.wav", sample_rate=24000)
```

!!! note
    Saving to MP3, FLAC, OGG, Opus, or Vorbis requires [ffmpeg](installation.md#ffmpeg). WAV works without it.
