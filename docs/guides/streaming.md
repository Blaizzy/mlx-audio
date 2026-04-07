# Streaming Audio

MLX Audio supports streaming audio generation for low-latency playback. Instead of waiting for the entire utterance to be synthesized, you can start playing audio chunks as they are produced.

## CLI Streaming

Add the `--stream` flag to any TTS generation command. When streaming is enabled, audio is played back in real time as chunks become available:

```bash
mlx_audio.tts.generate \
    --model mlx-community/Kokoro-82M-bf16 \
    --text "Hello, this is a streaming example!" \
    --lang_code a \
    --stream
```

!!! tip "Streaming implies playback"
    The `--stream` flag automatically enables `--play`, so there is no need to pass both.

### Controlling Chunk Size

The `--streaming_interval` argument controls how frequently audio chunks are emitted (in seconds). Smaller values reduce latency but add per-chunk overhead:

```bash
mlx_audio.tts.generate \
    --model mlx-community/Kokoro-82M-bf16 \
    --text "Adjusting the streaming interval changes latency." \
    --lang_code a \
    --stream \
    --streaming_interval 1.5
```

The default interval is **2.0 seconds**.

## Python Streaming

### TTS Streaming

Every model's `generate()` method accepts `stream=True`. When enabled, it yields `GenerationResult` objects as chunks rather than waiting for full synthesis:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Kokoro-82M-bf16")

for result in model.generate(
    text="This audio will stream chunk by chunk.",
    voice="af_heart",
    lang_code="a",
    stream=True,
    streaming_interval=2.0,
):
    # result.audio is an mx.array with one chunk of audio
    print(f"Chunk: {result.audio.shape[0]} samples")
    # Feed result.audio to an audio player or buffer
```

Each `GenerationResult` chunk contains:

| Attribute | Description |
|-----------|-------------|
| `audio` | `mx.array` waveform for this chunk |
| `sample_rate` | Sample rate in Hz |
| `is_streaming_chunk` | `True` for intermediate chunks |
| `is_final_chunk` | `True` for the last chunk |

### Qwen3-TTS Streaming

Qwen3-TTS models support streaming across all generation methods -- `generate()`, `generate_custom_voice()`, and `generate_voice_design()`:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit")

audio_chunks = []
for result in model.generate(
    text="Hello, how are you today?",
    voice="serena",
    stream=True,
    streaming_interval=0.32,  # ~4 tokens at 12.5 Hz
):
    audio_chunks.append(result.audio)
    # Play or process each chunk for low-latency output
```

!!! info "Streaming interval for Qwen3-TTS"
    At 12.5 Hz token rate, a `streaming_interval` of **0.32** seconds corresponds to roughly 4 tokens per chunk. Lower values reduce latency but increase overhead.

### STT Streaming

Several speech-to-text models support streaming transcription. Pass `stream=True` to `generate()`:

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

=== "VibeVoice-ASR"

    ```python
    from mlx_audio.stt.utils import load

    model = load("mlx-community/VibeVoice-ASR-bf16")

    for text in model.stream_transcribe(audio="speech.wav", max_tokens=4096):
        print(text, end="", flush=True)
    ```

### STT Streaming via CLI

```bash
python -m mlx_audio.stt.generate \
    --model mlx-community/whisper-large-v3-turbo-asr-fp16 \
    --audio speech.wav \
    --output-path output \
    --format json \
    --stream
```

## API Server Streaming

The API server supports streaming for both TTS and STT. Set `"stream": true` in your request:

### Streaming TTS

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Kokoro-82M-bf16",
    "input": "Streaming over HTTP!",
    "voice": "af_heart",
    "stream": true,
    "streaming_interval": 2.0,
    "response_format": "wav"
  }' \
  --output streamed_speech.wav
```

### Streaming STT

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo-asr-fp16" \
  -F "stream=true"
```

Streaming STT returns newline-delimited JSON (`application/x-ndjson`), with each line containing a `text` field and optional timing information.

## Real-Time WebSocket Transcription

The API server also exposes a WebSocket endpoint for real-time audio transcription:

```
ws://localhost:8000/v1/audio/transcriptions/realtime
```

This is useful for live microphone input or continuous audio streams.

## Tips

- **Latency vs. quality** -- Shorter `streaming_interval` values give lower latency but may produce more chunk boundaries. Start with the default and decrease as needed.
- **Memory** -- Streaming does not significantly change peak memory usage since model weights remain loaded; it only changes _when_ decoded audio is returned.
- **Joining chunks** -- If you need a single contiguous file, concatenate the `result.audio` arrays after the loop and write once with `mlx_audio.audio_io.write()`.
