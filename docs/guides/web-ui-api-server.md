# Web UI & API Server

MLX Audio ships with a FastAPI-based API server and a Next.js web interface (Studio UI) for interactive audio generation and transcription.

## Starting the Server

### API Server Only

```bash
mlx_audio.server --host 0.0.0.0 --port 8000
```

### API Server with Studio UI

Pass `--start-ui` to launch the Next.js web interface alongside the API server:

```bash
mlx_audio.server --host 0.0.0.0 --port 8000 --start-ui
```

The API will be available at `http://localhost:8000` and the Studio UI at `http://localhost:3000`.

### Server Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | Host to bind the server to |
| `--port` | `8000` | Port for the API server |
| `--reload` | `false` | Auto-reload on code changes (development) |
| `--start-ui` | `false` | Start the Studio UI alongside the API |
| `--allowed-origins` | `*` | CORS allowed origins (space-separated) |
| `--log-dir` | `logs` | Directory for server logs |
| `--realtime-model` | `null` | Default model for `/v1/realtime` when the client omits `?model=` |
| `--realtime-transcription-delay-ms` | `null` | Transcription latency/quality knob for models that support it (e.g. `voxtral_realtime`) |
| `--tts-max-batch-size` | `8` | Maximum compatible TTS speech requests per continuous batch session |

The two realtime flags also read from `MLX_AUDIO_REALTIME_MODEL` and `MLX_AUDIO_REALTIME_TRANSCRIPTION_DELAY_MS` if present; the CLI flags take precedence.
The TTS batching flag also reads from `MLX_AUDIO_TTS_MAX_BATCH_SIZE`; the CLI flag takes precedence.

### CORS Configuration

By default, all origins are allowed. To restrict origins:

```bash
mlx_audio.server --allowed-origins http://localhost:3000 https://myapp.example.com
```

Or set the `MLX_AUDIO_ALLOWED_ORIGINS` environment variable with a comma-separated list.

### TTS Continuous Batching

Compatible non-streaming TTS `/v1/audio/speech` requests are routed through the server's continuous batching path when the model exposes a `create_tts_batch_session(...)` hook.

Requests that are not supported by the model's continuous batching hook continue through the existing serial or fixed-window batch paths.

```bash
mlx_audio.server --host 127.0.0.1 --port 8000 \
  --tts-max-batch-size 8
```

## OpenAI-Compatible API

The server implements OpenAI-compatible endpoints, so existing code that targets the OpenAI audio API can point to MLX Audio with minimal changes.

### Text-to-Speech

**`POST /v1/audio/speech`**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Kokoro-82M-bf16",
    "input": "Hello, world!",
    "voice": "af_heart",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | Model ID (HuggingFace repo or local path) |
| `input` | string | required | Text to synthesize |
| `voice` | string | `null` | Voice preset name |
| `speed` | float | `1.0` | Playback speed multiplier |
| `lang_code` | string | `"a"` | Language code |
| `response_format` | string | `"mp3"` | Output format: `mp3`, `wav`, `flac`, `ogg`, `opus` |
| `stream` | bool | `false` | Stream audio chunks |
| `streaming_interval` | float | `2.0` | Seconds between stream chunks |
| `temperature` | float | `0.7` | Sampling temperature |
| `max_tokens` | int | `1200` | Maximum generation tokens |
| `ref_audio` | string | `null` | Path to reference audio for voice cloning |
| `ref_text` | string | `null` | Transcript of reference audio |
| `instruct` | string | `null` | Style/emotion instruction |

#### Streaming TTS

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Kokoro-82M-bf16",
    "input": "Streaming audio over HTTP.",
    "voice": "af_heart",
    "stream": true,
    "response_format": "wav"
  }' \
  --output streamed.wav
```

### Speech-to-Text

**`POST /v1/audio/transcriptions`**

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo-asr-fp16"
```

**Form fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Audio file to transcribe |
| `model` | string | required | STT model ID |
| `language` | string | `null` | Language code |
| `max_tokens` | int | `1024` | Maximum output tokens |
| `stream` | bool | `false` | Stream results as NDJSON |
| `context` | string | `null` | Hotwords or metadata to guide transcription |
| `verbose` | bool | `false` | Include extra details |

### Audio Source Separation

**`POST /v1/audio/separations`**

```bash
curl -X POST http://localhost:8000/v1/audio/separations \
  -F "file=@mixed.wav" \
  -F "model=mlx-community/sam-audio-large-fp16" \
  -F "description=speech"
```

Returns JSON with base64-encoded `target` and `residual` WAV buffers.

### Model Management

```bash
# List loaded models
curl http://localhost:8000/v1/models

# Load a model
curl -X POST "http://localhost:8000/v1/models?model_name=mlx-community/Kokoro-82M-bf16"

# Unload a model
curl -X DELETE "http://localhost:8000/v1/models?model_name=mlx-community/Kokoro-82M-bf16"
```

### Real-Time WebSocket Transcription

The server exposes two WebSocket endpoints for live transcription. Both accept 16-bit signed little-endian PCM audio; they differ in wire protocol and intended consumers.

#### `/v1/audio/transcriptions/realtime` (VAD-based streaming)

```
ws://localhost:8000/v1/audio/transcriptions/realtime
```

Send raw PCM frames as binary WebSocket messages; the server performs VAD, chunks on silence, and emits transcription JSON messages back. Uses the preloaded Whisper-style STT model — connect after calling `POST /v1/models` to select it.

#### `/v1/realtime` (OpenAI Realtime-compatible)

```
ws://localhost:8000/v1/realtime?model=<model-id>
```

Implements a subset of the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) wire protocol, so existing Realtime clients can target MLX Audio by swapping the base URL. Any STT model that exposes `create_streaming_session` works — including `voxtral_realtime`, which is streaming-first and recommended.

**Model selection order** (first match wins):

1. `?model=<id>` query parameter on connect.
2. `session.update.model` (or `session.audio.input.transcription.model`) event after connect.
3. `--realtime-model` CLI flag / `MLX_AUDIO_REALTIME_MODEL` env var.

If none is set, the server replies with an `error` event and closes the socket.

**Client → server events** (subset of OpenAI Realtime):

| Type | Purpose |
|------|---------|
| `session.update` | Change the model, input sample rate, or transcription config |
| `input_audio_buffer.append` | Append a chunk of base64-encoded PCM16 to the current item |
| `input_audio_buffer.commit` | Signal end-of-utterance; the server drains deltas and emits `completed` |

**Declaring the input sample rate.** The server assumes incoming PCM is 24 kHz by default (the OpenAI Realtime client convention). If you are sending audio at a different rate — e.g. a 16 kHz microphone capture or a 48 kHz file — tell the server via `session.update` so it resamples correctly. Without this, 16 kHz audio interpreted as 24 kHz would sound sped-up to the model and transcribe as garbage.

```json
{
  "type": "session.update",
  "session": {
    "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}
  }
}
```

This only declares *your* input rate. The server resamples from that to whatever rate the model expects internally — you never need to match the model's native rate.

**Server → client events:**

| Type | Purpose |
|------|---------|
| `session.created` / `session.updated` | Session snapshot (id, model, input format) |
| `conversation.item.added` | New user item opened for the next audio chunk |
| `input_audio_buffer.committed` | Acknowledges a commit from the client |
| `conversation.item.input_audio_transcription.delta` | Incremental transcript token(s) |
| `conversation.item.input_audio_transcription.completed` | Final transcript for the item |
| `error` | Error message; the socket may close afterwards |

**Minimal Python client:**

```python
import asyncio, base64, json, websockets, numpy as np, soundfile as sf

async def transcribe(path: str):
    audio, sr = sf.read(path, dtype="int16", always_2d=False)
    uri = "ws://localhost:8000/v1/realtime?model=iris-sfg/Voxtral-Mini-4B-Realtime-2602-4bit"
    async with websockets.connect(uri) as ws:
        # Wait for session.created
        await ws.recv()
        # Declare the input rate (the server will resample to the model's rate)
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": sr}}}},
        }))
        # Stream audio in 100 ms chunks
        step = sr // 10
        for i in range(0, len(audio), step):
            chunk = audio[i : i + step].tobytes()
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode(),
            }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async for raw in ws:
            evt = json.loads(raw)
            if evt["type"].endswith(".delta"):
                print(evt["delta"], end="", flush=True)
            elif evt["type"].endswith(".completed"):
                print()
                break

asyncio.run(transcribe("audio.wav"))
```

**Tuning transcription delay** (for models like `voxtral_realtime` that expose the knob):

```bash
mlx_audio.server \
  --realtime-model iris-sfg/Voxtral-Mini-4B-Realtime-2602-4bit \
  --realtime-transcription-delay-ms 960
```

Lower values reduce latency at the cost of accuracy. Models that don't declare a `transcription_delay_ms` parameter silently ignore the flag.

!!! note "Single-process MLX serialization"
    MLX inference on a given device is serialized inside the server. `/v1/realtime` schedules `session.step()` calls cooperatively — two concurrent websocket clients share GPU time rather than running in parallel. This is fine as long as each stream transcribes faster than real-time; otherwise run multiple server instances behind a load balancer.

## Web Interface (Studio UI)

The Studio UI is a Next.js application located in `mlx_audio/ui/`. It provides:

- A text input for TTS generation with voice and model selectors
- Audio upload and recording for STT
- 3D audio visualization
- Model loading and management

### Running the UI Separately

If you prefer to run the UI outside the server process:

```bash
cd mlx_audio/ui
npm install
npm run dev
```

The UI runs on `http://localhost:3000` and expects the API server at `http://localhost:8000`.

## Using with the OpenAI Python Client

Because the API is OpenAI-compatible, you can use the official OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # MLX Audio does not require an API key
)

# Text-to-Speech
response = client.audio.speech.create(
    model="mlx-community/Kokoro-82M-bf16",
    voice="af_heart",
    input="Hello from the OpenAI client!",
)
response.stream_to_file("output.mp3")

# Speech-to-Text
with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="mlx-community/whisper-large-v3-turbo-asr-fp16",
        file=f,
    )
print(transcript.text)
```

## Installation

The server requires the `server` optional dependency group:

```bash
pip install "mlx-audio[server]"
```

This installs FastAPI, Uvicorn, and python-multipart.
