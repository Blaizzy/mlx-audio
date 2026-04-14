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
| `--workers` | `2` | Number of worker processes (see below) |
| `--reload` | `false` | Auto-reload on code changes (development) |
| `--start-ui` | `false` | Start the Studio UI alongside the API |
| `--allowed-origins` | `*` | CORS allowed origins (space-separated) |
| `--log-dir` | `logs` | Directory for server logs |

### Worker Configuration

The `--workers` flag accepts either an integer or a float:

```bash
# Fixed number of workers
mlx_audio.server --workers 4

# Fraction of CPU cores (0.5 = half)
mlx_audio.server --workers 0.5

# All CPU cores
mlx_audio.server --workers 1.0
```

You can also set the `MLX_AUDIO_NUM_WORKERS` environment variable.

### CORS Configuration

By default, all origins are allowed. To restrict origins:

```bash
mlx_audio.server --allowed-origins http://localhost:3000 https://myapp.example.com
```

Or set the `MLX_AUDIO_ALLOWED_ORIGINS` environment variable with a comma-separated list.

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

The server exposes a WebSocket endpoint for live audio transcription:

```
ws://localhost:8000/v1/audio/transcriptions/realtime
```

Connect from a browser or any WebSocket client and send raw audio frames for continuous transcription.

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
