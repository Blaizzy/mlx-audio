# Architecture

This page provides a high-level overview of the MLX Audio codebase, its module structure, and the key design patterns.

## Module Layout

```
mlx_audio/
├── tts/                # Text-to-Speech
│   ├── models/         # Model implementations
│   │   ├── base.py     # BaseModelArgs, GenerationResult, BatchGenerationResult
│   │   ├── kokoro/     # Kokoro TTS
│   │   ├── qwen3_tts/  # Qwen3-TTS (Base, CustomVoice, VoiceDesign)
│   │   ├── sesame/     # CSM (Conversational Speech Model)
│   │   ├── dia/        # Dia TTS
│   │   ├── chatterbox/ # Chatterbox TTS
│   │   ├── spark/      # Spark TTS
│   │   ├── outetts/    # OuteTTS
│   │   ├── soprano/    # Soprano TTS
│   │   ├── bailingmm/  # Ming Omni TTS (BailingMM)
│   │   ├── dense/      # Ming Omni TTS (Dense)
│   │   ├── voxtral_tts/# Voxtral TTS
│   │   └── ...         # Other model implementations
│   ├── utils.py        # load(), load_model(), model discovery
│   ├── generate.py     # generate_audio() CLI entry point
│   └── audio_player.py # Real-time audio playback
│
├── stt/                # Speech-to-Text
│   ├── models/         # Model implementations
│   │   ├── whisper/    # OpenAI Whisper
│   │   ├── parakeet/   # NVIDIA Parakeet
│   │   ├── voxtral/    # Voxtral STT
│   │   ├── voxtral_realtime/ # Voxtral Realtime
│   │   ├── qwen3_asr/  # Qwen3-ASR & ForcedAligner
│   │   ├── vibevoice_asr/ # VibeVoice-ASR
│   │   ├── canary/     # NVIDIA Canary
│   │   ├── moonshine/  # Moonshine
│   │   ├── mms/        # Meta MMS
│   │   └── ...
│   ├── utils.py        # load(), load_model(), load_audio()
│   └── generate.py     # CLI for transcription
│
├── sts/                # Speech-to-Speech
│   ├── models/         # SAM-Audio, MossFormer2, DeepFilterNet, Liquid
│   ├── generate.py
│   └── voice_pipeline.py
│
├── vad/                # Voice Activity Detection
│   ├── models/         # Sortformer v1 & v2.1
│   └── utils.py
│
├── audio_io.py         # Audio file reading/writing (miniaudio + ffmpeg)
├── server.py           # FastAPI API server
├── convert.py          # Model conversion and quantization
└── ui/                 # Next.js Studio UI
```

## Core Design Patterns

### Model Loading Pipeline

All models (TTS, STT, STS) follow the same loading pattern:

```
User calls load("mlx-community/model-name")
    │
    ▼
Download from HuggingFace (or resolve local path)
    │
    ▼
Read config.json → determine model_type
    │
    ▼
Look up model_type in MODEL_REMAPPING dict
    │
    ▼
Dynamically import mlx_audio.{category}.models.{model_type}
    │
    ▼
Instantiate ModelConfig from config dict
    │
    ▼
Instantiate Model(config) as mlx.nn.Module
    │
    ▼
Load .safetensors weights into the model
    │
    ▼
Return initialized model
```

The shared loading logic lives in `mlx_audio/utils.py` (`base_load_model`, `get_model_class`, `get_model_path`, `load_config`). Each category (`tts`, `stt`) has a thin wrapper in its own `utils.py` that passes the category-specific `MODEL_REMAPPING`.

### Generation Pattern (TTS)

TTS models implement `generate()` as a **generator function** that yields `GenerationResult` objects:

```python
for result in model.generate(text="Hello!", voice="af_heart"):
    # result.audio is an mx.array waveform
    # result.sample_rate, result.audio_duration, etc.
```

This design supports both:

- **Non-streaming:** The generator yields one `GenerationResult` per text segment.
- **Streaming:** With `stream=True`, the generator yields multiple smaller chunks with `is_streaming_chunk=True`, and a final chunk with `is_final_chunk=True`.

The `generate_audio()` function in `mlx_audio/tts/generate.py` wraps this pattern with CLI argument handling, audio file writing, and optional playback.

### Generation Pattern (STT)

STT models expose a `generate()` method that returns a result object with a `.text` attribute:

```python
result = model.generate("audio.wav")
print(result.text)
```

Some models support streaming, where `generate()` returns an iterator of chunks instead.

### Base Classes

**`BaseModelArgs`** (`mlx_audio/tts/models/base.py`)

A dataclass base with a `from_dict()` class method that filters unknown keys. This allows `config.json` files to contain extra fields without causing errors:

```python
@dataclass
class MyModelConfig(BaseModelArgs):
    hidden_size: int = 512
```

**`GenerationResult`**

Returned by TTS `generate()`. Key fields:

- `audio` -- `mx.array` waveform
- `sample_rate` -- int, Hz
- `token_count` -- number of tokens generated
- `audio_duration` -- human-readable string
- `real_time_factor` -- speed relative to real time
- `processing_time_seconds` -- wall-clock time
- `peak_memory_usage` -- GB
- `is_streaming_chunk` / `is_final_chunk` -- streaming state

**`BatchGenerationResult`**

Similar to `GenerationResult` but includes `sequence_idx` for batched generation.

### Audio I/O

The `mlx_audio/audio_io.py` module provides `read()` and `write()` functions:

- **Reading:** Uses miniaudio for WAV/MP3/FLAC; falls back to ffmpeg for M4A/AAC/OGG/Opus.
- **Writing:** Uses miniaudio for WAV; uses ffmpeg for all other formats.
- Handles format detection from file extension or magic bytes.
- Provides `sf_read()` and `sf_write()` as drop-in replacements for the `soundfile` API.

### API Server

`mlx_audio/server.py` is a FastAPI application that exposes:

- `POST /v1/audio/speech` -- OpenAI-compatible TTS
- `POST /v1/audio/transcriptions` -- OpenAI-compatible STT
- `POST /v1/audio/separations` -- Audio source separation
- `GET /v1/models` -- List loaded models
- `POST /v1/models` -- Load a model
- `DELETE /v1/models` -- Unload a model
- `WS /v1/audio/transcriptions/realtime` -- WebSocket real-time transcription

The `ModelProvider` class manages loaded models with lazy loading and an async lock for thread safety.

### Model Conversion

`mlx_audio/convert.py` handles:

1. Downloading weights from HuggingFace
2. Converting PyTorch weights to MLX `.safetensors`
3. Applying quantization (affine, mxfp4, mxfp8, nvfp4)
4. Uploading converted models back to HuggingFace Hub

## Category-Specific Notes

### TTS (`mlx_audio/tts/`)

- Models range from small (Kokoro at 82M parameters) to large (Ming Omni at 16.8B).
- Some models have multiple generation methods (e.g., Qwen3-TTS: `generate()`, `generate_custom_voice()`, `generate_voice_design()`).
- Voice cloning is handled through `ref_audio` and `ref_text` parameters passed to `generate()`.
- The `AudioPlayer` class in `audio_player.py` handles real-time audio playback during streaming.

### STT (`mlx_audio/stt/`)

- Models return structured results with `.text`, and optionally `.segments`, `.sentences`, or timing information.
- The CLI (`generate.py`) supports multiple output formats: `txt`, `srt`, `vtt`, `json`.

### STS (`mlx_audio/sts/`)

- Includes source separation (SAM-Audio), speech enhancement (MossFormer2, DeepFilterNet), and speech-to-speech (Liquid).
- The `voice_pipeline.py` module provides higher-level voice processing workflows.

### VAD (`mlx_audio/vad/`)

- Contains Sortformer models for speaker diarization.
- Supports both offline and streaming diarization modes.
