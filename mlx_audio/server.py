"""Main module for MLX Audio API server.

This module provides a FastAPI-based server for hosting MLX Audio models,
including Text-to-Speech (TTS), Speech-to-Text (STT), and Speech-to-Speech (S2S) models.
It offers an OpenAI-compatible API for Audio completions and model management.
"""

import argparse
import asyncio
import base64
import inspect
import io
import json
import os
import subprocess
import time
import webbrowser
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import mlx.core as mx
import numpy as np
import uvicorn
import webrtcvad
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_audio.audio_io import read as audio_read
from mlx_audio.audio_io import write as audio_write
from mlx_audio.utils import load_model


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize NaN, Infinity, and -Infinity values for JSON serialization."""
    # Handle dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None
        return float(obj)
    else:
        return obj


MLX_AUDIO_NUM_WORKERS = os.getenv("MLX_AUDIO_NUM_WORKERS", "2")


class ModelProvider:
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    def load_model(self, model_name: str):
        if model_name not in self.models:
            self.models[model_name] = load_model(model_name)

        return self.models[model_name]

    async def remove_model(self, model_name: str) -> bool:
        async with self.lock:
            if model_name in self.models:
                del self.models[model_name]
                return True
            return False

    async def get_available_models(self):
        async with self.lock:
            return list(self.models.keys())


app = FastAPI()


def int_or_float(value):

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not an int or float")


def calculate_default_workers(workers: int = 2) -> int:
    if num_workers_env := os.getenv("MLX_AUDIO_NUM_WORKERS"):
        try:
            workers = int(num_workers_env)
        except ValueError:
            workers = max(1, int(os.cpu_count() * float(num_workers_env)))
    return workers


# Add CORS middleware
def setup_cors(app: FastAPI, allowed_origins: List[str]):
    """(Re)configure CORS middleware with the given origins."""
    # Remove any previously configured CORSMiddleware to avoid duplicates
    app.user_middleware = [
        m for m in app.user_middleware if m.cls is not CORSMiddleware
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Apply default CORS configuration when imported. The environment variable
# ``MLX_AUDIO_ALLOWED_ORIGINS`` can override the allowed origins by providing a
# comma-separated list. This ensures CORS headers are present even when running
# ``uvicorn mlx_audio.server:app`` directly.

allowed_origins_env = os.getenv("MLX_AUDIO_ALLOWED_ORIGINS")
default_origins = (
    [origin.strip() for origin in allowed_origins_env.split(",")]
    if allowed_origins_env
    else ["*"]
)

# Setup CORS
setup_cors(app, default_origins)


# Request schemas for OpenAI-compatible endpoints
class SpeechRequest(BaseModel):
    model: str
    input: str
    instruct: str | None = None
    voice: str | None = None
    speed: float | None = 1.0
    gender: str | None = "male"
    pitch: float | None = 1.0
    lang_code: str | None = "a"
    ref_audio: str | None = None
    ref_text: str | None = None
    temperature: float | None = 0.7
    top_p: float | None = 0.95
    top_k: int | None = 40
    repetition_penalty: float | None = 1.0
    response_format: str | None = "mp3"
    stream: bool = False
    streaming_interval: float = 2.0
    max_tokens: int = 1200
    verbose: bool = False


class TranscriptionRequest(BaseModel):
    model: str
    language: str | None = None
    verbose: bool = False
    max_tokens: int = 1024
    chunk_duration: float = 30.0
    frame_threshold: int = 25
    stream: bool = False
    context: str | None = None
    prefill_step_size: int = 2048
    text: str | None = None


class SeparationResponse(BaseModel):
    target: str  # Base64 encoded WAV
    residual: str  # Base64 encoded WAV
    sample_rate: int


# Initialize the ModelProvider
model_provider = ModelProvider()


@app.get("/")
async def root():
    return {
        "message": "Welcome to the MLX Audio API server! Visit https://localhost:3000 for the UI."
    }


@app.get("/v1/models")
async def list_models():
    """
    Get list of models - provided in OpenAI API compliant format.
    """
    models = await model_provider.get_available_models()
    models_data = []
    for model in models:
        models_data.append(
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "system",
            }
        )
    return {"object": "list", "data": models_data}


@app.post("/v1/models")
async def add_model(model_name: str):
    """
    Add a new model to the API.

    Args:
        model_name (str): The name of the model to add.

    Returns:
        dict (dict): A dictionary containing the status of the operation.
    """
    model_provider.load_model(model_name)
    return {"status": "success", "message": f"Model {model_name} added successfully"}


@app.delete("/v1/models")
async def remove_model(model_name: str):
    """
    Remove a model from the API.

    Args:
        model_name (str): The name of the model to remove.

    Returns:
        Response (str): A 204 No Content response if successful.

    Raises:
        HTTPException (str): If the model is not found.
    """
    model_name = unquote(model_name).strip('"')
    removed = await model_provider.remove_model(model_name)
    if removed:
        return Response(status_code=204)  # 204 No Content - successful deletion
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


async def generate_audio(model, payload: SpeechRequest):
    # Load reference audio if provided
    ref_audio = payload.ref_audio
    audio_chunks = []
    sample_rate = None
    if ref_audio and isinstance(ref_audio, str):
        if not os.path.exists(ref_audio):
            raise HTTPException(
                status_code=400, detail=f"Reference audio file not found: {ref_audio}"
            )
        # Import load_audio from generate module
        from mlx_audio.tts.generate import load_audio

        # Determine if volume normalization is needed
        normalize = hasattr(model, "model_type") and model.model_type == "spark"

        ref_audio = load_audio(
            ref_audio, sample_rate=model.sample_rate, volume_normalize=normalize
        )

    for result in model.generate(
        payload.input,
        voice=payload.voice,
        speed=payload.speed,
        gender=payload.gender,
        pitch=payload.pitch,
        instruct=payload.instruct,
        lang_code=payload.lang_code,
        ref_audio=ref_audio,
        ref_text=payload.ref_text,
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        repetition_penalty=payload.repetition_penalty,
        stream=payload.stream,
        streaming_interval=payload.streaming_interval,
        max_tokens=payload.max_tokens,
        verbose=payload.verbose,
    ):

        if payload.stream:
            buffer = io.BytesIO()
            audio_write(
                buffer, result.audio, result.sample_rate, format=payload.response_format
            )
            yield buffer.getvalue()
        else:
            audio_chunks.append(result.audio)
            if sample_rate is None:
                sample_rate = result.sample_rate

    if payload.stream:
        return

    if not audio_chunks:
        raise HTTPException(status_code=400, detail="No audio generated")

    concatenated_audio = np.concatenate(audio_chunks)
    buffer = io.BytesIO()
    audio_write(buffer, concatenated_audio, sample_rate, format=payload.response_format)
    yield buffer.getvalue()


@app.post("/v1/audio/speech")
async def tts_speech(payload: SpeechRequest):
    """Generate speech audio following the OpenAI text-to-speech API."""
    model = model_provider.load_model(payload.model)
    return StreamingResponse(
        generate_audio(model, payload),
        media_type=f"audio/{payload.response_format}",
        headers={
            "Content-Disposition": f"attachment; filename=speech.{payload.response_format}"
        },
    )


def generate_transcription_stream(stt_model, tmp_path: str, gen_kwargs: dict):
    """Generator that yields transcription chunks and cleans up temp file."""
    try:
        # Call generate with stream=True (models handle streaming internally)
        result = stt_model.generate(tmp_path, **gen_kwargs)

        # Check if result is a generator (streaming mode)
        if hasattr(result, "__iter__") and hasattr(result, "__next__"):
            accumulated_text = ""
            for chunk in result:
                # Handle different chunk types (string tokens vs structured chunks)
                if isinstance(chunk, str):
                    accumulated_text += chunk
                    chunk_data = {"text": chunk, "accumulated": accumulated_text}
                else:
                    # Structured chunk (e.g., Whisper streaming)
                    chunk_data = {
                        "text": chunk.text,
                        "start": getattr(chunk, "start_time", None),
                        "end": getattr(chunk, "end_time", None),
                        "is_final": getattr(chunk, "is_final", None),
                        "language": getattr(chunk, "language", None),
                    }
                yield json.dumps(sanitize_for_json(chunk_data)) + "\n"
        else:
            # Not a generator, yield the full result
            yield json.dumps(sanitize_for_json(result)) + "\n"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/v1/audio/transcriptions")
async def stt_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    verbose: bool = Form(False),
    max_tokens: int = Form(1024),
    chunk_duration: float = Form(30.0),
    frame_threshold: int = Form(25),
    stream: bool = Form(False),
    context: Optional[str] = Form(None),
    prefill_step_size: int = Form(2048),
    text: Optional[str] = Form(None),
):
    """Transcribe audio using an STT model in OpenAI format."""
    # Create TranscriptionRequest from form fields
    payload = TranscriptionRequest(
        model=model,
        language=language,
        verbose=verbose,
        max_tokens=max_tokens,
        chunk_duration=chunk_duration,
        frame_threshold=frame_threshold,
        stream=stream,
        context=context,
        prefill_step_size=prefill_step_size,
        text=text,
    )

    data = await file.read()
    tmp = io.BytesIO(data)
    audio, sr = audio_read(tmp, always_2d=False)
    tmp.close()
    tmp_path = f"/tmp/{time.time()}.mp3"
    audio_write(tmp_path, audio, sr)

    stt_model = model_provider.load_model(payload.model)

    # Build kwargs for generate, filtering None values
    gen_kwargs = payload.model_dump(exclude={"model"}, exclude_none=True)

    # Filter kwargs to only include parameters the model's generate method accepts
    signature = inspect.signature(stt_model.generate)
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in signature.parameters}

    return StreamingResponse(
        generate_transcription_stream(stt_model, tmp_path, gen_kwargs),
        media_type="application/x-ndjson",
    )


SAM_MODEL = None
SAM_PROCESSOR = None


@app.post("/v1/audio/separations")
async def audio_separations(
    file: UploadFile = File(...),
    model: str = Form("mlx-community/sam-audio-large-fp16"),
    description: str = Form("speech"),
    method: str = Form("midpoint"),
    steps: int = Form(16),
):
    """Separate audio using SAM Audio model.

    Args:
        file: Audio file to process
        model: SAM Audio model name (default: mlx-community/sam-audio-large-fp16)
        description: Text description of what to separate (e.g., "speech", "guitar", "drums")
        method: ODE solver method - "midpoint" or "euler" (default: midpoint)
        steps: Number of ODE steps - 2, 4, 8, 16, or 32 (default: 16)

    Returns:
        JSON with base64-encoded target and residual audio, plus sample rate
    """
    global SAM_MODEL, SAM_PROCESSOR
    from mlx_audio.sts import SAMAudio, SAMAudioProcessor

    # Read uploaded file
    data = await file.read()
    tmp = io.BytesIO(data)
    audio, sr = sf.read(tmp, always_2d=False)
    tmp.close()

    # Save to temp file for processor
    tmp_path = f"/tmp/separation_{time.time()}.wav"
    sf.write(tmp_path, audio, sr)

    try:
        # Load model and processor
        if SAM_PROCESSOR is None:
            SAM_PROCESSOR = SAMAudioProcessor.from_pretrained(model)
        if SAM_MODEL is None:
            SAM_MODEL = SAMAudio.from_pretrained(model)

        # Process inputs
        batch = SAM_PROCESSOR(
            descriptions=[description],
            audios=[tmp_path],
        )

        # Calculate step_size from steps
        step_size = 2 / (steps * 2)  # e.g., 16 steps -> 2/32 = 0.0625
        ode_opt = {"method": method, "step_size": step_size}

        # Separate audio
        result = SAM_MODEL.separate_long(
            audios=batch.audios,
            descriptions=batch.descriptions,
            anchor_ids=batch.anchor_ids,
            anchor_alignment=batch.anchor_alignment,
            ode_opt=ode_opt,
            ode_decode_chunk_size=50,
        )

        mx.clear_cache()

        # Convert results to numpy
        target_audio = np.array(result.target[0])
        residual_audio = np.array(result.residual[0])
        sample_rate = SAM_MODEL.sample_rate

        # Encode as base64 WAV
        def audio_to_base64(audio_array, sr):
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sr, format="wav")
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")

        return SeparationResponse(
            target=audio_to_base64(target_audio, sample_rate),
            residual=audio_to_base64(residual_audio, sample_rate),
            sample_rate=sample_rate,
        )

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def _stream_transcription(
    websocket: WebSocket,
    stt_model,
    audio_array: np.ndarray,
    sample_rate: int,
    language: Optional[str],
    is_partial: bool,
    streaming: bool = True,
):
    """Handle both streaming and non-streaming model inference over WebSocket.

    Streaming models (whose generate() accepts a ``stream`` parameter) receive
    the numpy array directly and yield token deltas sent as
    ``{"type": "delta", "delta": "..."}`` messages, followed by a
    ``{"type": "complete", ...}`` message.

    Non-streaming models fall back to temp-file + batch generate, sending the
    legacy ``{"text": ..., "is_partial": ...}`` format.
    """
    supports_stream = "stream" in inspect.signature(stt_model.generate).parameters
    lang_arg = language if language and language != "Detect" else None

    if supports_stream and streaming:
        result_iter = stt_model.generate(
            audio_array, stream=True, language=lang_arg, verbose=False
        )
        accumulated = ""
        for chunk in result_iter:
            delta = (
                chunk if isinstance(chunk, str) else getattr(chunk, "text", str(chunk))
            )
            accumulated += delta
            await websocket.send_json({"type": "delta", "delta": delta})

        await websocket.send_json(
            {
                "type": "complete",
                "text": accumulated,
                "segments": None,
                "language": lang_arg,
                "is_partial": is_partial,
            }
        )
    else:
        tmp_path = f"/tmp/realtime_{time.time()}.mp3"
        audio_write(tmp_path, audio_array, sample_rate)
        try:
            result = stt_model.generate(tmp_path, language=lang_arg, verbose=False)
            segments = (
                sanitize_for_json(result.segments)
                if hasattr(result, "segments") and result.segments
                else None
            )
            await websocket.send_json(
                {
                    "text": result.text,
                    "segments": segments,
                    "language": getattr(result, "language", language),
                    "is_partial": is_partial,
                }
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


@app.websocket("/v1/audio/transcriptions/realtime")
async def stt_realtime_transcriptions(websocket: WebSocket):
    """Realtime transcription via WebSocket."""
    await websocket.accept()

    try:
        # Receive initial configuration
        config = await websocket.receive_json()
        model_name = config.get(
            "model", "mlx-community/whisper-large-v3-turbo-asr-fp16"
        )
        language = config.get("language", None)
        sample_rate = config.get("sample_rate", 16000)
        streaming = config.get("streaming", True)

        print(
            f"Configuration received: model={model_name}, language={language}, sample_rate={sample_rate}, streaming={streaming}"
        )

        # Load the STT model
        print("Loading STT model...")
        stt_model = model_provider.load_model(model_name)
        print("STT model loaded successfully")

        # Initialize WebRTC VAD for speech detection
        vad = webrtcvad.Vad(
            3
        )  # Mode 3 is most aggressive (0-3, higher = more aggressive)
        # VAD requires specific frame sizes: 10ms, 20ms, or 30ms at 8kHz, 16kHz, 32kHz, or 48kHz
        vad_frame_duration_ms = 30  # 30ms frames
        vad_frame_size = int(sample_rate * vad_frame_duration_ms / 1000)
        print(
            f"VAD initialized: frame_size={vad_frame_size} samples ({vad_frame_duration_ms}ms at {sample_rate}Hz)"
        )

        # Buffer for accumulating audio chunks with speech
        audio_buffer = []
        min_chunk_size = int(sample_rate * 0.5)  # Minimum 0.5 seconds before processing
        initial_chunk_size = int(
            sample_rate * 1.5
        )  # Process first 1.5 seconds for real-time feedback
        max_chunk_size = int(
            sample_rate * 5.0
        )  # Maximum 10 seconds to avoid memory issues
        silence_skip_count = 0
        speech_chunk_count = 0
        last_speech_time = time.time()  # Track when we last detected speech
        silence_threshold_seconds = 0.5  # Process when silence > 0.5 seconds
        last_process_time = time.time()
        initial_chunk_processed = False  # Track if we've processed the initial chunk
        processed_samples = 0  # Track how many samples we've already processed

        await websocket.send_json({"status": "ready", "message": "Ready to transcribe"})
        print("Ready to transcribe")

        while True:
            # Receive message
            try:
                message = await websocket.receive()
            except Exception:
                break

            if "bytes" in message:
                # Audio data received as int16
                audio_chunk_int16 = np.frombuffer(message["bytes"], dtype=np.int16)

                # Process audio in VAD frame sizes to detect speech
                # WebRTC VAD requires frames of exactly 10ms, 20ms, or 30ms
                # at sample rates of 8000, 16000, 32000, or 48000 Hz
                num_frames = len(audio_chunk_int16) // vad_frame_size
                has_speech = False
                speech_frames = 0

                # Check each VAD frame for speech activity
                for i in range(num_frames):
                    frame_start = i * vad_frame_size
                    frame_end = frame_start + vad_frame_size
                    frame = audio_chunk_int16[frame_start:frame_end]

                    # VAD requires exact frame size
                    if len(frame) == vad_frame_size:
                        try:
                            if vad.is_speech(frame.tobytes(), sample_rate):
                                has_speech = True
                                speech_frames += 1
                        except (ValueError, OSError) as e:
                            # If VAD fails (wrong sample rate or frame size), assume speech (conservative)
                            # This can happen if sample rate doesn't match VAD requirements
                            print(f"VAD error (assuming speech): {e}")
                            has_speech = True
                            speech_frames += 1

                # Handle remaining samples that don't form a complete frame
                # These will be processed in the next chunk

                # Only accumulate audio if it contains speech
                current_time = time.time()
                if has_speech:
                    # Convert to float32 for buffer
                    audio_chunk_float = audio_chunk_int16.astype(np.float32) / 32768.0
                    audio_buffer.extend(audio_chunk_float)
                    speech_chunk_count += 1
                    silence_skip_count = 0
                    last_speech_time = current_time

                    if len(audio_buffer) % (sample_rate * 2) < len(audio_chunk_float):
                        # Log every ~2 seconds of buffer
                        print(
                            f"Speech detected ({speech_frames}/{num_frames} frames): buffer {len(audio_buffer)} samples ({len(audio_buffer)/sample_rate:.2f}s)"
                        )
                else:
                    silence_skip_count += 1
                    # Only log silence periodically to reduce noise
                    if silence_skip_count % 20 == 0:
                        print(f"Silence detected: skipped {silence_skip_count} chunks")

                # Determine if we should process:
                # 1. Process initial chunk (first 1.5s) for real-time feedback while accumulating
                # 2. If we have silence > 0.5 seconds and buffer has speech (end of utterance)
                # 3. If buffer reaches maximum size (to avoid memory issues)
                time_since_last_speech = current_time - last_speech_time
                should_process_initial = False
                should_process_final = False

                if len(audio_buffer) > 0:
                    # Process initial chunk for real-time feedback (only once per speech segment)
                    if (
                        not initial_chunk_processed
                        and len(audio_buffer) >= initial_chunk_size
                        and has_speech  # Only if we're still detecting speech
                    ):
                        should_process_initial = True
                        print(
                            f"Processing initial chunk for real-time feedback: {initial_chunk_size/sample_rate:.2f}s, total buffer: {len(audio_buffer)/sample_rate:.2f}s"
                        )
                    # Process if we have enough silence after speech (end of utterance)
                    elif (
                        time_since_last_speech >= silence_threshold_seconds
                        and len(audio_buffer) >= min_chunk_size
                    ):
                        should_process_final = True
                        print(
                            f"Processing due to silence gap: {time_since_last_speech:.2f}s silence, buffer: {len(audio_buffer)/sample_rate:.2f}s"
                        )
                    # Or if buffer is getting too large (continuous speech)
                    elif len(audio_buffer) >= max_chunk_size:
                        should_process_final = True
                        print(
                            f"Processing due to max buffer size: {len(audio_buffer)/sample_rate:.2f}s"
                        )

                # Process initial chunk for real-time feedback
                if should_process_initial and len(audio_buffer) >= initial_chunk_size:
                    process_size = initial_chunk_size
                    audio_array = np.array(audio_buffer[:process_size])
                    processed_samples = process_size
                    initial_chunk_processed = True

                    try:
                        await _stream_transcription(
                            websocket,
                            stt_model,
                            audio_array,
                            sample_rate,
                            language,
                            is_partial=True,
                            streaming=streaming,
                        )
                    except Exception as e:
                        import traceback

                        error_msg = str(e)
                        traceback.print_exc()
                        print(f"Error during initial transcription: {error_msg}")
                        await websocket.send_json(
                            {"error": error_msg, "status": "error"}
                        )

                # Process final chunk (entire accumulated buffer)
                if should_process_final and len(audio_buffer) > 0:
                    # Process the entire buffer (continuous speech chunk)
                    process_size = len(audio_buffer)
                    audio_array = np.array(audio_buffer)

                    try:
                        await _stream_transcription(
                            websocket,
                            stt_model,
                            audio_array,
                            sample_rate,
                            language,
                            is_partial=False,
                            streaming=streaming,
                        )

                        # Clear processed audio from buffer and reset state
                        audio_buffer = []
                        processed_samples = 0
                        initial_chunk_processed = False
                        last_process_time = current_time
                        print(
                            f"Processed final chunk: {process_size} samples ({process_size/sample_rate:.2f}s), buffer cleared"
                        )

                    except Exception as e:
                        import traceback

                        error_msg = str(e)
                        traceback.print_exc()
                        print(f"Error during transcription: {error_msg}")
                        await websocket.send_json(
                            {"error": error_msg, "status": "error"}
                        )

            elif "text" in message:
                # JSON message received (e.g., stop command)
                try:
                    data = json.loads(message["text"])
                    if data.get("action") == "stop":
                        break
                except Exception:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e), "status": "error"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


class MLXAudioStudioServer:
    def __init__(self, start_ui=False, log_dir="logs"):
        self.start_ui = start_ui
        self.ui_process = None
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def start_ui_background(self):
        """Start UI with logs redirected to file"""
        ui_path = Path(__file__).parent / "ui"

        try:
            # Install deps silently
            result = subprocess.run(
                ["npm", "install"],
                cwd=str(ui_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError:
            raise Exception(
                "✗ Error: 'npm' is not installed or not found in PATH. UI will not start."
            )
        except subprocess.CalledProcessError as e:
            raise Exception("✗ Error running 'npm install':\n", e)

        try:
            # Start UI with logs to file
            ui_log = open(self.log_dir / "ui.log", "w")
            self.ui_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(ui_path),
                stdout=ui_log,
                stderr=subprocess.STDOUT,
            )
            print(f"✓ UI started (logs: {self.log_dir}/ui.log)")
        except FileNotFoundError:
            raise Exception(
                "✗ Error: 'npm' is not installed or not found in PATH. UI server not started."
            )
        except Exception as e:
            raise Exception(f"✗ Failed to start UI: {e}")

    def start_server(self, host="localhost", port=8000, reload=False, workers=2):
        if self.start_ui:
            self.start_ui_background()
            time.sleep(2)
            webbrowser.open("http://localhost:3000")
            print(f"✓ API server starting on http://{host}:{port}")
            print(f"✓ Studio UI available at http://localhost:3000")
            print("\nPress Ctrl+C to stop both servers")

        try:
            uvicorn.run(
                "mlx_audio.server:app",
                host=host,
                port=port,
                reload=reload,
                workers=workers,
                loop="asyncio",
            )
        finally:
            if self.ui_process:
                self.ui_process.terminate()
                print("✓ UI server stopped")

            ui_log_path = self.log_dir / "ui.log"
            if ui_log_path.exists():
                ui_log_path.unlink()
                print(f"✓ UI logs deleted from {ui_log_path}")


def main():
    parser = argparse.ArgumentParser(description="MLX Audio API server")
    parser.add_argument(
        "--allowed-origins",
        nargs="+",
        default=["*"],
        help="List of allowed origins for CORS",
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--reload",
        type=bool,
        default=False,
        help="Enable auto-reload of the server. Only works when 'workers' is set to None.",
    )

    parser.add_argument(
        "--workers",
        type=int_or_float,
        default=0,
        help="""Number of workers. Overrides the `MLX_AUDIO_NUM_WORKERS` env variable.
        Can be either an int or a float.
        If an int, it will be the number of workers to use.
        If a float, number of workers will be this fraction of the  number of CPU cores available, with a minimum of 1.
        Defaults to the `MLX_AUDIO_NUM_WORKERS` env variable if set and to 2 if not.
        To use all available CPU cores, set it to 1.0.

        Examples:
        --workers 1 (will use 1 worker)
        --workers 1.0 (will use all available CPU cores)
        --workers 0.5 (will use half the number of CPU cores available)
        --workers 0.0 (will use 1 worker)""",
    )
    parser.add_argument(
        "--start-ui",
        action="store_true",
        help="Start the Studio UI alongside the API server",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save server logs",
    )

    args = parser.parse_args()
    if isinstance(args.workers, float):
        args.workers = max(1, int(os.cpu_count() * args.workers))

    setup_cors(app, args.allowed_origins)

    client = MLXAudioStudioServer(start_ui=args.start_ui, log_dir=args.log_dir)
    client.start_server(
        host=args.host,
        port=args.port,
        reload=args.reload if args.workers is None else False,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
