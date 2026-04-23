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
import uuid
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
    Request,
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

DIARIZE_FIRST_MODELS = ["voxtral", "vibevoice", "moonshine"]

def needs_diarize_first(model_name: str) -> bool:
    """Returns True for LLM-based STT models that don't produce timestamps."""
    return any(m in model_name.lower() for m in DIARIZE_FIRST_MODELS)

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
    diarize: bool = False
    diarization_model: str | None = None


class SeparationResponse(BaseModel):
    target: str  # Base64 encoded WAV
    residual: str  # Base64 encoded WAV
    sample_rate: int


# Initialize the ModelProvider
model_provider = ModelProvider()
REALTIME_INFERENCE_LOCK = asyncio.Lock()


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


async def generate_audio(model, payload: SpeechRequest, request: Request):
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

        if await request.is_disconnected():
            mx.clear_cache()
            return

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

        await asyncio.sleep(0)  # register any disconnects

    if payload.stream:
        return

    if not audio_chunks:
        raise HTTPException(status_code=400, detail="No audio generated")

    concatenated_audio = np.concatenate(audio_chunks)
    buffer = io.BytesIO()
    audio_write(buffer, concatenated_audio, sample_rate, format=payload.response_format)
    yield buffer.getvalue()


@app.post("/v1/audio/speech")
async def tts_speech(payload: SpeechRequest, request: Request):
    """Generate speech audio following the OpenAI text-to-speech API."""
    model = model_provider.load_model(payload.model)
    return StreamingResponse(
        generate_audio(model, payload, request),
        media_type=f"audio/{payload.response_format}",
        headers={
            "Content-Disposition": f"attachment; filename=speech.{payload.response_format}"
        },
    )

def model_produces_segments(stt_model) -> bool:
    """Check if model natively returns timestamped segments."""
    # Check result of a dummy or inspect model type/class name
    model_class = type(stt_model).__name__.lower()
    return "whisper" in model_class


def generate_diarize_first_stream(
    stt_model, tmp_path: str, gen_kwargs: dict, diarization_model_name: str
):
    #Diarize-first pipeline: segment by speaker, then transcribe each slice.
    try:
        import soundfile as sf
        import tempfile
        from mlx_audio.vad import load as load_vad

        audio_data, sample_rate = sf.read(tmp_path)
        diar_model = load_vad(diarization_model_name)

        all_segments = []
        for result in diar_model.generate_stream(
            tmp_path, threshold=0.4, min_duration=0.25,
            merge_gap=0.5, chunk_duration=5.0, verbose=False,
        ):
            for seg in result.segments:
                all_segments.append(seg)

        collected = []
        for seg in all_segments:
            duration = seg.end - seg.start
            if duration < 0.3:
                continue

            start_sample = int(seg.start * sample_rate)
            end_sample = int(seg.end * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, segment_audio, sample_rate)
                seg_path = tmp.name

            try:
                result = stt_model.generate(seg_path, **gen_kwargs)
                text = result.text.strip() if hasattr(result, "text") else str(result).strip()
            except Exception as e:
                text = f"[error: {e}]"
            finally:
                os.unlink(seg_path)

            chunk = {
                "text": text,
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "accumulated": text,
            }
            collected.append(chunk)
            # Stream each segment as it completes for real-time UI feedback
            yield json.dumps(sanitize_for_json(chunk)) + "\n"

        yield json.dumps({
            "type": "diarization_complete",
            "segments": collected,
        }) + "\n"

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def generate_transcription_stream(stt_model, tmp_path: str, gen_kwargs: dict, diarize: bool = False, diarization_model_name: str | None = None):
    try:
        # Force non-streaming when diarizing so we get full segments with timestamps
        if diarize:
            gen_kwargs = {**gen_kwargs, "stream": False}

        result = stt_model.generate(tmp_path, **gen_kwargs)

        if hasattr(result, "__iter__") and hasattr(result, "__next__") and not diarize:
            # Streaming path — only used when NOT diarizing
            accumulated_text = ""
            for chunk in result:
                if isinstance(chunk, str):
                    accumulated_text += chunk
                    chunk_data = {"text": chunk, "accumulated": accumulated_text}
                else:
                    chunk_data = {
                        "text": chunk.text,
                        "start": getattr(chunk, "start_time", None),
                        "end": getattr(chunk, "end_time", None),
                        "is_final": getattr(chunk, "is_final", None),
                        "language": getattr(chunk, "language", None),
                    }
                yield json.dumps(sanitize_for_json(chunk_data)) + "\n"
        else:
            # batch result — always used when diarizing
            result_dict = sanitize_for_json(result)

            if diarize and diarization_model_name and hasattr(result, "segments") and result.segments:
                try:
                    from mlx_audio.vad import load as load_vad
                    diar_model = load_vad(diarization_model_name)
                    diar_result = diar_model.generate(tmp_path, verbose=False)

                    segments = sanitize_for_json(result.segments)
                    for seg in segments:
                        if seg.get("start") is None or seg.get("end") is None:
                            seg["speaker"] = None
                            continue
                        best_speaker = None
                        best_overlap = 0.0
                        for d in diar_result.segments:
                            overlap = max(0, min(seg["end"], d.end) - max(seg["start"], d.start))
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_speaker = f"SPEAKER_{int(d.speaker):02d}"
                        seg["speaker"] = best_speaker

                    yield json.dumps(sanitize_for_json(result_dict)) + "\n"
                    yield json.dumps({"type": "diarization_complete", "segments": segments}) + "\n"
                except Exception as e:
                    yield json.dumps(sanitize_for_json(result_dict)) + "\n"
                    yield json.dumps({"type": "diarization_error", "error": str(e)}) + "\n"
            else:
                yield json.dumps(sanitize_for_json(result_dict)) + "\n"
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
    diarize: bool = Form(False),
    diarization_model: Optional[str] = Form(None), 
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
        diarize=diarize,
        diarization_model=diarization_model,
    )

    data = await file.read()
    tmp = io.BytesIO(data)
    audio, sr = audio_read(tmp, always_2d=False)
    tmp.close()
    _, ext = os.path.splitext(file.filename)
    tmp_path = f"/tmp/{time.time()}.{ext if ext else 'mp3'}"
    audio_write(tmp_path, audio, sr)

    stt_model = model_provider.load_model(payload.model)

    # Build kwargs for generate, filtering None values
    gen_kwargs = payload.model_dump(exclude={"model", "diarize", "diarization_model"}, exclude_none=True)

    # Filter kwargs to only include parameters the model's generate method accepts
    signature = inspect.signature(stt_model.generate)
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in signature.parameters}
    if payload.diarize and payload.diarization_model:
        if needs_diarize_first(payload.model):
            generator = generate_diarize_first_stream(
                stt_model, tmp_path, gen_kwargs, payload.diarization_model
            )
        else:
            generator = generate_transcription_stream(
                stt_model, tmp_path, gen_kwargs,
                diarize=True,
                diarization_model_name=payload.diarization_model
            )
    else:
        generator = generate_transcription_stream(stt_model, tmp_path, gen_kwargs)

    return StreamingResponse(generator, media_type="application/x-ndjson")


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
    audio, sr = audio_read(tmp, always_2d=False)
    tmp.close()

    # Save to temp file for processor
    tmp_path = f"/tmp/separation_{time.time()}.wav"
    audio_write(tmp_path, audio, sr)

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
            audio_write(buffer, audio_array, sr, format="wav")
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
    the audio as an ``mx.array`` and yield token deltas sent as
    ``{"type": "delta", "delta": "..."}`` messages, followed by a
    ``{"type": "complete", ...}`` message.

    Non-streaming models fall back to temp-file + batch generate, sending the
    legacy ``{"text": ..., "is_partial": ...}`` format.
    """
    supports_stream = "stream" in inspect.signature(stt_model.generate).parameters

    if supports_stream and streaming:
        result_iter = stt_model.generate(
            mx.array(audio_array), stream=True, language=language, verbose=False
        )
        accumulated = ""
        detected_language = language
        for chunk in result_iter:
            delta = (
                chunk if isinstance(chunk, str) else getattr(chunk, "text", str(chunk))
            )
            accumulated += delta
            # Pick up detected language from streaming results
            chunk_lang = getattr(chunk, "language", None)
            if chunk_lang and detected_language is None:
                detected_language = chunk_lang
            await websocket.send_json({"type": "delta", "delta": delta})

        await websocket.send_json(
            {
                "type": "complete",
                "text": accumulated,
                "segments": None,
                "language": detected_language,
                "is_partial": is_partial,
            }
        )
    else:
        tmp_path = f"/tmp/realtime_{time.time()}.mp3"
        audio_write(tmp_path, audio_array, sample_rate)
        try:
            result = stt_model.generate(tmp_path, language=language, verbose=False)
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
        initial_chunk_processed = False  # Track if we've processed the initial chunk

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
                        initial_chunk_processed = False
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


# OpenAI Realtime clients send PCM at 24 kHz by default, so we assume that
# until the client declares a different ``audio.input.format.rate``. The
# model-side rate is read off the streaming session's ``input_sample_rate``
# attribute; the server resamples client → model on ingress.
_REALTIME_DEFAULT_CLIENT_RATE = 24000


def _default_transcription_delay_ms() -> Optional[int]:
    raw = os.getenv("MLX_AUDIO_REALTIME_TRANSCRIPTION_DELAY_MS")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _open_streaming_session(model, *, temperature: float, delay_ms: Optional[int]):
    """Open a streaming session, forwarding ``transcription_delay_ms`` only to
    models that declare the parameter.
    """
    kwargs: dict = {"temperature": temperature}
    if delay_ms is not None:
        sig = inspect.signature(model.create_streaming_session)
        if "transcription_delay_ms" in sig.parameters:
            kwargs["transcription_delay_ms"] = delay_ms
    return model.create_streaming_session(**kwargs)


def _resample_pcm16_to_rate(
    pcm16: np.ndarray, from_rate: int, to_rate: int
) -> np.ndarray:
    """Linear resample int16 PCM to ``to_rate``.

    Returns float32 samples in [-1, 1]. If the rates already match, this is
    a plain dtype cast.
    """
    samples = pcm16.astype(np.float32) / 32768.0
    if from_rate == to_rate or samples.size == 0:
        return samples
    n_out = int(round(samples.size * to_rate / from_rate))
    if n_out <= 1:
        return samples[:n_out].astype(np.float32, copy=False)
    src_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=False, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def _resolve_realtime_model_name(requested_model: Optional[str]) -> Optional[str]:
    """Return the requested model ID, or the server-configured default.

    The default is read from ``$MLX_AUDIO_REALTIME_MODEL`` (settable via the
    ``--realtime-model`` CLI flag). Returns ``None`` when neither the client
    nor the server supplies a model — the caller is expected to reject the
    connection with a clear error.
    """
    normalized = (requested_model or "").strip()
    if normalized:
        return normalized
    configured = (os.getenv("MLX_AUDIO_REALTIME_MODEL") or "").strip()
    return configured or None


@app.websocket("/v1/realtime")
async def realtime_ws(websocket: WebSocket):
    """OpenAI Realtime API-compatible WebSocket endpoint.

    Works with any STT model whose ``load_model`` result exposes the
    streaming-session protocol. A streaming session must provide:
      - ``feed(samples: np.ndarray[float32])``
      - ``close()``
      - ``step(max_decode_tokens: int) -> list[str]``
      - ``done: bool``
      - ``input_sample_rate: int``  (the native rate expected by ``feed``)
    Models that support an adjustable transcription-delay / latency knob
    should accept ``transcription_delay_ms: Optional[int]`` on
    ``create_streaming_session``; models without the concept can ignore it.

    Client → server events follow OpenAI Realtime (subset):
      ``session.update``, ``input_audio_buffer.append``,
      ``input_audio_buffer.commit``.
    Server → client events: ``session.created`` / ``session.updated``,
    ``conversation.item.added``, ``input_audio_buffer.committed``,
    ``conversation.item.input_audio_transcription.delta`` /
    ``.completed``, and ``error``.

    Model is selected via ``?model=<id>`` or ``session.update.model``;
    defaults to ``$MLX_AUDIO_REALTIME_MODEL``.
    """
    await websocket.accept()

    def _new_event_id() -> str:
        return f"event_{uuid.uuid4().hex[:16]}"

    async def send_event(payload: dict):
        payload = {"event_id": _new_event_id(), **payload}
        await websocket.send_json(payload)

    async def send_error(message: str):
        await send_event({"type": "error", "error": {"message": message}})

    requested_model = websocket.query_params.get("model")
    model_name = _resolve_realtime_model_name(requested_model)
    if model_name is None:
        await send_error(
            "no realtime model configured: pass ?model=<id>, set "
            "session.update.model, or start the server with --realtime-model"
        )
        await websocket.close()
        return
    try:
        model = model_provider.load_model(model_name)
    except Exception as e:
        await send_error(f"load failed: {e}")
        await websocket.close()
        return

    if not hasattr(model, "create_streaming_session"):
        await send_error(f"model {model_name!r} does not support streaming")
        await websocket.close()
        return

    temperature = 0.0
    transcription_delay_ms = _default_transcription_delay_ms()
    session = _open_streaming_session(
        model, temperature=temperature, delay_ms=transcription_delay_ms
    )
    full_text_parts: list[str] = []
    current_item_id: Optional[str] = None
    client_input_rate = _REALTIME_DEFAULT_CLIENT_RATE

    def _new_item_id() -> str:
        return f"item_{uuid.uuid4().hex[:16]}"

    async def drain_deltas(max_decode_tokens: int = 8) -> bool:
        """Run one session.step off-loop, ship deltas. Returns session.done."""
        async with REALTIME_INFERENCE_LOCK:
            deltas = await asyncio.to_thread(
                session.step, max_decode_tokens=max_decode_tokens
            )
        for delta in deltas:
            full_text_parts.append(delta)
            await send_event(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": current_item_id,
                    "content_index": 0,
                    "delta": delta,
                }
            )
        return session.done

    async def send_done():
        text = "".join(full_text_parts)
        await send_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": current_item_id,
                "content_index": 0,
                "transcript": text,
            }
        )

    session_id = f"sess_{uuid.uuid4().hex[:16]}"

    def _session_snapshot() -> dict:
        return {
            "id": session_id,
            "object": "realtime.session",
            "type": "transcription",
            "model": model_name,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": client_input_rate},
                    "transcription": {"model": model_name},
                    "turn_detection": None,
                }
            },
        }

    await send_event({"type": "session.created", "session": _session_snapshot()})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send_error("invalid JSON")
                continue

            msg_type = msg.get("type", "")

            if msg_type == "session.update":
                session_payload = msg.get("session") or {}
                audio_input = (session_payload.get("audio") or {}).get("input") or {}
                transcription_cfg = audio_input.get("transcription") or {}
                resolved_model = session_payload.get("model") or transcription_cfg.get(
                    "model"
                )
                fmt = audio_input.get("format") or {}
                requested_rate = fmt.get("rate")
                if isinstance(requested_rate, int) and requested_rate > 0:
                    client_input_rate = requested_rate
                target_model_name = _resolve_realtime_model_name(resolved_model)
                if target_model_name and target_model_name != model_name:
                    try:
                        model = model_provider.load_model(target_model_name)
                    except Exception as e:
                        await send_error(f"load failed: {e}")
                        continue
                    if not hasattr(model, "create_streaming_session"):
                        await send_error(
                            f"model {target_model_name!r} does not support streaming"
                        )
                        continue
                    model_name = target_model_name
                    async with REALTIME_INFERENCE_LOCK:
                        session = _open_streaming_session(
                            model,
                            temperature=temperature,
                            delay_ms=transcription_delay_ms,
                        )
                    full_text_parts = []
                    current_item_id = None

                await send_event(
                    {"type": "session.updated", "session": _session_snapshot()}
                )

            elif msg_type == "input_audio_buffer.append":
                audio_b64 = msg.get("audio", "")
                if not audio_b64:
                    continue
                if current_item_id is None:
                    current_item_id = _new_item_id()
                    await send_event(
                        {
                            "type": "conversation.item.added",
                            "item": {
                                "id": current_item_id,
                                "object": "realtime.item",
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_audio"}],
                            },
                        }
                    )
                pcm16 = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                samples = _resample_pcm16_to_rate(
                    pcm16, client_input_rate, session.input_sample_rate
                )
                async with REALTIME_INFERENCE_LOCK:
                    session.feed(samples)
                # Opportunistic draining between chunks so deltas flow early.
                await drain_deltas(max_decode_tokens=8)

            elif msg_type == "input_audio_buffer.commit":
                if current_item_id is None:
                    current_item_id = _new_item_id()
                    await send_event(
                        {
                            "type": "conversation.item.added",
                            "item": {
                                "id": current_item_id,
                                "object": "realtime.item",
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_audio"}],
                            },
                        }
                    )
                await send_event(
                    {
                        "type": "input_audio_buffer.committed",
                        "item_id": current_item_id,
                        "previous_item_id": None,
                    }
                )
                async with REALTIME_INFERENCE_LOCK:
                    session.close()
                while not await drain_deltas(max_decode_tokens=16):
                    pass
                await send_done()
                async with REALTIME_INFERENCE_LOCK:
                    session = _open_streaming_session(
                        model,
                        temperature=temperature,
                        delay_ms=transcription_delay_ms,
                    )
                full_text_parts = []
                current_item_id = None

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await send_error(str(e))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        mx.clear_cache()


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
            subprocess.run(
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
            print("✓ Studio UI available at http://localhost:3000")
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
    parser.add_argument(
        "--realtime-model",
        type=str,
        default=None,
        help=(
            "Default model for /v1/realtime when the client omits ?model=. "
            "Overrides $MLX_AUDIO_REALTIME_MODEL."
        ),
    )
    parser.add_argument(
        "--realtime-transcription-delay-ms",
        type=int,
        default=None,
        help=(
            "Transcription latency/quality knob for streaming STT models that "
            "expose a ``transcription_delay_ms`` parameter (e.g. voxtral_realtime). "
            "Lower values reduce latency at the cost of accuracy. When unset, "
            "each model uses its own default. Overrides "
            "$MLX_AUDIO_REALTIME_TRANSCRIPTION_DELAY_MS."
        ),
    )

    args = parser.parse_args()
    if args.realtime_model:
        os.environ["MLX_AUDIO_REALTIME_MODEL"] = args.realtime_model
    if args.realtime_transcription_delay_ms is not None:
        os.environ["MLX_AUDIO_REALTIME_TRANSCRIPTION_DELAY_MS"] = str(
            args.realtime_transcription_delay_ms
        )
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
