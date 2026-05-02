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
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
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
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from mlx_audio.audio_io import read as audio_read
from mlx_audio.audio_io import write as audio_write
from mlx_audio.server_inference import (
    BaseModelExecutionAdapter,
    InferenceBroker,
    InferenceHandle,
    InferenceRequest,
    InferenceResultChunk,
)
from mlx_audio.tts.continuous import TTSBatchItem, TTSBatchOptions
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


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    del app
    try:
        yield
    finally:
        global INFERENCE_BROKER
        if INFERENCE_BROKER is not None:
            INFERENCE_BROKER.stop_and_join()
            INFERENCE_BROKER = None


app.router.lifespan_context = app_lifespan


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
REALTIME_INFERENCE_LOCK = asyncio.Lock()
INFERENCE_BROKER: Optional[InferenceBroker] = None


@dataclass
class SpeechTaskPayload:
    request: SpeechRequest


@dataclass
class TranscriptionTaskPayload:
    request: TranscriptionRequest
    filename: str
    audio: np.ndarray
    sample_rate: int


@dataclass
class SeparationTaskPayload:
    model_name: str
    audio: np.ndarray
    sample_rate: int
    description: str
    method: str
    steps: int


def _load_model_for_inference(model_name: str):
    return model_provider.load_model(model_name)


class STTExecutionAdapter(BaseModelExecutionAdapter):
    def run_serial(self, request: InferenceRequest) -> None:
        payload: TranscriptionTaskPayload = request.payload
        _, ext = os.path.splitext(payload.filename or "")
        suffix = ext or ".mp3"
        tmp_path = f"/tmp/{time.time()}_{request.request_id}{suffix}"
        audio_write(tmp_path, payload.audio, payload.sample_rate)

        try:
            stt_model = _load_model_for_inference(request.model_name)
            gen_kwargs = payload.request.model_dump(
                exclude={"model"}, exclude_none=True
            )
            signature = inspect.signature(stt_model.generate)
            gen_kwargs = {
                key: value
                for key, value in gen_kwargs.items()
                if key in signature.parameters
            }

            result = stt_model.generate(tmp_path, **gen_kwargs)
            if hasattr(result, "__iter__") and hasattr(result, "__next__"):
                accumulated_text = ""
                for chunk in result:
                    if request.cancel_event.is_set():
                        break
                    if isinstance(chunk, str):
                        accumulated_text += chunk
                        chunk_data = {
                            "text": chunk,
                            "accumulated": accumulated_text,
                        }
                    else:
                        chunk_data = {
                            "text": chunk.text,
                            "start": getattr(chunk, "start_time", None),
                            "end": getattr(chunk, "end_time", None),
                            "is_final": getattr(chunk, "is_final", None),
                            "language": getattr(chunk, "language", None),
                        }
                    request.emit_data(json.dumps(sanitize_for_json(chunk_data)) + "\n")
            elif not request.cancel_event.is_set():
                request.emit_data(json.dumps(sanitize_for_json(result)) + "\n")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        request.emit_done()


class _TTSAdapterContinuousSession:
    def __init__(
        self,
        *,
        adapter: "TTSExecutionAdapter",
        model_session,
    ):
        self.adapter = adapter
        self.model_session = model_session
        self._requests: dict[int, InferenceRequest] = {}
        self._emitted_audio: set[int] = set()
        self._next_sequence_id = 0

    @property
    def idle(self) -> bool:
        return self.model_session.idle and not self._requests

    def submit(self, request: InferenceRequest) -> None:
        if request.cancel_event.is_set():
            request.emit_done()
            return

        payload: SpeechTaskPayload = request.payload
        speech_request = payload.request
        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1
        self._requests[sequence_id] = request
        self.model_session.add(
            [
                TTSBatchItem(
                    sequence_id=sequence_id,
                    text=speech_request.input,
                    voice=speech_request.voice,
                    instruct=speech_request.instruct,
                    speed=speech_request.speed,
                    gender=speech_request.gender,
                    pitch=speech_request.pitch,
                    ref_audio=speech_request.ref_audio,
                    ref_text=speech_request.ref_text,
                )
            ]
        )

    def step(self) -> None:
        self._cancel_disconnected_requests()
        events = self.model_session.step()
        for event in events:
            request = self._requests.get(event.sequence_id)
            if request is None:
                continue

            if event.error is not None:
                request.emit_error(event.error)
                request.emit_done()
                self._requests.pop(event.sequence_id, None)
                self._emitted_audio.discard(event.sequence_id)
                continue

            if event.audio is not None and event.samples > 0:
                speech_request: SpeechRequest = request.payload.request
                if event.sample_rate is None:
                    request.emit_error(
                        ValueError("TTS batch event included audio without sample_rate")
                    )
                    request.emit_done()
                    self._requests.pop(event.sequence_id, None)
                    self._emitted_audio.discard(event.sequence_id)
                    continue
                self.adapter._emit_audio(
                    request,
                    speech_request,
                    event.audio,
                    event.sample_rate,
                )
                self._emitted_audio.add(event.sequence_id)

            if event.done:
                if event.sequence_id not in self._emitted_audio:
                    request.emit_error(
                        HTTPException(status_code=400, detail="No audio generated")
                    )
                request.emit_done()
                self._requests.pop(event.sequence_id, None)
                self._emitted_audio.discard(event.sequence_id)

    def fail(self, error: BaseException) -> None:
        for request in list(self._requests.values()):
            request.emit_error(error)
            request.emit_done()
        self._requests.clear()
        self._emitted_audio.clear()

    def _cancel_disconnected_requests(self) -> None:
        for sequence_id, request in list(self._requests.items()):
            if not request.cancel_event.is_set():
                continue
            self.model_session.cancel(sequence_id)
            request.emit_done()
            self._requests.pop(sequence_id, None)
            self._emitted_audio.discard(sequence_id)


class TTSExecutionAdapter(BaseModelExecutionAdapter):
    _REQUEST_MODEL_ATTR = "_mlx_audio_loaded_model"

    def __init__(self):
        raw_max_batch_size = os.getenv("MLX_AUDIO_TTS_MAX_BATCH_SIZE", "8")
        try:
            self.max_batch_size = max(1, int(raw_max_batch_size))
        except ValueError:
            self.max_batch_size = 8

    def _get_model_for_request(self, request: InferenceRequest):
        model = getattr(request, self._REQUEST_MODEL_ATTR, None)
        if model is None:
            model = _load_model_for_inference(request.model_name)
            setattr(request, self._REQUEST_MODEL_ATTR, model)
        return model

    def _build_batch_options(self, request: InferenceRequest) -> TTSBatchOptions:
        speech_request: SpeechRequest = request.payload.request
        return TTSBatchOptions(
            temperature=speech_request.temperature,
            top_p=speech_request.top_p,
            top_k=speech_request.top_k,
            repetition_penalty=speech_request.repetition_penalty,
            max_tokens=speech_request.max_tokens,
            lang_code=speech_request.lang_code,
            stream=speech_request.stream,
            streaming_interval=speech_request.streaming_interval,
            max_batch_size=self.max_batch_size,
            verbose=speech_request.verbose,
        )

    def _request_kwargs(self, request: InferenceRequest) -> dict[str, Any]:
        speech_request: SpeechRequest = request.payload.request
        return speech_request.model_dump(exclude={"model"}, exclude_none=True)

    def _get_callable_model_attr(self, model, name: str):
        value = getattr(model, name, None)
        if not callable(value):
            return None

        # MagicMock fabricates missing attributes on demand; only treat explicit
        # mock attributes as model hooks in tests.
        if type(model).__module__ == "unittest.mock" and name not in vars(model):
            return None
        return value

    def _model_supports_request(
        self,
        model,
        request: InferenceRequest,
        *,
        continuous: bool,
    ) -> bool:
        speech_request: SpeechRequest = request.payload.request
        hook_name = (
            "supports_tts_continuous_batch" if continuous else "supports_tts_batch"
        )
        hook = self._get_callable_model_attr(model, hook_name)
        if callable(hook):
            return bool(hook(**self._request_kwargs(request)))

        if speech_request.stream:
            return False
        if speech_request.ref_audio or speech_request.ref_text:
            return False
        if speech_request.gender not in (None, "male"):
            return False
        if speech_request.speed not in (None, 1.0):
            return False
        if speech_request.pitch not in (None, 1.0):
            return False
        return True

    def _can_call_batch_generate(self, model, request: InferenceRequest) -> bool:
        batch_generate = self._get_callable_model_attr(model, "batch_generate")
        if batch_generate is None:
            return False
        if not self._model_supports_request(model, request, continuous=False):
            return False

        speech_request: SpeechRequest = request.payload.request
        if speech_request.stream:
            return False

        signature = inspect.signature(batch_generate)
        params = signature.parameters
        if "texts" not in params:
            return False
        if speech_request.voice and "voices" not in params:
            return False
        if speech_request.instruct and "instructs" not in params:
            return False
        if speech_request.gender not in (None, "male") and "genders" not in params:
            return False
        if speech_request.speed not in (None, 1.0) and "speeds" not in params:
            return False
        if speech_request.pitch not in (None, 1.0) and "pitches" not in params:
            return False
        if speech_request.ref_audio and "ref_audios" not in params:
            return False
        if speech_request.ref_text and "ref_texts" not in params:
            return False
        return True

    def _can_call_continuous_session(self, model, request: InferenceRequest) -> bool:
        create_session = self._get_callable_model_attr(
            model,
            "create_tts_batch_session",
        )
        return create_session is not None and self._model_supports_request(
            model, request, continuous=True
        )

    def supports_batch(self, request: InferenceRequest) -> bool:
        model = self._get_model_for_request(request)
        return self._can_call_batch_generate(model, request)

    def supports_continuous_batch(self, request: InferenceRequest) -> bool:
        model = self._get_model_for_request(request)
        return self._can_call_continuous_session(model, request)

    def batch_key(self, request: InferenceRequest) -> Any:
        speech_request: SpeechRequest = request.payload.request
        return (
            "tts",
            speech_request.stream,
            speech_request.lang_code,
            speech_request.temperature,
            speech_request.top_p,
            speech_request.top_k,
            speech_request.repetition_penalty,
            speech_request.max_tokens,
            speech_request.ref_audio,
            speech_request.ref_text,
            speech_request.streaming_interval if speech_request.stream else None,
            speech_request.verbose,
        )

    def continuous_batch_key(self, request: InferenceRequest) -> Any:
        return self.batch_key(request)

    def create_continuous_batch_session(self, request: InferenceRequest):
        model = self._get_model_for_request(request)
        create_session = self._get_callable_model_attr(
            model,
            "create_tts_batch_session",
        )
        if create_session is None:
            raise NotImplementedError("Model does not provide create_tts_batch_session")
        model_session = create_session(self._build_batch_options(request))
        return _TTSAdapterContinuousSession(
            adapter=self,
            model_session=model_session,
        )

    def _emit_audio(
        self,
        request: InferenceRequest,
        speech_request: SpeechRequest,
        audio,
        sample_rate: int,
    ) -> None:
        buffer = io.BytesIO()
        audio_write(
            buffer,
            audio,
            sample_rate,
            format=speech_request.response_format,
        )
        request.emit_data(buffer.getvalue())

    def run_serial(self, request: InferenceRequest) -> None:
        payload: SpeechTaskPayload = request.payload
        speech_request = payload.request
        model = self._get_model_for_request(request)

        ref_audio = speech_request.ref_audio
        if ref_audio and isinstance(ref_audio, str):
            if not os.path.exists(ref_audio):
                raise HTTPException(
                    status_code=400,
                    detail=f"Reference audio file not found: {ref_audio}",
                )

            from mlx_audio.tts.generate import load_audio

            normalize = hasattr(model, "model_type") and model.model_type == "spark"
            ref_audio = load_audio(
                ref_audio,
                sample_rate=model.sample_rate,
                volume_normalize=normalize,
            )

        audio_chunks = []
        sample_rate = None
        generate_kwargs = {
            "voice": speech_request.voice,
            "speed": speech_request.speed,
            "gender": speech_request.gender,
            "pitch": speech_request.pitch,
            "instruct": speech_request.instruct,
            "lang_code": speech_request.lang_code,
            "ref_audio": ref_audio,
            "ref_text": speech_request.ref_text,
            "temperature": speech_request.temperature,
            "top_p": speech_request.top_p,
            "top_k": speech_request.top_k,
            "repetition_penalty": speech_request.repetition_penalty,
            "stream": speech_request.stream,
            "streaming_interval": speech_request.streaming_interval,
            "max_tokens": speech_request.max_tokens,
            "verbose": speech_request.verbose,
        }

        for result in model.generate(speech_request.input, **generate_kwargs):
            if request.cancel_event.is_set():
                mx.clear_cache()
                request.emit_done()
                return

            if speech_request.stream:
                self._emit_audio(
                    request,
                    speech_request,
                    result.audio,
                    result.sample_rate,
                )
            else:
                audio_chunks.append(result.audio)
                if sample_rate is None:
                    sample_rate = result.sample_rate

        if speech_request.stream:
            request.emit_done()
            return

        if not audio_chunks:
            raise HTTPException(status_code=400, detail="No audio generated")

        concatenated_audio = np.concatenate(audio_chunks)
        self._emit_audio(request, speech_request, concatenated_audio, sample_rate)
        request.emit_done()

    def run_batch(self, requests: list[InferenceRequest]) -> None:
        if len(requests) == 1:
            self.run_serial(requests[0])
            return

        model = self._get_model_for_request(requests[0])
        if not all(
            self._can_call_batch_generate(model, request) for request in requests
        ):
            for request in requests:
                self.run_serial(request)
            return

        first_speech_request: SpeechRequest = requests[0].payload.request
        texts = [request.payload.request.input for request in requests]
        voices = [request.payload.request.voice for request in requests]
        instructs = [request.payload.request.instruct for request in requests]
        speeds = [request.payload.request.speed for request in requests]
        genders = [request.payload.request.gender for request in requests]
        pitches = [request.payload.request.pitch for request in requests]
        ref_audios = [request.payload.request.ref_audio for request in requests]
        ref_texts = [request.payload.request.ref_text for request in requests]

        batch_generate = self._get_callable_model_attr(model, "batch_generate")
        if batch_generate is None:
            for request in requests:
                self.run_serial(request)
            return

        signature = inspect.signature(batch_generate)
        kwargs = {
            "texts": texts,
            "voices": voices,
            "instructs": instructs,
            "speeds": speeds,
            "genders": genders,
            "pitches": pitches,
            "ref_audios": ref_audios,
            "ref_texts": ref_texts,
            "temperature": first_speech_request.temperature,
            "lang_code": first_speech_request.lang_code,
            "max_tokens": first_speech_request.max_tokens,
            "top_k": first_speech_request.top_k,
            "top_p": first_speech_request.top_p,
            "repetition_penalty": first_speech_request.repetition_penalty,
            "stream": False,
            "verbose": first_speech_request.verbose,
        }
        kwargs = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }

        audio_chunks_by_sequence: list[list[Any]] = [[] for _ in requests]
        sample_rates: list[int | None] = [None for _ in requests]

        for result in batch_generate(**kwargs):
            sequence_idx = result.sequence_idx
            if sequence_idx < 0 or sequence_idx >= len(requests):
                continue
            request = requests[sequence_idx]
            if request.cancel_event.is_set():
                continue
            audio_chunks_by_sequence[sequence_idx].append(result.audio)
            sample_rates[sequence_idx] = result.sample_rate

        for sequence_idx, request in enumerate(requests):
            if request.cancel_event.is_set():
                request.emit_done()
                continue

            chunks = audio_chunks_by_sequence[sequence_idx]
            sample_rate = sample_rates[sequence_idx]
            if not chunks or sample_rate is None:
                request.emit_error(
                    HTTPException(status_code=400, detail="No audio generated")
                )
                request.emit_done()
                continue

            audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            self._emit_audio(
                request,
                request.payload.request,
                audio,
                sample_rate,
            )
            request.emit_done()


class SeparationExecutionAdapter(BaseModelExecutionAdapter):
    def __init__(self):
        self._model_name: str | None = None
        self._model = None
        self._processor = None

    def _get_resources(self, model_name: str):
        if self._model_name == model_name and self._model is not None:
            return self._model, self._processor

        from mlx_audio.sts import SAMAudio, SAMAudioProcessor

        self._processor = SAMAudioProcessor.from_pretrained(model_name)
        self._model = SAMAudio.from_pretrained(model_name)
        self._model_name = model_name
        return self._model, self._processor

    def run_serial(self, request: InferenceRequest) -> None:
        payload: SeparationTaskPayload = request.payload
        tmp_path = f"/tmp/separation_{time.time()}_{request.request_id}.wav"
        audio_write(tmp_path, payload.audio, payload.sample_rate)

        try:
            model, processor = self._get_resources(payload.model_name)
            batch = processor(
                descriptions=[payload.description],
                audios=[tmp_path],
            )

            step_size = 2 / (payload.steps * 2)
            ode_opt = {"method": payload.method, "step_size": step_size}

            result = model.separate_long(
                audios=batch.audios,
                descriptions=batch.descriptions,
                anchor_ids=batch.anchor_ids,
                anchor_alignment=batch.anchor_alignment,
                ode_opt=ode_opt,
                ode_decode_chunk_size=50,
            )

            mx.clear_cache()

            target_audio = np.array(result.target[0])
            residual_audio = np.array(result.residual[0])
            sample_rate = model.sample_rate

            def audio_to_base64(audio_array, sr):
                buffer = io.BytesIO()
                audio_write(buffer, audio_array, sr, format="wav")
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode("utf-8")

            request.emit_data(
                SeparationResponse(
                    target=audio_to_base64(target_audio, sample_rate),
                    residual=audio_to_base64(residual_audio, sample_rate),
                    sample_rate=sample_rate,
                )
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        request.emit_done()


def get_inference_broker() -> InferenceBroker:
    global INFERENCE_BROKER
    if INFERENCE_BROKER is None:
        broker = InferenceBroker()
        broker.register_adapter("stt", STTExecutionAdapter())
        broker.register_adapter("tts", TTSExecutionAdapter())
        broker.register_adapter("separation", SeparationExecutionAdapter())
        INFERENCE_BROKER = broker
    return INFERENCE_BROKER


async def _next_inference_chunk(handle: InferenceHandle) -> InferenceResultChunk:
    return await asyncio.to_thread(handle.result_queue.get)


async def _stream_inference_results(handle: InferenceHandle, request: Request):
    try:
        while True:
            chunk = await _next_inference_chunk(handle)
            if chunk.kind == "done":
                break
            if chunk.kind == "error":
                raise chunk.error
            yield chunk.payload
            await asyncio.sleep(0)
            if await request.is_disconnected():
                handle.cancel()
                break
    finally:
        handle.cancel()


async def _await_inference_result(handle: InferenceHandle):
    result = None
    try:
        while True:
            chunk = await _next_inference_chunk(handle)
            if chunk.kind == "done":
                return result
            if chunk.kind == "error":
                raise chunk.error
            result = chunk.payload
    finally:
        handle.cancel()


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


@app.post("/v1/audio/speech")
async def tts_speech(payload: SpeechRequest, request: Request):
    """Generate speech audio following the OpenAI text-to-speech API."""
    if payload.ref_audio and isinstance(payload.ref_audio, str):
        if not os.path.exists(payload.ref_audio):
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio file not found: {payload.ref_audio}",
            )

    handle = get_inference_broker().submit(
        endpoint_kind="tts",
        model_name=payload.model,
        payload=SpeechTaskPayload(request=payload),
        normalized_kwargs=payload.model_dump(exclude={"model"}, exclude_none=True),
        stream=payload.stream,
    )
    return StreamingResponse(
        _stream_inference_results(handle, request),
        media_type=f"audio/{payload.response_format}",
        headers={
            "Content-Disposition": f"attachment; filename=speech.{payload.response_format}"
        },
    )


@app.post("/v1/audio/transcriptions")
async def stt_transcriptions(
    request: Request,
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
    response_format: str = Form("ndjson"),
):
    """Transcribe audio using an STT model.

    The default ``response_format`` (``ndjson``) preserves mlx-audio's native
    ``application/x-ndjson`` streaming transport, where each line is a JSON
    object emitted by the underlying STT model (text deltas while streaming,
    or the full whisper segment payload for batch transcription).

    For OpenAI Audio API compatibility, ``response_format`` also accepts:

    * ``text`` -- ``text/plain`` body with the final transcript only.
    * ``json`` -- ``application/json`` body shaped ``{"text": "..."}``.
    * ``verbose_json`` -- ``application/json`` body with the full payload
      from the underlying model (``text``, ``segments``, ``language``,
      ...), passed through unchanged.

    See https://platform.openai.com/docs/api-reference/audio/createTranscription
    """
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

    handle = get_inference_broker().submit(
        endpoint_kind="stt",
        model_name=payload.model,
        payload=TranscriptionTaskPayload(
            request=payload,
            filename=file.filename or "audio.mp3",
            audio=audio,
            sample_rate=sr,
        ),
        normalized_kwargs=payload.model_dump(exclude={"model"}, exclude_none=True),
        stream=payload.stream,
    )

    if response_format in ("text", "json", "verbose_json"):
        full: Optional[dict] = None
        accumulated = ""
        try:
            while True:
                chunk = await _next_inference_chunk(handle)
                if chunk.kind == "done":
                    break
                if chunk.kind == "error":
                    raise chunk.error
                line = chunk.payload
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                for raw in str(line).splitlines():
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and (
                        "segments" in obj or "language" in obj
                    ):
                        full = obj
                    elif isinstance(obj, dict) and "text" in obj:
                        accumulated += obj.get("text") or ""
        finally:
            handle.cancel()

        if full is None:
            full = {"text": accumulated}

        if response_format == "text":
            return PlainTextResponse((full.get("text") or "").strip())
        if response_format == "json":
            return JSONResponse({"text": (full.get("text") or "").strip()})
        # verbose_json: full payload (text, segments, language, ...) as-is
        return JSONResponse(full)

    return StreamingResponse(
        _stream_inference_results(handle, request),
        media_type="application/x-ndjson",
    )


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
    data = await file.read()
    tmp = io.BytesIO(data)
    audio, sr = audio_read(tmp, always_2d=False)
    tmp.close()

    handle = get_inference_broker().submit(
        endpoint_kind="separation",
        model_name=model,
        payload=SeparationTaskPayload(
            model_name=model,
            audio=audio,
            sample_rate=sr,
            description=description,
            method=method,
            steps=steps,
        ),
        normalized_kwargs={
            "description": description,
            "method": method,
            "steps": steps,
        },
    )
    return await _await_inference_result(handle)


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

    def start_server(self, host="localhost", port=8000, reload=False, realtime=False):
        if self.start_ui:
            self.start_ui_background()
            time.sleep(2)
            webbrowser.open("http://localhost:3000")
            print(f"✓ API server starting on http://{host}:{port}")
            print("✓ Studio UI available at http://localhost:3000")
            print("\nPress Ctrl+C to stop both servers")
        elif realtime:
            print(f"✓ Realtime server starting on http://{host}:{port}")
            print("✓ Standard endpoints remain mounted; prefer realtime endpoints.")

        try:
            uvicorn.run(
                "mlx_audio.server:app",
                host=host,
                port=port,
                reload=reload,
                workers=1,
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
        help="Enable auto-reload of the server.",
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
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Start the server for /v1/realtime usage.",
    )
    parser.add_argument(
        "--tts-max-batch-size",
        type=int,
        default=None,
        help=(
            "Maximum compatible TTS speech requests per continuous batch session. "
            "Overrides $MLX_AUDIO_TTS_MAX_BATCH_SIZE."
        ),
    )

    args = parser.parse_args()
    if args.realtime_model:
        os.environ["MLX_AUDIO_REALTIME_MODEL"] = args.realtime_model
    if args.realtime_transcription_delay_ms is not None:
        os.environ["MLX_AUDIO_REALTIME_TRANSCRIPTION_DELAY_MS"] = str(
            args.realtime_transcription_delay_ms
        )
    if args.tts_max_batch_size is not None:
        os.environ["MLX_AUDIO_TTS_MAX_BATCH_SIZE"] = str(args.tts_max_batch_size)

    setup_cors(app, args.allowed_origins)

    client = MLXAudioStudioServer(start_ui=args.start_ui, log_dir=args.log_dir)
    client.start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        realtime=args.realtime,
    )


if __name__ == "__main__":
    main()
