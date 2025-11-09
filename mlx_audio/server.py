"""Main module for MLX Audio API server.

This module provides a FastAPI-based server for hosting MLX Audio models,
including Text-to-Speech (TTS), Speech-to-Text (STT), and Speech-to-Speech (S2S) models.
It offers an OpenAI-compatible API for Audio completions and model management.
"""

import argparse
import asyncio
import importlib
import io
import math
import os
import sys
import shutil
import time
from numbers import Real
from textwrap import dedent
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from mlx_audio.utils import load_model


def configure_fugashi_dictionary():
    """
    Ensure fugashi (used by misaki Cutlet) always has a dictionary.

    Prefer unidic_lite, fall back to unidic if present.
    """

    dicdir = None
    chosen_module = None
    for module_name in ("unidic_lite", "unidic"):
        try:
            module = __import__(module_name)
            dicdir = getattr(module, "DICDIR", None)
            if dicdir and os.path.isdir(dicdir):
                mecabrc_path = os.path.join(dicdir, "mecabrc")
                if os.path.exists(mecabrc_path):
                    chosen_module = module
                    break
                dicdir = None
        except ImportError:
            continue

    if not dicdir:
        return

    # Ensure any code importing `unidic` sees a module whose DICDIR actually exists.
    if chosen_module.__name__ == "unidic_lite":
        try:
            import unidic  # type: ignore

            unidic.DICDIR = dicdir  # type: ignore[attr-defined]
        except ImportError:
            sys.modules["unidic"] = chosen_module

    mecabrc_path = os.path.join(dicdir, "mecabrc")

    os.environ["FUGASHI_DICDIR"] = dicdir
    os.environ["FUGASHI_ARGS"] = f"-d {dicdir} -r {mecabrc_path}"
    if os.path.exists(mecabrc_path):
        os.environ["MECABRC"] = mecabrc_path


configure_fugashi_dictionary()

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


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Simple landing page so the root URL no longer returns a 404. The full GUI
    lives in the Next.js project under ``mlx_audio/ui``.
    """

    html = dedent(
        """
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8" />
                <title>MLX Audio Server</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
                        margin: 0;
                        padding: 2rem;
                        background: #0b1120;
                        color: #e2e8f0;
                    }
                    a {
                        color: #38bdf8;
                        text-decoration: none;
                    }
                    .card {
                        max-width: 720px;
                        margin: auto;
                        padding: 2rem;
                        border-radius: 1rem;
                        background: rgba(15, 23, 42, 0.85);
                        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
                    }
                    h1 {
                        margin-top: 0;
                    }
                    code {
                        background: rgba(15, 118, 110, 0.25);
                        padding: 0.2rem 0.4rem;
                        border-radius: 0.35rem;
                    }
                </style>
            </head>
            <body>
                <main class="card">
                    <h1>MLX Audio API server is running</h1>
                    <p>
                        This endpoint exposes the FastAPI backend (OpenAI-compatible TTS / STT APIs).
                        Open <a href="/docs">/docs</a> for interactive API documentation
                        or <a href="/redoc">/redoc</a> for the ReDoc view.
                    </p>
                    <p>
                        Looking for the graphical interface? Start the Next.js app inside
                        <code>mlx_audio/ui</code>:
                    </p>
                    <pre><code>cd mlx_audio/ui
npm install
export NEXT_PUBLIC_API_BASE_URL=http://localhost
export NEXT_PUBLIC_API_PORT=8000
npm run dev</code></pre>
                    <p>
                        Then open <a href="http://localhost:3000" target="_blank" rel="noreferrer">
                        http://localhost:3000</a>.
                    </p>
                </main>
            </body>
        </html>
        """
    )

    return HTMLResponse(content=html)


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


LANGUAGE_DEPENDENCIES = {
    "j": ("misaki.ja", "pip install misaki[ja]"),
    "z": ("misaki.zh", "pip install misaki[zh]"),
}


def ensure_tts_language_support(lang_code: Optional[str]):
    """Ensure extra dependencies for specific languages are installed."""
    code = (lang_code or "").lower()
    requirement = LANGUAGE_DEPENDENCIES.get(code)
    if requirement is None:
        return

    module_name, install_hint = requirement
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = module_name.split(".")[0]
        raise HTTPException(
            status_code=503,
            detail=(
                f"Language '{code}' requires `{install_hint}` "
                f"(missing dependency: {missing})."
            ),
        ) from exc


def sanitize_jsonable(value: Any):
    """
    Recursively sanitize any jsonable structure by replacing NaN/inf values
    with ``None`` so the response can be serialized by the JSON encoder.
    """

    if isinstance(value, dict):
        return {k: sanitize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_jsonable(v) for v in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return value


# Request schemas for OpenAI-compatible endpoints
class SpeechRequest(BaseModel):
    model: str
    input: str
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


# Initialize the ModelProvider
model_provider = ModelProvider()


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


async def generate_audio(model, payload: SpeechRequest, verbose: bool = False):
    for result in model.generate(
        payload.input,
        voice=payload.voice,
        speed=payload.speed,
        gender=payload.gender,
        pitch=payload.pitch,
        lang_code=payload.lang_code,
        ref_audio=payload.ref_audio,
        ref_text=payload.ref_text,
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        repetition_penalty=payload.repetition_penalty,
    ):

        sample_rate = result.sample_rate
        buffer = io.BytesIO()
        sf.write(buffer, result.audio, sample_rate, format="WAV")
        buffer.seek(0)
        yield buffer.getvalue()


@app.post("/v1/audio/speech")
async def tts_speech(payload: SpeechRequest):
    """Generate speech audio following the OpenAI text-to-speech API."""
    ensure_tts_language_support(payload.lang_code)
    model = model_provider.load_model(payload.model)
    return StreamingResponse(
        generate_audio(model, payload),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


@app.post("/v1/audio/transcriptions")
async def stt_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
):
    """Transcribe audio using an STT model in OpenAI format."""
    data = await file.read()
    tmp = io.BytesIO(data)
    audio, sr = sf.read(tmp, always_2d=False)
    tmp.close()
    tmp_path = f"/tmp/{time.time()}.wav"
    sf.write(tmp_path, audio, sr)

    stt_model = model_provider.load_model(model)
    generate_kwargs = {}
    if language:
        generate_kwargs["language"] = language
    result = stt_model.generate(tmp_path, **generate_kwargs)
    os.remove(tmp_path)
    response = sanitize_jsonable(jsonable_encoder(result))
    return response


def main():
    parser = argparse.ArgumentParser(description="MLX Audio API server")
    parser.add_argument(
        "--allowed-origins",
        nargs="+",
        default=["*"],
        help="List of allowed origins for CORS",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
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
        default=calculate_default_workers(),
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

    args = parser.parse_args()
    if isinstance(args.workers, float):
        args.workers = max(1, int(os.cpu_count() * args.workers))

    setup_cors(app, args.allowed_origins)

    uvicorn.run(
        "mlx_audio.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()


def main():
    parser = argparse.ArgumentParser(description="MLX Audio API server")
    parser.add_argument(
        "--allowed-origins",
        nargs="+",
        default=["*"],
        help="List of allowed origins for CORS",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
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
        default=calculate_default_workers(),
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

    args = parser.parse_args()
    if isinstance(args.workers, float):
        args.workers = max(1, int(os.cpu_count() * args.workers))

    setup_cors(app, args.allowed_origins)

    uvicorn.run(
        "mlx_audio.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
