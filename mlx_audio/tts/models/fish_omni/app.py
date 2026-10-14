"""Local test UI for Drama 3 (needs the `server` extra: pip install -e ".[server]").

    python -m mlx_audio.tts.models.fish_omni.app \
        --models cguldogan/drama-3-mlx-q4 cguldogan/drama-3-mlx-q8 \
        --voice ref.wav "Transcript of ref.wav."

Then open http://127.0.0.1:7860. One model is loaded at a time; every MLX call
runs on a single worker thread.
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import itertools
import json
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from mlx_audio.audio_io import write as audio_write

PAGE = Path(__file__).with_name("app.html")
MAX_REFERENCE_SECONDS = 30.0


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    audio_write(buffer, audio, sample_rate, format="wav")
    return buffer.getvalue()


@dataclass
class Voice:
    id: str
    name: str
    speaker: int
    transcript: str
    seconds: float
    codes: mx.array

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "speaker": self.speaker,
            "transcript": self.transcript,
            "seconds": round(self.seconds, 2),
            "frames": int(self.codes.shape[1]),
        }


@dataclass
class Take:
    id: str
    number: int
    text: str
    model: str
    chunks: List[tuple] = field(default_factory=list)  # (chunk text, codes)
    audio: List[np.ndarray] = field(default_factory=list)
    settings: dict = field(default_factory=dict)


class Worker:
    """Owns the model; runs jobs one at a time on its own thread."""

    def __init__(self, models: Dict[str, str]):
        self.models = models  # display name -> local directory or Hub id
        self.model = None
        self.model_name: Optional[str] = None
        self.loading: Optional[str] = None
        self.voices: Dict[str, Voice] = {}
        self.takes: Dict[str, Take] = {}
        self.take_numbers = itertools.count(1)
        self.jobs: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self.busy = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            with self.busy:
                job()

    def call(self, fn: Callable[[], Any]) -> Any:
        """Run fn on the worker thread and wait for its result."""
        done: "queue.Queue" = queue.Queue()

        def job():
            try:
                done.put((True, fn()))
            except Exception as error:  # surfaced to the caller
                done.put((False, error))

        self.jobs.put(job)
        ok, value = done.get()
        if not ok:
            raise value
        return value

    # ------------------------------------------------------------- model

    def status(self) -> dict:
        return {
            "models": list(self.models),
            "loaded": self.model_name,
            "loading": self.loading,
            "busy": self.busy.locked(),
            "active_gb": round(mx.get_active_memory() / 1e9, 1),
            "voices": [v.public() for v in self.voices.values()],
        }

    def load(self, name: str) -> None:
        from mlx_audio.tts.utils import load_model
        from mlx_audio.utils import get_model_path

        if name not in self.models:
            raise ValueError(f"unknown model {name}")
        if name == self.model_name:
            return
        self.loading = name
        try:
            self.model, self.model_name = None, None
            gc.collect()
            mx.clear_cache()
            # Hub ids download here, on first load, not at startup.
            model = load_model(str(get_model_path(self.models[name])))
            mx.eval(model.parameters())
            self.model, self.model_name = model, name
        finally:
            self.loading = None

    def require_model(self):
        if self.model is None:
            raise ValueError("load a model first")
        return self.model

    # ------------------------------------------------------------ voices

    def add_voice(self, path: Path, name: str, transcript: str, speaker: int) -> dict:
        from .fish_omni import load_reference

        model = self.require_model()
        if not transcript.strip():
            raise ValueError("a reference voice needs its transcript")
        audio = load_reference(path, model.sample_rate)
        seconds = audio.shape[0] / model.sample_rate
        if seconds > MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"reference is {seconds:.0f}s; keep it under {MAX_REFERENCE_SECONDS:.0f}s"
            )
        codes = model.encode_reference(audio)
        mx.eval(codes)
        voice = Voice(
            uuid.uuid4().hex[:8], name, speaker, transcript.strip(), seconds, codes
        )
        self.voices[voice.id] = voice
        return voice.public()

    # -------------------------------------------------------- generation

    def generate(self, request: dict, events: "queue.Queue", cancel: threading.Event):
        model = self.require_model()
        voices = [self.voices[v] for v in request.get("voices", []) if v in self.voices]
        previous = self.takes.get(request.get("previous_take") or "")
        seed = request.get("seed")
        mx.random.seed(int(seed) if seed not in (None, "") else int(time.time()))
        settings = {
            "temperature": float(request.get("temperature", 1.0)),
            "top_p": float(request.get("top_p", 0.9)),
            "top_k": int(request.get("top_k", 30)),
            "max_tokens": int(request.get("max_frames", 600)),
            "single_chunk": bool(request.get("single_chunk", False)),
        }
        take = Take(
            uuid.uuid4().hex[:8],
            next(self.take_numbers),
            request["text"],
            self.model_name,
            settings=dict(
                settings,
                seed=seed,
                voices=[v.name for v in voices],
                previous=previous.number if previous else None,
            ),
        )
        events.put({"type": "start", "take": take.id, "number": take.number})
        started = time.perf_counter()
        results = model.generate(
            request["text"],
            ref_codes=[v.codes for v in voices],
            ref_texts=[f"<|speaker:{v.speaker}|> {v.transcript}" for v in voices],
            previous_takes=list(previous.chunks) if previous else None,
            **settings,
        )
        for result in results:
            audio = np.array(result.audio.astype(mx.float32))
            take.chunks.append((result.text, result.codes))
            take.audio.append(audio)
            events.put(
                {
                    "type": "chunk",
                    "index": result.segment_idx,
                    "text": result.text,
                    "frames": result.token_count,
                    "seconds": round(len(audio) / model.sample_rate, 2),
                    "elapsed": round(result.processing_time_seconds, 2),
                    "fps": round(
                        result.token_count / result.processing_time_seconds, 1
                    ),
                    "peak_gb": round(result.peak_memory_usage, 1),
                    "wav": base64.b64encode(
                        wav_bytes(audio, model.sample_rate)
                    ).decode(),
                }
            )
            if cancel.is_set():
                events.put({"type": "cancelled"})
                break
        if take.audio:
            self.takes[take.id] = take
        total = sum(len(a) for a in take.audio) / model.sample_rate
        events.put(
            {
                "type": "done",
                "take": take.id if take.audio else None,
                "seconds": round(total, 2),
                "elapsed": round(time.perf_counter() - started, 2),
            }
        )

    def take_wav(self, take_id: str) -> bytes:
        take = self.takes[take_id]
        return wav_bytes(np.concatenate(take.audio), 44100)


def create_app(worker: Worker) -> FastAPI:
    app = FastAPI(title="Drama 3")

    def guarded(fn):
        try:
            return worker.call(fn)
        except (ValueError, KeyError) as error:
            raise HTTPException(400, str(error))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.read_text()

    @app.get("/api/status")
    def status():
        return worker.status()

    @app.post("/api/load")
    def load(body: dict):
        guarded(lambda: worker.load(body["model"]))
        return worker.status()

    @app.post("/api/voices")
    async def add_voice(
        file: UploadFile = File(...),
        transcript: str = Form(...),
        speaker: int = Form(0),
        name: str = Form(""),
    ):
        suffix = Path(file.filename or "voice.wav").suffix or ".wav"
        data = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            path = Path(handle.name)
        try:
            return guarded(
                lambda: worker.add_voice(
                    path,
                    name or Path(file.filename or "voice").stem,
                    transcript,
                    speaker,
                )
            )
        finally:
            path.unlink(missing_ok=True)

    @app.patch("/api/voices/{voice_id}")
    def update_voice(voice_id: str, body: dict):
        voice = worker.voices.get(voice_id)
        if voice is None:
            raise HTTPException(404, "no such voice")
        voice.speaker = int(body.get("speaker", voice.speaker))
        voice.transcript = body.get("transcript", voice.transcript)
        return voice.public()

    @app.delete("/api/voices/{voice_id}")
    def delete_voice(voice_id: str):
        worker.voices.pop(voice_id, None)
        return {"ok": True}

    @app.post("/api/generate")
    def generate(body: dict):
        if not str(body.get("text", "")).strip():
            raise HTTPException(400, "enter some text")
        events: "queue.Queue" = queue.Queue()
        cancel = threading.Event()

        def job():
            try:
                worker.generate(body, events, cancel)
            except Exception as error:
                events.put(
                    {"type": "error", "message": f"{type(error).__name__}: {error}"}
                )
            finally:
                events.put(None)

        worker.jobs.put(job)

        def stream():
            try:
                while (event := events.get()) is not None:
                    yield json.dumps(event) + "\n"
            finally:
                # The client went away (Stop or a closed tab): end after this chunk.
                cancel.set()

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.get("/api/takes/{take_id}.wav")
    def take_wav(take_id: str):
        if take_id not in worker.takes:
            raise HTTPException(404, "no such take")
        data = worker.take_wav(take_id)  # numpy only; no need to queue behind MLX
        return Response(
            data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="drama3-take-{take_id}.wav"'
            },
        )

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Drama 3 test UI")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="converted model dirs or Hub ids, e.g. cguldogan/drama-3-mlx-q4",
    )
    parser.add_argument(
        "--voice",
        nargs=2,
        action="append",
        default=[],
        metavar=("WAV", "TRANSCRIPT"),
        help="preload a reference voice (repeatable; speakers 0, 1, ... in order)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--no-preload", action="store_true", help="don't load the first model"
    )
    args = parser.parse_args()

    models = {}
    for spec in args.models:
        local = Path(spec).expanduser()
        if local.exists():
            if not (local / "config.json").exists():
                parser.error(f"{spec} is not a converted Drama 3 model directory")
            spec = str(local)
        # Anything else is taken as a Hub id and downloaded when first loaded.
        models[Path(spec.rstrip("/")).name] = spec
    worker = Worker(models)
    first = next(iter(models))
    if not args.no_preload:
        print(f"loading {first} ...", flush=True)
        worker.call(lambda: worker.load(first))
        for speaker, (wav, transcript) in enumerate(args.voice):
            wav = Path(wav).expanduser()
            worker.call(lambda: worker.add_voice(wav, wav.stem, transcript, speaker))
    print(f"Drama 3 UI: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(worker), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
