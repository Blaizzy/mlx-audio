
"""Patched MLX‑Audio server ready for FastRTC 0.0.28+

Usage
-----
uvicorn server_fixed:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Tuple

import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from mlx_audio import tts, stt  # type: ignore
from fastrtc import FastRTC
from fastrtc.utils import WebRTCData, Stream

##### ------------------------------------------------------------------ globals
ROOT = Path.home() / ".mlx_audio" / "outputs"
ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"
DEFAULT_VOICE = "af_heart"

stt_model = stt.load_model("openai/whisper-tiny")
tts_model = tts.load_model(DEFAULT_MODEL)

app = FastAPI()
rtc = FastRTC(app)

##### ---------------------------------------------------------------- endpoints
class TTSPayload(BaseModel):
    text: str
    voice: str | None = None
    speed: float = 1.0


@app.post("/tts")
async def tts_endpoint(payload: TTSPayload):
    wav_name = await generate_tts(
        text=payload.text,
        voice=payload.voice or DEFAULT_VOICE,
        speed=payload.speed,
    )
    return {"filename": wav_name}


@app.get("/audio/{fname}")
async def get_audio(fname: str):
    return FileResponse(ROOT / fname, media_type="audio/wav")


##### --------------------------------------------------------- helper functions
async def generate_tts(text: str, voice: str, speed: float) -> str:
    for out in tts_model.generate(text=text, voice=voice, speed=speed):
        sr, audio = out
    wav_name = f"tts_{asyncio.time.time():.0f}.wav"
    wav_path = ROOT / wav_name
    tts.save_wav(audio, sr, wav_path)
    return wav_name


##### --------------------------------------------------- speech‑to‑speech stream
@rtc.stream("speech_to_speech")
def speech_to_speech_handler(
    packet: WebRTCData,
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
    model: str = DEFAULT_MODEL,
):
    """Convert incoming mic audio → text → TTS audio.

    FastRTC passes `packet` (wrapper with .audio) + any extra inputs sent by the
    browser.  `model` now has a default, so the handler never raises if the UI
    hasn’t sent `model` yet.
    """
    sr, pcm16 = packet.audio  # tuple[int, ndarray]

    text = stt_model.stt((sr, pcm16))
    for sr_out, audio_out in tts.load_model(model).generate(
        text=text, voice=voice, speed=speed
    ):
        yield (sr_out, audio_out)


##### -------------------------------------------------------------- root route
@app.get("/")
def index():
    return {
        "message": "MLX‑Audio speech‑to‑speech demo – POST /webrtc/offer then stream"
    }
