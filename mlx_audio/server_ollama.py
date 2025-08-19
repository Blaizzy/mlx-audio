import argparse
import importlib.util
import logging
import os
import sys
import tempfile
import os
import uuid

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastrtc import ReplyOnPause, Stream, get_stt_model
from numpy.typing import NDArray
from pydantic import BaseModel


# Configure logging
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if verbose:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger("mlx_audio_server")


# Force Hugging Face to offline mode once at import time
os.environ.setdefault("HF_HUB_OFFLINE", "1")

logger = setup_logging()  # Will be updated with verbose setting in main()

from mlx_audio.tts.generate import main as generate_main

# Import from mlx_audio package
from mlx_audio.tts.utils import load_model

from .tts.audio_player import AudioPlayer

# === LLM chat backend (Ollama) ===
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4:latest")
# System / character prompt for the chat model. Accept either env var name.
SYSTEM_PROMPT = os.getenv("OLLAMA_SYSTEM_PROMPT") or os.getenv("LLAMA_SYSTEM_PROMPT") or "You are an empathetic concise voice assistant."


def chat_llm(prompt: str) -> str:
    """Return a conversational reply from the local Ollama server.
    Falls back to echoing the prompt on error."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/chat/completions",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return prompt  # graceful fallback

app = FastAPI()

# Serve local static assets (e.g., Three.js) at /static
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Ensure models load when uvicorn imports this module
@app.on_event("startup")
async def _startup_load_models() -> None:
    """Load TTS/STT models once when the FastAPI app starts."""
    setup_server()

# Add CORS middleware to allow requests from the same origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, will be restricted by host binding
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once on server startup.
# You can change the model path or pass arguments as needed.
# For performance, load once globally:
tts_model = None  # Will be loaded when the server starts
audio_player = None  # Will be initialized when the server starts
stt_model = get_stt_model()
# Make sure the output folder for generated TTS files exists
# Use an absolute path that's guaranteed to be writable
OUTPUT_FOLDER = os.path.join(os.path.expanduser("~"), ".mlx_audio", "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logger.debug(f"Using output folder: {OUTPUT_FOLDER}")

# Let users override the initial TTS model via environment variable.
# Provide a sensible fallback (4-bit HF repo) so nothing breaks.
DEFAULT_MODEL = os.getenv("TTS_DEFAULT_MODEL", "mlx-community/Kokoro-82M-4bit")

def speech_to_speech_handler(
    audio: tuple[int, NDArray[np.int16]],
    voice: str,
    speed: float,
    model: str = DEFAULT_MODEL,  # make optional – FastRTC may omit it
):
    global tts_model

    # Log raw parameters for easier debugging
    logger.info(
        f"Speech-to-speech raw params: voice={voice} | model={model} | speed={speed}"
    )

    # Heuristic 1 – argument shift: `voice` actually came in as the numeric speed
    if isinstance(voice, float):
        speed, model, voice = voice, speed, model or DEFAULT_MODEL

    else:
        # Heuristic 2 – mistaken swap detection
        voice_looks_like_model = any(t in voice for t in ("/", "mlx-community", "Kokoro"))
        model_looks_like_model = any(t in model for t in ("/", "mlx-community", "Kokoro"))

        # Swap only if *voice* looks like a model path AND *model* does not
        if voice_looks_like_model and not model_looks_like_model:
            voice, model = model, voice

    # Final safeguard – if voice still appears to be a model path, use a safe default
    if any(t in voice for t in ("/", "mlx-community", "Kokoro")):
        logger.warning(
            f"Invalid voice '{voice}' detected; falling back to 'af_heart'."
        )
        voice = "af_heart"

    if tts_model is None:
        tts_model = load_model(model)

    # 1) Speech-to-text
    user_text = stt_model.stt(audio)
    logger.info(f"STT: '{user_text}'")

    # 2) Chat model reply
    reply_text = chat_llm(user_text)
    logger.info(f"LLM reply: '{reply_text}'")

    # 3) Text-to-speech
    for segment in tts_model.generate(
        text=reply_text,
        voice=voice,
        speed=speed,
        verbose=False,
    ):
        yield (24_000, np.array(segment.audio, copy=False))
        yield (24_000, np.zeros(2_400, dtype=np.float32))


stream = Stream(
    ReplyOnPause(speech_to_speech_handler, output_sample_rate=24_000),
    mode="send-receive",
    modality="audio",
)
stream.mount(app)


class SpeechToSpeechArgs(BaseModel):
    voice: str
    speed: float
    model: str
    webrtc_id: str


@app.post("/speech_to_speech_input")
def speech_to_speech_endpoint(args: SpeechToSpeechArgs):
    # FastRTC appears to rotate the extra arguments: the handler receives (arg1→ignored, arg2→voice, arg3→speed, arg1→model).
    # Empirically, sending them as (model, voice, speed) makes the handler receive
    #   voice = voice,
    #   speed = speed,
    #   model = model.
    stream.set_input(args.webrtc_id, args.model, args.voice, args.speed)
    return {"status": "success"}


@app.post("/tts")
def tts_endpoint(
    text: str = Form(...),
    voice: str = Form("af_heart"),
    speed: float = Form(1.0),
    model: str = Form("mlx-community/Kokoro-82M-4bit"),
):
    """
    POST an x-www-form-urlencoded form with 'text' (and optional 'voice', 'speed', and 'model').
    We run TTS on the text, save the audio in a unique file,
    and return JSON with the filename so the client can retrieve it.
    """
    global tts_model

    if not text.strip():
        return JSONResponse({"error": "Text is empty"}, status_code=400)

    # Validate speed parameter
    try:
        speed_float = float(speed)
        if speed_float < 0.5 or speed_float > 2.0:
            return JSONResponse(
                {"error": "Speed must be between 0.5 and 2.0"}, status_code=400
            )
    except ValueError:
        return JSONResponse({"error": "Invalid speed value"}, status_code=400)

    # Validate model parameter
    valid_models = [
        "mlx-community/Kokoro-82M-4bit",
        "mlx-community/Kokoro-82M-6bit",
        "mlx-community/Kokoro-82M-8bit",
        "mlx-community/Kokoro-82M-bf16",
    ]
    if model not in valid_models:
        return JSONResponse(
            {"error": f"Invalid model. Must be one of: {', '.join(valid_models)}"},
            status_code=400,
        )

    # Store current model repo_id for comparison
    current_model_repo_id = (
        getattr(tts_model, "repo_id", None) if tts_model is not None else None
    )

    # Load the model if it's not loaded or if a different model is requested
    if tts_model is None or current_model_repo_id != model:
        try:
            logger.debug(f"Loading TTS model from {model}")
            tts_model = load_model(model)
            logger.debug("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading TTS model: {str(e)}")
            return JSONResponse(
                {"error": f"Failed to load model: {str(e)}"}, status_code=500
            )

    # We'll do something like the code in model.generate() from the TTS library:
    # Generate the unique filename
    unique_id = str(uuid.uuid4())
    filename = f"tts_{unique_id}.wav"
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    logger.debug(
        f"Generating TTS for text: '{text[:50]}...' with voice: {voice}, speed: {speed_float}, model: {model}"
    )
    logger.debug(f"Output file will be: {output_path}")

    # We'll use the high-level "model.generate" method:
    results = tts_model.generate(
        text=text,
        voice=voice,
        speed=speed_float,
        lang_code=voice[0],
        verbose=False,
    )

    # We'll just gather all segments (if any) into a single wav
    # It's typical for multi-segment text to produce multiple wave segments:
    audio_arrays = []
    for segment in results:
        audio_arrays.append(segment.audio)

    # If no segments, return error
    if not audio_arrays:
        logger.error("No audio segments generated")
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    # Concatenate all segments
    cat_audio = np.concatenate(audio_arrays, axis=0)

    # Write the audio as a WAV
    try:
        sf.write(output_path, cat_audio, 24000)
        logger.debug(f"Successfully wrote audio file to {output_path}")

        # Verify the file exists
        if not os.path.exists(output_path):
            logger.error(f"File was not created at {output_path}")
            return JSONResponse(
                {"error": "Failed to create audio file"}, status_code=500
            )

        # Check file size
        file_size = os.path.getsize(output_path)
        logger.debug(f"File size: {file_size} bytes")

        if file_size == 0:
            logger.error("File was created but is empty")
            return JSONResponse(
                {"error": "Generated audio file is empty"}, status_code=500
            )

    except Exception as e:
        logger.error(f"Error writing audio file: {str(e)}")
        return JSONResponse(
            {"error": f"Failed to save audio: {str(e)}"}, status_code=500
        )

    return {"filename": filename}


@app.get("/audio/{filename}")
def get_audio_file(filename: str):
    """
    Return an audio file from the outputs folder.
    The user can GET /audio/<filename> to fetch the WAV file.
    """
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    logger.debug(f"Requested audio file: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        # List files in the directory to help debug
        try:
            files = os.listdir(OUTPUT_FOLDER)
            logger.debug(f"Files in output directory: {files}")
        except Exception as e:
            logger.error(f"Error listing output directory: {str(e)}")

        return JSONResponse({"error": "File not found"}, status_code=404)

    logger.debug(f"Serving audio file: {file_path}")
    return FileResponse(file_path, media_type="audio/wav")


@app.get("/")
def root():
    """
    Serve the audio_player.html page or a fallback HTML if not found
    """
    try:
        # Try to find the audio_player.html file in the package
        static_dir = find_static_dir()
        audio_player_path = os.path.join(static_dir, "audio_player.html")
        return FileResponse(audio_player_path)
    except Exception as e:
        # If there's an error, return a simple HTML page with error information
        return HTMLResponse(
            content=f"""
            <html>
                <head><title>MLX-Audio TTS Server</title></head>
                <body>
                    <h1>MLX-Audio TTS Server</h1>
                    <p>The server is running, but the web interface could not be loaded.</p>
                    <p>Error: {str(e)}</p>
                    <h2>API Endpoints</h2>
                    <ul>
                        <li><code>POST /tts</code> - Generate TTS audio</li>
                        <li><code>GET /audio/{{filename}}</code> - Retrieve generated audio file</li>
                    </ul>
                </body>
            </html>
            """,
            status_code=200,
        )


def find_static_dir():
    """Find the static directory containing HTML files."""
    # Try different methods to find the static directory

    # Method 1: Use importlib.resources (Python 3.9+)
    try:
        import importlib.resources as pkg_resources

        static_dir = pkg_resources.files("mlx_audio").joinpath("tts")
        static_dir_str = str(static_dir)
        if os.path.exists(static_dir_str):
            return static_dir_str
    except (ImportError, AttributeError):
        pass

    # Method 2: Use importlib_resources (Python 3.8)
    try:
        import importlib_resources

        static_dir = importlib_resources.files("mlx_audio").joinpath("tts")
        static_dir_str = str(static_dir)
        if os.path.exists(static_dir_str):
            return static_dir_str
    except ImportError:
        pass

    # Method 3: Use pkg_resources
    try:
        static_dir_str = pkg_resources.resource_filename("mlx_audio", "tts")
        if os.path.exists(static_dir_str):
            return static_dir_str
    except (ImportError, pkg_resources.DistributionNotFound):
        pass

    # Method 4: Try to find the module path directly
    try:
        module_spec = importlib.util.find_spec("mlx_audio")
        if module_spec and module_spec.origin:
            package_dir = os.path.dirname(module_spec.origin)
            static_dir_str = os.path.join(package_dir, "tts")
            if os.path.exists(static_dir_str):
                return static_dir_str
    except (ImportError, AttributeError):
        pass

    # Method 5: Look in sys.modules
    try:
        if "mlx_audio" in sys.modules:
            module = sys.modules["mlx_audio"]
            if hasattr(module, "__file__"):
                package_dir = os.path.dirname(module.__file__)
                static_dir_str = os.path.join(package_dir, "tts")
                if os.path.exists(static_dir_str):
                    return static_dir_str
    except Exception:
        pass

    # If all methods fail, raise an error
    raise RuntimeError("Could not find static directory")


@app.post("/play")
def play_audio(filename: str = Form(...)):
    """
    Play audio directly from the server using the AudioPlayer.
    Expects a filename that exists in the OUTPUT_FOLDER.
    """
    global audio_player

    if audio_player is None:
        return JSONResponse({"error": "Audio player not initialized"}, status_code=500)

    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    try:
        # Load the audio file
        audio_data, sample_rate = sf.read(file_path)

        # If audio is stereo, convert to mono
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)

        # Queue the audio for playback
        audio_player.queue_audio(audio_data)

        return {"status": "playing", "filename": filename}
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to play audio: {str(e)}"}, status_code=500
        )


@app.post("/stop")
def stop_audio():
    """
    Stop any currently playing audio.
    """
    global audio_player

    if audio_player is None:
        return JSONResponse({"error": "Audio player not initialized"}, status_code=500)

    try:
        audio_player.stop()
        return {"status": "stopped"}
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to stop audio: {str(e)}"}, status_code=500
        )


@app.post("/open_output_folder")
def open_output_folder():
    """
    Open the output folder in the system file explorer (Finder on macOS).
    This only works when running on localhost for security reasons.
    """
    global OUTPUT_FOLDER

    # Check if the request is coming from localhost
    # Note: In a production environment, you would want to check the request IP

    try:
        # For macOS (Finder)
        if sys.platform == "darwin":
            os.system(f"open {OUTPUT_FOLDER}")
        # For Windows (Explorer)
        elif sys.platform == "win32":
            os.system(f"explorer {OUTPUT_FOLDER}")
        # For Linux (various file managers)
        elif sys.platform == "linux":
            os.system(f"xdg-open {OUTPUT_FOLDER}")
        else:
            return JSONResponse(
                {"error": f"Unsupported platform: {sys.platform}"}, status_code=500
            )

        logger.debug(f"Opened output folder: {OUTPUT_FOLDER}")
        return {"status": "opened", "path": OUTPUT_FOLDER}
    except Exception as e:
        logger.error(f"Error opening output folder: {str(e)}")
        return JSONResponse(
            {"error": f"Failed to open output folder: {str(e)}"}, status_code=500
        )


def setup_server():
    """Setup the server by loading the model and creating the output directory."""
    global tts_model, audio_player, OUTPUT_FOLDER

    # Make sure the output folder for generated TTS files exists
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        # Test write permissions by creating a test file
        test_file = os.path.join(OUTPUT_FOLDER, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("Test write permissions")
        os.remove(test_file)
        logger.debug(f"Output directory {OUTPUT_FOLDER} is writable")
    except Exception as e:
        logger.error(f"Error with output directory {OUTPUT_FOLDER}: {str(e)}")
        # Try to use a fallback directory in /tmp
        fallback_dir = os.path.join("/tmp", "mlx_audio_outputs")
        logger.debug(f"Trying fallback directory: {fallback_dir}")
        try:
            os.makedirs(fallback_dir, exist_ok=True)
            OUTPUT_FOLDER = fallback_dir
            logger.debug(f"Using fallback output directory: {OUTPUT_FOLDER}")
        except Exception as fallback_error:
            logger.error(f"Error with fallback directory: {str(fallback_error)}")

    # Load the model if not already loaded
    if tts_model is None:
        try:
            default_model = (
                "mlx-community/Kokoro-82M-4bit"  # Same default as in tts_endpoint
            )
            logger.debug(f"Loading TTS model from {default_model}")
            tts_model = load_model(default_model)
            logger.debug("TTS model loaded successfully")

            # Point Kokoro pipeline to local repo dir so it fetches voices from disk
            try:
                local_repo = os.path.join(
                    os.path.dirname(__file__),
                    "tts",
                    "models",
                    "kokoro",
                    "Kokoro-82M",
                )
                if hasattr(tts_model, "pipeline"):
                    tts_model.pipeline.repo_id = local_repo
                    logger.debug(f"Kokoro pipeline repo_id set to local path: {local_repo}")
            except Exception as patch_err:
                logger.warning(f"Could not patch pipeline repo path: {patch_err}")
        except Exception as e:
            logger.error(f"Error loading TTS model: {str(e)}")
            raise

    # Initialize the audio player if not already initialized
    if audio_player is None:
        try:
            logger.debug("Initializing audio player")
            audio_player = AudioPlayer()
            logger.debug("Audio player initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing audio player: {str(e)}")

    # Try to mount the static files directory
    try:
        static_dir = find_static_dir()
        logger.debug(f"Found static directory: {static_dir}")
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.debug("Static files mounted successfully")
    except Exception as e:
        logger.error(f"Could not mount static files directory: {e}")
        logger.warning(
            "The server will still function, but the web interface may be limited."
        )


def main(host="127.0.0.1", port=8000, verbose=False):
    """Parse command line arguments for the server and start it."""
    parser = argparse.ArgumentParser(description="Start the MLX-Audio TTS server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging with detailed debug information",
    )
    args = parser.parse_args()

    # Update logger with verbose setting
    global logger
    logger = setup_logging(args.verbose)

    # Start the server with the parsed arguments
    setup_server()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.verbose else "info",
    )


if __name__ == "__main__":
    main()
