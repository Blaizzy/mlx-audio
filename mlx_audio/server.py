import argparse
import importlib.util
import logging
import os
import sys
import uuid

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import list_repo_files


# Configure logging
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if verbose:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger("mlx_audio_server")


logger = setup_logging()  # Will be updated with verbose setting in main()

from mlx_audio.tts.generate import main as generate_main

# Import from mlx_audio package
from mlx_audio.tts.utils import load_model

from .tts.audio_player import AudioPlayer

app = FastAPI()

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

# Make sure the output folder for generated TTS files exists
# Use an absolute path that's guaranteed to be writable
OUTPUT_FOLDER = os.path.join(os.path.expanduser("~"), ".mlx_audio", "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logger.debug(f"Using output folder: {OUTPUT_FOLDER}")


@app.post("/tts")
def tts_endpoint(
    text: str = Form(...),
    voice: str = Form("af_heart"),
    speed: float = Form(1.0),
    model: str = Form("mlx-community/Kokoro-82M-4bit"),
    language: str = Form("american_english"),
):
    """
    POST an x-www-form-urlencoded form with 'text' (and optional 'voice', 'speed', 'model', and 'language').
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
        "mlx-community/orpheus-3b-0.1-ft-bf16",
        "mlx-community/orpheus-3b-0.1-ft-8bit",
        "mlx-community/orpheus-3b-0.1-ft-6bit",
        "mlx-community/orpheus-3b-0.1-ft-4bit",
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
        f"Generating TTS for text: '{text[:50]}...' with voice: {voice}, speed: {speed_float}, model: {model}, language: {language}"
    )
    logger.debug(f"Output file will be: {output_path}")

    # Map language parameter to language code
    language_to_code = {
        "american_english": "a",
        "british_english": "b",
        "hindi": "h",
        "spanish": "s",
        "french": "f",
        "italian": "i",
        "brazilian_portuguese": "p",
        "japanese": "j",
        "mandarin_chinese": "c",
    }

    # Set language code based on model type
    # For Orpheus models, always use "a" (American English)
    # For other models, use the language mapping
    if "orpheus" in model.lower():
        lang_code = "a"  # Always use American English for Orpheus
    else:
        # Use language code from mapping, or fall back to first char of voice
        lang_code = language_to_code.get(language, voice[0])

    # We'll use the high-level "model.generate" method:
    results = tts_model.generate(
        text=text,
        voice=voice,
        speed=speed_float,
        lang_code=lang_code,
        verbose=False,
    )

    # We'll just gather all segments (if any) into a single wav
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


def get_voice_names(repo_id):
    """Fetches and returns a list of voice names (without extensions) from the given Hugging Face repository."""
    return [
        os.path.splitext(file.replace("voices/", ""))[0]
        for file in list_repo_files(repo_id)
        if file.startswith("voices/")
    ]


# Global variable to store the available voices
available_voices = []

# List of supported models
available_models = [
    {"id": "mlx-community/Kokoro-82M-4bit", "name": "Kokoro 82M 4bit"},
    {"id": "mlx-community/Kokoro-82M-6bit", "name": "Kokoro 82M 6bit"},
    {"id": "mlx-community/Kokoro-82M-8bit", "name": "Kokoro 82M 8bit"},
    {"id": "mlx-community/Kokoro-82M-bf16", "name": "Kokoro 82M bf16"},
    {"id": "mlx-community/orpheus-3b-0.1-ft-bf16", "name": "Orpheus 3B bf16"},
    {"id": "mlx-community/orpheus-3b-0.1-ft-8bit", "name": "Orpheus 3B 8bit"},
    {"id": "mlx-community/orpheus-3b-0.1-ft-6bit", "name": "Orpheus 3B 6bit"},
    {"id": "mlx-community/orpheus-3b-0.1-ft-4bit", "name": "Orpheus 3B 4bit"},
]


@app.get("/voices")
def get_voices(repo_id: str = "hexgrad/Kokoro-82M", language: str = None):
    """
    Return a list of available voice names.
    If language parameter is provided, filter voices starting with that language code.
    """
    global available_voices

    # For orpheus models, return a fixed list of voices
    if "orpheus" in repo_id.lower():
        voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
        return {"voices": voices}
    else:
        # Use the voices loaded during server startup
        voices = available_voices

        # Filter voices by language code if provided
        if language:
            voices = [voice for voice in voices if voice.startswith(language)]

        return {"voices": voices}


@app.get("/models")
def get_models():
    """Return a list of available models."""
    return {"models": available_models}


def setup_server():
    """Setup the server by loading the model and creating the output directory."""
    global tts_model, audio_player, OUTPUT_FOLDER, available_voices

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

    # Load available voices
    try:
        default_repo = "hexgrad/Kokoro-82M"
        logger.debug(f"Loading voices from {default_repo}")
        available_voices = get_voice_names(default_repo)
        logger.debug(f"Successfully loaded {len(available_voices)} voices")
    except Exception as e:
        logger.error(f"Error loading voices: {str(e)}")
        logger.info("No voices loaded during startup")
        # We'll leave available_voices as an empty list

    # Load the model if not already loaded
    if tts_model is None:
        try:
            default_model = (
                "mlx-community/Kokoro-82M-4bit"  # Same default as in tts_endpoint
            )
            logger.debug(f"Loading TTS model from {default_model}")
            tts_model = load_model(default_model)
            logger.debug("TTS model loaded successfully")
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
