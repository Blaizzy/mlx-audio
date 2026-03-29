# Dev Setup

This page walks you through setting up a local development environment for MLX Audio.

## Prerequisites

- **Python 3.10+**
- **Apple Silicon Mac** (M1 / M2 / M3 / M4)
- **Git**
- **ffmpeg** (optional; required for non-WAV audio formats)

## Clone the Repository

```bash
git clone https://github.com/Blaizzy/mlx-audio.git
cd mlx-audio
```

## Install Dependencies

### Using pip

Install in editable mode with the development extras:

```bash
pip install -e ".[dev]"
```

### Using uv

If you use [uv](https://github.com/astral-sh/uv) for project management:

```bash
uv sync
```

This installs all dependencies (core, dev, and optional groups) from `pyproject.toml`.

### Optional Extras

Install only what you need:

```bash
# TTS dependencies only
pip install -e ".[tts]"

# STT dependencies only
pip install -e ".[stt]"

# API server dependencies
pip install -e ".[server]"

# Everything
pip install -e ".[dev,tts,stt,server]"
```

## Running Tests

```bash
# Run the full test suite
pytest

# Run tests for a specific module
pytest mlx_audio/tts/tests/
pytest mlx_audio/stt/tests/
pytest mlx_audio/sts/tests/
```

## Code Formatting

The project uses [Black](https://github.com/psf/black) for code formatting:

```bash
# Check formatting
black --check .

# Auto-format
black .
```

## Linting

```bash
# If ruff is configured
ruff check .
```

## Project Structure

```
mlx-audio/
├── mlx_audio/
│   ├── tts/           # Text-to-Speech models and utilities
│   ├── stt/           # Speech-to-Text models and utilities
│   ├── sts/           # Speech-to-Speech models
│   ├── vad/           # Voice Activity Detection
│   ├── audio_io.py    # Audio reading/writing
│   ├── server.py      # FastAPI server
│   ├── convert.py     # Model conversion and quantization
│   └── ui/            # Next.js web interface
├── docs/              # MkDocs documentation
├── examples/          # Example scripts
├── pyproject.toml     # Project configuration
└── mkdocs.yml         # Documentation configuration
```

## Running the Server Locally

```bash
# API server
mlx_audio.server --host localhost --port 8000

# Web UI (separate terminal)
cd mlx_audio/ui
npm install
npm run dev
```

## Common Development Tasks

### Adding a New Dependency

Add the dependency to the appropriate group in `pyproject.toml`, then reinstall:

```bash
pip install -e ".[dev]"
```

### Testing a Specific Model

```bash
# Generate speech with a local or HuggingFace model
mlx_audio.tts.generate \
    --model mlx-community/Kokoro-82M-bf16 \
    --text "Testing my changes." \
    --lang_code a \
    --play
```

### Building Documentation

```bash
# Install docs dependencies
pip install mkdocs-material mkdocstrings[python]

# Serve docs locally
mkdocs serve

# Build static site
mkdocs build
```

The docs site will be available at `http://localhost:8000`.
