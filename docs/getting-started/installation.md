# Installation

## Quick Install

=== "pip"

    ```bash
    pip install mlx-audio
    ```

=== "uv (tool)"

    Install the CLI tools globally with uv:

    ```bash
    # Latest release from PyPI
    uv tool install --force mlx-audio --prerelease=allow
    ```

    Or install from the latest source:

    ```bash
    # Latest code from GitHub
    uv tool install --force git+https://github.com/Blaizzy/mlx-audio.git --prerelease=allow
    ```

## Optional Extras

mlx-audio ships optional dependency groups so you only install what you need:

| Extra | What it includes |
|-------|-----------------|
| `tts` | Text-to-speech dependencies |
| `stt` | Speech-to-text dependencies |
| `sts` | Speech-to-speech dependencies |
| `all` | Everything (TTS + STT + STS) |
| `dev` | Development and testing tools |

Install extras with bracket syntax:

```bash
# TTS only
pip install "mlx-audio[tts]"

# STT only
pip install "mlx-audio[stt]"

# Everything
pip install "mlx-audio[all]"
```

## Development Setup

Clone the repository and install in editable mode with dev dependencies:

```bash
git clone https://github.com/Blaizzy/mlx-audio.git
cd mlx-audio
pip install -e ".[dev]"
```

## ffmpeg

ffmpeg is required for saving audio in **MP3**, **FLAC**, **OGG**, **Opus**, or **Vorbis** format. WAV works without it.

=== "macOS (Homebrew)"

    ```bash
    brew install ffmpeg
    ```

=== "Ubuntu / Debian"

    ```bash
    sudo apt install ffmpeg
    ```

!!! tip
    If you only need WAV output, you can skip installing ffmpeg entirely.
