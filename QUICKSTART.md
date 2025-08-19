# Quick Start Guide

Get mlx-audio up and running in minutes with this quick start guide.

## Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- Git installed
- Basic command line knowledge

## Quick Installation

```bash
# 1. Clone the repository
git clone https://github.com/dax8it/mlx-audio.git
cd mlx-audio

# 2. Create and activate conda environment
conda create -n mlx-audio python=3.11 -y
conda activate mlx-audio

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test TTS generation
python -m mlx_audio.tts.generate --text "Hello, welcome to MLX Audio!" --voice af_heart
```

This will:
1. Download the default Kokoro TTS model
2. Generate audio with the default voice
3. Save the output as a WAV file

## Quick Web Interface Setup

```bash
# Start the web server
python -m mlx_audio.server

# Open your browser to:
# http://127.0.0.1:8000
```

The web interface provides:
- Real-time TTS generation
- 3D audio visualization
- Voice selection
- Speed control

## Quick LLM Integration

To use the full STT → LLM → TTS pipeline:

1. Set up environment variables in `.env`:
   ```bash
   LLM_PROVIDER=MLX
   MLX_LLM_MODEL=mlx-community/Mistral-7B-v0.1-4bit
   ```

2. Run the server:
   ```bash
   python -m mlx_audio.server
   ```

3. Use the web interface for voice conversations

## First-Time User Tips

1. **First run**: The first execution will download models (1-5GB depending on model)
2. **Voice selection**: Try different voices (`af_heart`, `af_nova`, `bf_emma`)
3. **Speed control**: Adjust between 0.5x (slow) to 2.0x (fast)
4. **Model caching**: Models are cached locally for faster subsequent runs

## Common First Commands

```bash
# Generate TTS with specific voice and speed
python -m mlx_audio.tts.generate --text "Hello world" --voice af_nova --speed 1.3

# Generate TTS and play immediately
python -m mlx_audio.tts.generate --text "Hello world" --play

# Generate TTS with output file prefix
python -m mlx_audio.tts.generate --text "Hello world" --file_prefix welcome_message

# Use different model
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-8bit --text "Hello"
```

## Web Interface Quick Start

1. Run `python -m mlx_audio.server`
2. Navigate to `http://127.0.0.1:8000`
3. Enter text in the input box
4. Select voice and adjust speed
5. Click "Generate" to create audio
6. The audio will play automatically and be saved to the output folder

## Troubleshooting Quick Fixes

If you encounter issues:

```bash
# Clear Python cache
find . -type d -name __pycache__ -delete

# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Clear model cache
rm -rf ~/.cache/huggingface/
```

## Next Steps

After successful installation:
1. Explore the [full documentation](DOCUMENTATION.md)
2. Review the [setup guide](SETUP.md) for advanced configuration
3. Try the Bible audiobook example in the [examples](examples/) directory:
   ```bash
   cd examples/bible-audiobook
   bun install
   bun run src/index.ts
   ```
4. Join our community for support and updates