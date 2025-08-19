# Setup Guide

Detailed installation and setup instructions for mlx-audio.

## System Requirements

- Apple Silicon Mac (M1, M2, M3, or M4 chip)
- macOS 13.0 (Ventura) or later recommended
- At least 16GB RAM (32GB recommended)
- 20GB+ free disk space for models

## Installation Steps

### 1. Install Conda

If you don't have Conda installed:

```bash
# Download Miniconda for macOS (Apple Silicon)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install Miniconda
bash Miniconda3-latest-MacOSX-arm64.sh

# Restart terminal or run:
source ~/.zshrc
```

### 2. Clone the Repository

```bash
git clone https://github.com/dax8it/mlx-audio.git
cd mlx-audio
```

### 3. Create Conda Environment

```bash
# Create environment with Python 3.11
conda create -n mlx-audio python=3.11

# Activate environment
conda activate mlx-audio
```

### 4. Install Dependencies

```bash
# Install core requirements
pip install -r requirements.txt
```

If you encounter any issues, try installing dependencies individually:

```bash
# Core MLX dependencies
pip install mlx mlx-lm

# Audio processing
pip install soundfile sounddevice

# Web framework
pip install fastapi uvicorn

# Machine learning
pip install transformers torch

# Speech processing
pip install fastrtc webrtcvad

# Other utilities
pip install numpy scipy einops
```

### 5. Install Ollama (Optional)

To use Ollama as the LLM backend:

1. Download Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Install the downloaded package
3. Pull a model:
   ```bash
   ollama pull phi4:latest
   ```
4. Start the Ollama service:
   ```bash
   ollama serve
   ```

### 6. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file to configure your setup:

```bash
# --- TTS Settings ---
TTS_DEFAULT_MODEL=mlx-community/Kokoro-82M-4bit

# --- LLM Settings ---
# Options: OLLAMA, MLX
LLM_PROVIDER=MLX

# --- MLX Settings (used if LLM_PROVIDER=MLX)
MLX_LLM_MODEL=/path/to/your/mlx/model
MLX_SYSTEM_PROMPT=You are a helpful voice assistant.

# --- Ollama Settings (used if LLM_PROVIDER=OLLAMA)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=phi4:latest
OLLAMA_SYSTEM_PROMPT=You are a helpful voice assistant.
```

### 7. Download Models

Models are automatically downloaded on first use, but you can manually download them:

#### TTS Models

```bash
# Kokoro models (automatically downloaded)
# Available variants:
# - mlx-community/Kokoro-82M-4bit (smallest, fastest)
# - mlx-community/Kokoro-82M-8bit (balanced)
# - prince-canuma/Kokoro-82M (full precision)
```

#### LLM Models

For MLX backend, you can use Hugging Face models converted to MLX format:

```bash
# Example: Download and convert a model
python -m mlx_lm.convert --hf-path mistralai/Mistral-7B-v0.1 --mlx-path ./models/mistral-7b-mlx
```

For Ollama backend:

```bash
# Pull commonly used models
ollama pull phi4:latest
ollama pull mistral:latest
ollama pull llama3:latest
```

## Testing the Installation

### 1. Test TTS Generation

```bash
python -m mlx_audio.tts.generate --text "Hello, this is a test of the TTS system." --voice af_heart --speed 1.2
```

### 2. Test STT Transcription

```bash
# If you have an audio file to test with
python -m mlx_audio.stt.generate --model mlx-community/whisper-large-v3-turbo --audio ./test.wav --output ./transcript --format txt
```

### 3. Test Web Interface

```bash
python -m mlx_audio.server --verbose
```

Open `http://127.0.0.1:8000` in your browser.

## Troubleshooting

### Common Issues and Solutions

#### 1. "No module named ..." errors

```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

#### 2. Audio device errors

```bash
# Install system audio dependencies
brew install portaudio
pip install sounddevice --force-reinstall
```

#### 3. Model download issues

```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Or set cache directory explicitly
export HF_HOME=/path/to/your/hf/cache
```

#### 4. MLX compatibility issues

```bash
# Update MLX to latest version
pip install --upgrade mlx mlx-lm
```

#### 5. Ollama connection errors

```bash
# Check if Ollama service is running
ps aux | grep ollama

# Restart Ollama service
brew services restart ollama
```

### Performance Tuning

#### 1. Memory Management

For systems with limited RAM:

```bash
# Set environment variables to limit memory usage
export MLX_MEMORY_LIMIT=8G
```

#### 2. Model Quantization

Use quantized models for better performance:

```bash
# Use 4-bit models for lower memory usage
TTS_DEFAULT_MODEL=mlx-community/Kokoro-82M-4bit
```

#### 3. CPU/GPU Configuration

MLX automatically uses the best available hardware, but you can control this:

```bash
# Force CPU-only execution (not recommended for performance)
export MLX_DISABLE_GPU=1
```

## Updating the Project

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Update MLX
pip install --upgrade mlx mlx-lm
```

## Environment Verification

Run this script to verify your setup:

```bash
#!/bin/bash
echo "=== MLX-Audio Environment Verification ==="

echo "1. Python version:"
python --version

echo "2. Conda environment:"
conda info --envs | grep mlx-audio

echo "3. MLX installation:"
python -c "import mlx; print('MLX version:', mlx.__version__)"

echo "4. Required packages:"
pip list | grep -E "mlx|fastapi|transformers|soundfile"

echo "5. Audio devices:"
python -c "import sounddevice as sd; print(sd.query_devices())"

echo "6. Environment variables:"
echo "LLM_PROVIDER: ${LLM_PROVIDER:-Not set}"
echo "TTS_DEFAULT_MODEL: ${TTS_DEFAULT_MODEL:-Not set}"

echo "=== Verification Complete ==="
```

Save this as `verify_setup.sh` and run:
```bash
chmod +x verify_setup.sh
./verify_setup.sh
```