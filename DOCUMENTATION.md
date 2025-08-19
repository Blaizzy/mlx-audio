# MLX-Audio Documentation

A comprehensive text-to-speech (TTS), speech-to-text (STT), and large language model (LLM) pipeline built on Apple's MLX framework, providing efficient voice interaction entirely on Apple Silicon devices with no cloud dependencies.

## Features

- **Local Voice Pipeline**: Complete STT → LLM → TTS processing without internet
- **Apple Silicon Optimized**: Uses MLX framework for maximum performance on M-series chips
- **Flexible LLM Backend**: Switch between local MLX models and Ollama
- **Multiple Language Support**: Supports various languages with voice customization
- **Real-time Processing**: WebRTC integration for conversational interactions
- **Web Interface**: Interactive UI with 3D audio visualization
- **API Access**: RESTful endpoints for integration
- **Voice Cloning**: Custom voice generation using reference audio

## Architecture

The mlx-audio pipeline consists of three integrated components:

1. **Speech-to-Text (STT)**: Real-time speech recognition using Whisper models
2. **Large Language Model (LLM)**: Local inference using either MLX or Ollama backends
3. **Text-to-Speech (TTS)**: Voice synthesis using Kokoro and CSM models

## Installation

### Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4 chip)
- Python 3.8 or higher
- Conda (recommended for environment management)

### Conda Environment Setup

Create a dedicated conda environment for mlx-audio:

```bash
# Clone the repository
git clone https://github.com/dax8it/mlx-audio.git
cd mlx-audio

# Create conda environment
conda create -n mlx-audio python=3.11

# Activate environment
conda activate mlx-audio

# Install core dependencies
pip install -r requirements.txt
```

### Framework Installation

#### MLX Framework

MLX is automatically installed with the requirements, but you can also install it directly:

```bash
pip install mlx
pip install mlx-lm
```

#### Ollama (Optional)

To use Ollama as the LLM backend:

1. Download and install Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Pull a compatible model:
   ```bash
   ollama pull phi4:latest
   ```
3. Start the Ollama service:
   ```bash
   ollama serve
   ```

### Model Requirements

#### TTS Models

The project uses Kokoro models by default. Models are automatically downloaded from Hugging Face on first use, or you can specify local paths.

#### LLM Models

For MLX backend, you can use any compatible model. For Ollama backend, compatible models include:
- phi4:latest
- mistral:latest
- llama3:latest
- gemma2:latest

## Configuration

### Environment Variables

Create a `.env` file in the project root to configure the system:

```bash
# --- TTS Settings ---
TTS_DEFAULT_MODEL=mlx-community/Kokoro-82M-4bit

# --- LLM Settings ---
# Set the LLM provider: OLLAMA or MLX
LLM_PROVIDER=MLX  # Options: OLLAMA, MLX

# --- MLX Settings (used if LLM_PROVIDER=MLX)
MLX_LLM_MODEL=/path/to/your/mlx/model
MLX_SYSTEM_PROMPT=You are a helpful voice assistant.

# --- Ollama Settings (used if LLM_PROVIDER=OLLAMA)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=phi4:latest
OLLAMA_SYSTEM_PROMPT=You are a helpful voice assistant.
```

### LLM Provider Switching

Switch between LLM backends by changing the `LLM_PROVIDER` environment variable:

- `LLM_PROVIDER=MLX`: Uses local MLX models (no internet required)
- `LLM_PROVIDER=OLLAMA`: Uses local Ollama models (requires Ollama service)

## Usage

### Command Line Interface

#### TTS Generation

```bash
# Basic usage
python -m mlx_audio.tts.generate --text "Hello, world"

# Specify voice and speed
python -m mlx_audio.tts.generate --text "Hello, world" --voice af_heart --speed 1.4

# Voice cloning with reference audio
python -m mlx_audio.tts.generate --model mlx-community/csm-1b --text "Hello" --ref_audio ./reference.wav
```

#### STT Transcription

```bash
# Transcribe audio file
python -m mlx_audio.stt.generate --model mlx-community/whisper-large-v3-turbo --audio ./input.wav --output ./transcript --format txt
```

### Web Interface

Start the web server with API:

```bash
# Start the server
python -m mlx_audio.server

# Start with custom settings
python -m mlx_audio.server --host 0.0.0.0 --port 9000 --verbose
```

Then open your browser to `http://127.0.0.1:8000`

### API Endpoints

The server provides REST API endpoints:

- `POST /tts`: Generate TTS audio
  - Parameters: `text`, `voice`, `speed`
  - Returns: JSON with filename of generated audio

- `GET /audio/{filename}`: Retrieve generated audio file

- `POST /speech_to_speech_input`: Real-time speech-to-speech processing

## Implementation Details

### MLX Integration

The project leverages Apple's MLX framework for all compute-intensive operations:

1. **TTS Models**: Kokoro and CSM models optimized for MLX
2. **STT Models**: Whisper models using MLX acceleration
3. **LLM Models**: Transformer models using `mlx-lm` library
4. **Performance**: Native Apple Silicon acceleration with memory optimization

### Ollama Integration

When configured to use Ollama:

1. **API Communication**: Local HTTP calls to Ollama service
2. **Model Management**: Ollama handles model downloading and quantization
3. **Compatibility**: Supports all Ollama-compatible models
4. **Fallback**: Automatic fallback if MLX models aren't available

### Local Operation

All components operate locally:

- Models downloaded to Hugging Face cache or specified local paths
- No external API calls for core functionality
- Audio processing entirely on-device
- Network only used for Ollama when configured

## Models

### Kokoro TTS

Multilingual TTS model with voice styles:

- 🇺🇸 `'a'` - American English
- 🇬🇧 `'b'` - British English
- 🇯🇵 `'j'` - Japanese
- 🇨🇳 `'z'` - Mandarin Chinese

### CSM (Conversational Speech Model)

Voice cloning model from Sesame:

- Clone voices using reference audio samples
- High-quality conversational speech synthesis
- Supports various reference audio inputs

### Whisper STT

Speech recognition models:

- Automatic language detection
- Real-time transcription capabilities
- Multiple model sizes for performance/accuracy tradeoff

## Advanced Features

### Quantization

Models can be quantized for improved performance:

```python
from mlx_audio.tts.utils import quantize_model, load_model
import json
import mlx.core as mx

model = load_model(repo_id='prince-canuma/Kokoro-82M')
config = model.config

# Quantize to 8-bit
group_size = 64
bits = 8
weights, config = quantize_model(model, config, group_size, bits)

# Save quantized model
with open('./8bit/config.json', 'w') as f:
    json.dump(config, f)

mx.save_safetensors("./8bit/kokoro-v1_0.safetensors", weights, metadata={"format": "mlx"})
```

### Voice Customization

Customize voices with various parameters:

- Voice selection (af_heart, af_nova, af_bella, bf_emma)
- Speed control (0.5x to 2.0x)
- Pitch adjustment
- Reference audio for voice cloning

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure models are correctly downloaded
   - Check Hugging Face cache permissions
   - Verify model paths in configuration

2. **Audio Quality Issues**
   - Check reference audio quality for voice cloning
   - Adjust volume normalization settings
   - Verify sample rate compatibility

3. **LLM Response Problems**
   - Check Ollama service status when using Ollama backend
   - Verify MLX model paths when using MLX backend
   - Adjust system prompts for better responses

### Performance Optimization

1. **Use Quantized Models**: 4-bit or 8-bit models for better performance
2. **Limit Context Length**: Reduce conversation history for faster responses
3. **Optimize Audio Segments**: Use appropriate audio chunk sizes for real-time processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)

## Acknowledgements

- Apple MLX team for the efficient machine learning framework
- Kokoro model creators for text-to-speech synthesis
- Ollama project for local LLM deployment
- Three.js for the 3D visualization components