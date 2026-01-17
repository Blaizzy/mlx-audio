# MLX-Audio

The best audio processing library built on Apple's MLX framework, providing fast and efficient text-to-speech (TTS), speech-to-text (STT), and speech-to-speech (STS) on Apple Silicon.

## Features

- Fast inference optimized for Apple Silicon (M series chips)
- Multiple model architectures for TTS, STT, and STS
- Multilingual support across models
- Voice customization and cloning capabilities
- Adjustable speech speed control
- Interactive web interface with 3D audio visualization
- OpenAI-compatible REST API
- Quantization support (3-bit, 4-bit, 6-bit, 8-bit, and more) for optimized performance
- Swift package for iOS/macOS integration

## Installation

```bash
pip install mlx-audio
```

For development or web interface:

```bash
git clone https://github.com/Blaizzy/mlx-audio.git
cd mlx-audio
pip install -e ".[dev]"
```

## Quick Start

### Command Line

```bash
# Basic TTS generation
mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 --text "Hello, world!"

# With voice selection and speed adjustment
mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 --text "Hello!" --voice af_heart --speed 1.2

# Play audio immediately
mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 --text "Hello!" --play
```

### Python API

```python
from mlx_audio.tts.utils import load_model

# Load model
model = load_model("mlx-community/Kokoro-82M-bf16")

# Generate speech
for result in model.generate("Hello from MLX-Audio!", voice="af_heart"):
    print(f"Generated {result.audio.shape[0]} samples")
    # result.audio contains the waveform as mx.array
```

## Supported Models

### Text-to-Speech (TTS)

| Model | Description | Languages | Repo |
|-------|-------------|-----------|------|
| **Kokoro** | Fast, high-quality multilingual TTS | EN, JA, ZH, FR, ES, IT, PT, HI | [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) |
| **CSM** | Conversational Speech Model with voice cloning | EN | [mlx-community/csm-1b](https://huggingface.co/mlx-community/csm-1b) |
| **Dia** | Dialogue-focused TTS | EN | [mlx-community/Dia-1.6B-bf16](https://huggingface.co/mlx-community/Dia-1.6B-bf16) |
| **OuteTTS** | Efficient TTS model | EN | [mlx-community/OuteTTS-0.2-500M](https://huggingface.co/mlx-community/OuteTTS-0.2-500M) |
| **Spark** | SparkTTS model | EN, ZH | [mlx-community/SparkTTS-0.5B-bf16](https://huggingface.co/mlx-community/SparkTTS-0.5B-bf16) |
| **Chatterbox** | Expressive multilingual TTS | EN, ES, FR, DE, IT, PT, PL, TR, RU, NL, CS, AR, ZH, JA, HU, KO | [mlx-community/Chatterbox-bf16](https://huggingface.co/mlx-community/Chatterbox-bf16) |
| **Soprano** | High-quality TTS | EN | [mlx-community/Soprano-bf16](https://huggingface.co/mlx-community/Soprano-bf16) |

### Speech-to-Text (STT)

| Model | Description | Languages | Repo |
|-------|-------------|-----------|------|
| **Whisper** | OpenAI's robust STT model | 99+ languages | [mlx-community/whisper-large-v3-turbo-asr-fp16](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-fp16) |
| **Parakeet** | NVIDIA's accurate STT | EN | [mlx-community/parakeet-tdt-0.6b-v2](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2) |
| **Voxtral** | Mistral's speech model | Multiple | [mlx-community/Voxtral-Mini-3B-2507-bf16](https://huggingface.co/mlx-community/Voxtral-Mini-3B-2507-bf16) |

### Speech-to-Speech (STS)

| Model | Description | Use Case | Repo |
|-------|-------------|----------|------|
| **SAM-Audio** | Text-guided source separation | Extract specific sounds | [mlx-community/sam-audio-large](https://huggingface.co/mlx-community/sam-audio-large) |
| **MossFormer2 SE** | Speech enhancement | Noise removal | [starkdmi/MossFormer2_SE_48K_MLX](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX) |

## Model Examples

### Kokoro TTS

Kokoro is a fast, multilingual TTS model with 54 voice presets.

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Kokoro-82M-bf16")

# Generate with different voices
for result in model.generate(
    text="Welcome to MLX-Audio!",
    voice="af_heart",  # American female
    speed=1.0,
    lang_code="a"  # American English
):
    audio = result.audio
```

**Available Voices:**
- American English: `af_heart`, `af_bella`, `af_nova`, `af_sky`, `am_adam`, `am_echo`, etc.
- British English: `bf_alice`, `bf_emma`, `bm_daniel`, `bm_george`, etc.
- Japanese: `jf_alpha`, `jm_kumo`, etc.
- Chinese: `zf_xiaobei`, `zm_yunxi`, etc.

**Language Codes:**
| Code | Language | Note |
|------|----------|------|
| `a` | American English | Default |
| `b` | British English | |
| `j` | Japanese | Requires `pip install misaki[ja]` |
| `z` | Mandarin Chinese | Requires `pip install misaki[zh]` |
| `e` | Spanish | |
| `f` | French | |

### CSM (Voice Cloning)

Clone any voice using a reference audio sample:

```bash
mlx_audio.tts.generate \
    --model mlx-community/csm-1b \
    --text "Hello from Sesame." \
    --ref_audio ./reference_voice.wav \
    --play
```

### Whisper STT

```python
from mlx_audio.stt.utils import load_model, transcribe

model = load_model("mlx-community/whisper-large-v3-turbo-asr-fp16")
result = transcribe("audio.wav", model=model)
print(result["text"])
```

### SAM-Audio (Source Separation)

Separate specific sounds from audio using text prompts:

```python
from mlx_audio.sts import SAMAudio, SAMAudioProcessor, save_audio

model = SAMAudio.from_pretrained("mlx-community/sam-audio-large")
processor = SAMAudioProcessor.from_pretrained("mlx-community/sam-audio-large")

batch = processor(
    descriptions=["A person speaking"],
    audios=["mixed_audio.wav"],
)

result = model.separate_long(
    batch.audios,
    descriptions=batch.descriptions,
    anchors=batch.anchor_ids,
    chunk_seconds=10.0,
    overlap_seconds=3.0,
    ode_opt={"method": "midpoint", "step_size": 2/32},
)

save_audio(result.target[0], "voice.wav")
save_audio(result.residual[0], "background.wav")
```

### MossFormer2 (Speech Enhancement)

Remove noise from speech recordings:

```python
from mlx_audio.sts import MossFormer2SEModel, save_audio

model = MossFormer2SEModel.from_pretrained("starkdmi/MossFormer2_SE_48K_MLX")
enhanced = model.enhance("noisy_speech.wav")
save_audio(enhanced, "clean.wav", 48000)
```

## Web Interface & API Server

MLX-Audio includes a modern web interface and OpenAI-compatible API.

### Starting the Server

```bash
# Start API server
mlx_audio.server --host 0.0.0.0 --port 8000

# Start web UI (in another terminal)
cd mlx_audio/ui
npm install && npm run dev
```

### API Endpoints

**Text-to-Speech** (OpenAI-compatible):
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Kokoro-82M-bf16", "input": "Hello!", "voice": "af_heart"}' \
  --output speech.wav
```

**Speech-to-Text**:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo-asr-fp16"
```

## Quantization

Reduce model size and improve performance with quantization using the convert script:

```bash
# Convert and quantize to 4-bit
python -m mlx_audio.convert \
    --hf-path prince-canuma/Kokoro-82M \
    --mlx-path ./Kokoro-82M-4bit \
    --quantize \
    --q-bits 4 \
    --upload-repo username/Kokoro-82M-4bit (optional: if you want to upload the model to Hugging Face)

# Convert with specific dtype (bfloat16)
python -m mlx_audio.convert \
    --hf-path prince-canuma/Kokoro-82M \
    --mlx-path ./Kokoro-82M-bf16 \
    --dtype bfloat16 \
    --upload-repo username/Kokoro-82M-bf16 (optional: if you want to upload the model to Hugging Face)
```

**Options:**
| Flag | Description |
|------|-------------|
| `--hf-path` | Source Hugging Face model or local path |
| `--mlx-path` | Output directory for converted model |
| `-q, --quantize` | Enable quantization |
| `--q-bits` | Bits per weight (4, 6, or 8) |
| `--q-group-size` | Group size for quantization (default: 64) |
| `--dtype` | Weight dtype: `float16`, `bfloat16`, `float32` |
| `--upload-repo` | Upload converted model to HF Hub |

**Pre-quantized models available:**
- [mlx-community/Kokoro-82M-4bit](https://huggingface.co/mlx-community/Kokoro-82M-4bit)
- [mlx-community/Kokoro-82M-6bit](https://huggingface.co/mlx-community/Kokoro-82M-6bit)
- [mlx-community/Kokoro-82M-8bit](https://huggingface.co/mlx-community/Kokoro-82M-8bit)
- [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16)

## Swift Integration

Native Swift package for iOS and macOS apps.

### Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/Blaizzy/mlx-audio.git", from: "0.2.5")
]
```

### Usage

```swift
import MLXAudio

// Create session (auto-downloads model)
let session = try await MarvisSession(voice: .conversationalA)

// Generate speech (auto-plays)
let result = try await session.generate(for: "Hello from Swift!")

// Streaming generation
for try await chunk in session.stream(text: "Streaming audio...") {
    print("Chunk: \(chunk.sampleCount) samples")
}
```

**Supported Platforms:** macOS 14.0+, iOS 16.0+

## Requirements

- Python 3.9+
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX framework
- **ffmpeg** (required for MP3/FLAC audio encoding)

### Installing ffmpeg

ffmpeg is required for saving audio in MP3 or FLAC format. Install it using:

```bash
# macOS (using Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

WAV format works without ffmpeg.

## License

[MIT License](LICENSE)

## Citation

```bibtex
@misc{mlx-audio,
  author = {Canuma, Prince},
  title = {MLX Audio},
  year = {2025},
  howpublished = {\url{https://github.com/Blaizzy/mlx-audio}},
  note = {Audio processing library for Apple Silicon with TTS, STT, and STS capabilities.}
}
```

## Acknowledgements

- [Apple MLX Team](https://github.com/ml-explore/mlx) for the MLX framework
