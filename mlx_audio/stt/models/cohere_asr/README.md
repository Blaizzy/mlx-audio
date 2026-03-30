# Cohere Transcribe

Cohere Transcribe is an open source release of a 2B parameter dedicated audio-in, text-out, automatic speech recognition (ASR) model. The model supports 14 languages.

Developed by: [Cohere](https://cohere.com) and [Cohere Labs](https://cohere.com/research).

## Available Model

| Model | Parameters | Description |
|-------|------------|-------------|
| [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | 2B | Multilingual offline ASR with prompt-controlled punctuation and long-form chunking |

**Supported Languages:** Arabic, German, Greek, English, Spanish, French, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Vietnamese, Chinese.

## CLI Usage

```bash
# Basic transcription
python -m mlx_audio.stt.generate \
  --model CohereLabs/cohere-transcribe-03-2026 \
  --audio audio.wav \
  --output-path output \
  --language en

# Save JSON output
python -m mlx_audio.stt.generate \
  --model CohereLabs/cohere-transcribe-03-2026 \
  --audio audio.wav \
  --output-path output \
  --format json \
  --language en

# Load from a local checkpoint directory
python -m mlx_audio.stt.generate \
  --model /path/to/cohere-transcribe-03-2026 \
  --audio audio.wav \
  --output-path output \
  --language fr
```

## Python Usage

### Single Audio Transcription

```python
from mlx_audio.stt import load

model = load("CohereLabs/cohere-transcribe-03-2026")

result = model.generate("audio.wav", language="en")
print(result.text)

for segment in result.segments:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
```

### Batched Offline Transcription

```python
from mlx_audio.stt import load

model = load("CohereLabs/cohere-transcribe-03-2026")

texts = model.transcribe(
    language="en",
    audio_files=["audio1.wav", "audio2.wav"],
    punctuation=True,
)

for text in texts:
    print(text)
```

### In-Memory Audio Arrays

```python
import numpy as np
from mlx_audio.stt import load

model = load("CohereLabs/cohere-transcribe-03-2026")

waveform = np.load("audio.npy").astype(np.float32)

texts = model.transcribe(
    language="en",
    audio_arrays=[waveform],
    sample_rates=[16000],
)

print(texts[0])
```

## Output Format

```python
STTOutput(
    text="Full transcription text",
    segments=[
        {"text": "segment text", "start": 0.0, "end": 12.3},
        ...
    ],
    language="en",
    prompt_tokens=9,
    generation_tokens=42,
    total_tokens=51,
    total_time=1.8,
    prompt_tps=5.0,
    generation_tps=23.3,
)
```

## Architecture

- FastConformer encoder with depthwise striding subsampling
- Transformer decoder with cross-attention
- SentencePiece tokenizer with prompt-controlled language and punctuation tokens
- 16kHz audio input with 128-bin log-mel frontend
- Energy-based long-form chunking with chunk-level segments in `STTOutput`
