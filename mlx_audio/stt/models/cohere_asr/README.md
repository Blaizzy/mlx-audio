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

## Optional VAD Pre-processing

Cohere's encoder has a positional-encoding limit of ≈ 6.7 minutes. Long-form audio is therefore split into chunks before transcription. Two strategies are available:

| Strategy | When | Notes |
|---|---|---|
| Energy-based fixed-duration chunking (default, `vad=False`) | Clean dense speech (audiobooks, narration) | Splits at low-energy points within each `chunk_duration` window |
| Silero VAD (opt-in, `vad=True`) | Long-form audio with silences / non-speech (meetings, podcasts, interviews) | Trims silence, aligns chunks to natural pauses |

```python
from mlx_audio.stt import load
from mlx_audio.vad import load as load_vad

model = load("CohereLabs/cohere-transcribe-03-2026")
result = model.generate("meeting.wav", language="en", vad=True)
print(result.text)
```

`vad=True` uses the in-tree `mlx_audio.vad.silero_vad` module (added in #701) with 256 ms noisy-OR aggregation over 8 × 32 ms chunks. Tunables are exposed as keyword arguments: `vad_merge_gap_s`, `vad_max_chunk_s`.

### Measured trade-offs

10-min English meeting recording (silence + speech, M1 Max):

| | Wall | Notable |
|---|---|---|
| `vad=False` | 32 s | Hallucinations on silent leading audio (e.g. `"a very strong sense of humor" × 3`), mid-sentence cuts at 30 s boundaries |
| `vad=True` | 30 s (-7 %) | Hallucinations gone, natural sentence boundaries, fewer downstream chunks |

30-min concatenated LibriSpeech reads (clean audiobook narration, ground-truth WER):

| | Wall | WER | Insertions |
|---|---|---|---|
| `vad=False` | 55 s | **1.58 %** | 8 |
| `vad=True` | 100 s (+82 %) | 2.29 % | 25 |

Take-away: VAD pre-processing **is opt-in by design**. It improves long-form ASR on audio with silences or non-speech sections (meetings, podcasts), at a small wall-clock cost on those workloads. On clean dense narration it produces no quality benefit and can add insertions at chunk boundaries — keep `vad=False` for that case.

The Swift sibling (`mlx-audio-swift` PR #177) reproduces the same shape (Δ WER ≈ +0.7 pp on dense LibriSpeech reads, +20 insertions), confirming the trade-off is architectural and not implementation-specific.

## Architecture

- FastConformer encoder with depthwise striding subsampling
- Transformer decoder with cross-attention
- SentencePiece tokenizer with prompt-controlled language and punctuation tokens
- 16kHz audio input with 128-bin log-mel frontend
- Energy-based long-form chunking with chunk-level segments in `STTOutput`
