# Granite Speech

MLX implementation of IBM's Granite Speech, a speech-to-text model that combines a CTC Conformer encoder with a Granite LLM decoder via a BLIP-2 QFormer projector. Beyond transcription, the model supports audio understanding tasks via custom prompts.

## Available Models

| Model | Parameters | Description |
|-------|------------|-------------|
| [ibm-granite/granite-4.0-1b-speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | ~1B | Speech recognition and understanding |

## CLI Usage

```bash
# Basic transcription
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output

# Verbose output with timing info
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --verbose

# Streaming output
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --stream

# Custom prompt via gen-kwargs
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output \
    --gen-kwargs '{"prompt": "Summarize the following audio."}'

# Output formats: txt, srt, vtt, json
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --format json
```

## Python Usage

### Transcription

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-4.0-1b-speech")

# Basic transcription
result = model.generate("audio.wav")
print(result.text)

# With custom prompt
result = model.generate("audio.wav", prompt="Transcribe the following audio.")
print(result.text)
```

### Audio Understanding (AST)

The model supports audio understanding tasks through custom prompts:

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-4.0-1b-speech")

# Summarization
result = model.generate("audio.wav", prompt="Summarize the following audio.")
print(result.text)

# Question answering
result = model.generate("audio.wav", prompt="What is being discussed in the audio?")
print(result.text)

# Topic extraction
result = model.generate("audio.wav", prompt="What are the main topics covered in this audio?")
print(result.text)
```

### Streaming

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-4.0-1b-speech")

for text in model.generate("audio.wav", stream=True):
    print(text, end="", flush=True)
```

### Generation Parameters

```python
result = model.generate(
    "audio.wav",
    max_tokens=4096,
    temperature=0.0,       # 0 = greedy decoding
    top_p=1.0,
    top_k=0,
    repetition_penalty=None,
    prompt="Transcribe the following audio.",
    prefill_step_size=2048,
    verbose=True,          # print timing info
)
```

## Architecture

- **Encoder**: CTC Conformer (16 layers, 1024 hidden dim, Shaw's relative positional embeddings, block-wise attention with context_size=200)
- **Projector**: BLIP-2 QFormer (2 layers, windowed cross-attention with window_size=15, downsample_rate=5)
- **Decoder**: Granite LLM (40 layers, 2048 hidden dim, GQA with 16/4 heads, RoPE, SwiGLU MLP)
- Audio input: any sample rate, 80-bin mel spectrogram with pair stacking (160-dim input)

## Audio Input

Granite Speech processes audio at its **original sample rate** without resampling. Supported input types:

- File path (WAV, FLAC, MP3, etc.)
- NumPy array (raw waveform)
- MLX array (raw waveform)

## Output Format

```python
STTOutput(
    text="Full transcription text",
    segments=[],
    prompt_tokens=207,
    generation_tokens=43,
    total_tokens=250,
    total_time=1.04,
    prompt_tps=199.0,
    generation_tps=41.2,
)
```
