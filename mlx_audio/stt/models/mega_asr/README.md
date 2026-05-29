# Mega-ASR

Mega-ASR is a robustness layer over Qwen3-ASR-1.7B: a tiny audio-quality **router** classifies each utterance as clean or degraded and switches a dense **LoRA adapter** in/out of the base weights — clean audio runs the unmodified Qwen3-ASR base path, degraded audio runs the LoRA (robust) path. This recovers large WER gains on noisy/far-field speech while leaving clean-speech accuracy unchanged.

## Available Models

| Model | Parameters | Description |
|-------|------------|-------------|
| [mlx-community/Mega-ASR-bf16](https://huggingface.co/mlx-community/Mega-ASR-bf16) | 1.7B | Routed Qwen3-ASR with audio-quality router + LoRA (dynamic clean/degraded switching) |

**Supported Languages:** inherits the Qwen3-ASR-1.7B backbone — Chinese, Cantonese, English, German, Spanish, French, Italian, Portuguese, Russian, Korean, Japanese.

## CLI Usage

```bash
# Basic transcription
uv run mlx_audio.stt.generate --model mlx-community/Mega-ASR-bf16 --audio audio.wav --output-path output

# With language specification
uv run mlx_audio.stt.generate --model mlx-community/Mega-ASR-bf16 --audio audio.wav --output-path output --language English

# Streaming output
uv run mlx_audio.stt.generate --model mlx-community/Mega-ASR-bf16 --audio audio.wav --output-path output --stream
```

## Python Usage

```python
from mlx_audio.stt import load

# Load model
model = load("mlx-community/Mega-ASR-bf16")

# Transcribe audio
result = model.generate("audio.wav", language="English")
print(result.text)
```

Routing is automatic:

- clean audio → base Qwen3-ASR path
- degraded audio → LoRA-enhanced (robust) path

You do not need to toggle adapters manually.

### Streaming Transcription

```python
from mlx_audio.stt import load

model = load("mlx-community/Mega-ASR-bf16")

for text in model.stream_transcribe("audio.wav", language="English"):
    print(text, end="", flush=True)
```

## Convert from Hugging Face

To build the MLX model yourself instead of using the prebuilt repo above, convert the upstream `zhifeixie/Mega-ASR`:

```bash
uv run python -m mlx_audio.stt.models.mega_asr.convert \
  --hf-path zhifeixie/Mega-ASR \
  --mlx-path /path/to/mega-asr-mlx \
  --dtype bfloat16
```

The base Qwen3-ASR weights are converted as **dense bf16** shards on purpose. Mega-ASR adds LoRA deltas at inference time, so the base must stay dense rather than quantized. The converted directory can then be loaded directly: `load("/path/to/mega-asr-mlx")`.
