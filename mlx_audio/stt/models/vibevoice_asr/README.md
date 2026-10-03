# VibeVoice ASR

MLX support for Microsoft's VibeVoice ASR checkpoints:

| Checkpoint | Output | Streaming protocol |
|---|---|---|
| [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR) | Speaker, timestamps, content | No |
| [`microsoft/VibeVoice-ASR-Streaming-1.5B`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B) | Speaker, content | Yes |
| [`microsoft/VibeVoice-ASR-Streaming-7B`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B) | Speaker, content | Yes |

The streaming checkpoints use 22 speech-token frames (2.933 s) of new audio
and 4 lookahead frames (0.533 s) per step.

## File transcription

```python
from mlx_audio.stt import load

model = load("microsoft/VibeVoice-ASR-Streaming-1.5B")

result = model.generate("meeting.wav", hotwords=["VibeVoice", "MLX"])
print(result.text)
```

The upstream BF16 repositories are about 5.6 GB for the 1.5B checkpoint and
17 GB for the 7B checkpoint. Both load directly. A model converted with the
shared `mlx_audio.convert` command keeps the tokenizer and processor sidecars
needed by streaming inference.

```bash
python -m mlx_audio.convert \
  --hf-path microsoft/VibeVoice-ASR-Streaming-1.5B \
  --mlx-path VibeVoice-ASR-Streaming-1.5B-mlx \
  --model-domain stt \
  --dtype bfloat16
```

## Incremental output

```python
for chunk in model.stream_transcribe(
    "meeting.wav",
    hotwords=["VibeVoice", "MLX"],
    max_tokens_per_chunk=256,
):
    print(chunk, end="", flush=True)
```

Each yielded string is the text for one trained audio chunk. The language-model
KV cache persists across chunks, which lets the model maintain speaker and text
context throughout the stream.

## Live audio

For a microphone or WebSocket integration, buffer
`model.streaming_window_samples` samples, advance the buffer by
`model.streaming_chunk_samples`, and retain the lookahead for the next window:

```python
import mlx.core as mx

state = model.init_streaming_state(context_info="VibeVoice, MLX")

# `window` is mono float32 at model.sample_rate and has
# model.streaming_window_samples samples.
features = model.encode_speech(mx.array(window)[None, :])
text, state = model.streaming_generate_step(features, state)
print(text, end="", flush=True)
```

Zero-pad the final partial window before encoding it.

## CLI

```bash
python -m mlx_audio.stt.generate \
  --model microsoft/VibeVoice-ASR-Streaming-1.5B \
  --audio meeting.wav \
  --output-path transcript \
  --context "VibeVoice, MLX"
```

Supported languages for the streaming checkpoints are Chinese, English,
French, German, Italian, Japanese, Korean, Portuguese, Russian, and Spanish.
