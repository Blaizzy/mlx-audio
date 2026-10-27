# Ming-3.0 ASR Tiny

Native MLX transcription for [mlx-community/Ming-3.0-asr-tiny-bf16](https://huggingface.co/mlx-community/Ming-3.0-asr-tiny-bf16). The model supports English and Chinese, automatic language identification, language hints and hotwords. It uses a Whisper encoder and a hybrid Kimi Delta Attention / MLA MoE language backbone.

| Precision | Repository |
|-----------|------------|
| BF16 | [mlx-community/Ming-3.0-asr-tiny-bf16](https://huggingface.co/mlx-community/Ming-3.0-asr-tiny-bf16) |
| 8-bit | [mlx-community/Ming-3.0-asr-tiny-8bit](https://huggingface.co/mlx-community/Ming-3.0-asr-tiny-8bit) |
| 4-bit | [mlx-community/Ming-3.0-asr-tiny-4bit](https://huggingface.co/mlx-community/Ming-3.0-asr-tiny-4bit) |

Use any of these repositories with the Python and CLI examples below.

```python
from mlx_audio.stt import load

model = load("mlx-community/Ming-3.0-asr-tiny-bf16", strict=True)
result = model.generate("speech.wav", language="en", hotwords=["MLX", "Ming"])
print(result.language, result.text)
```

```bash
python -m mlx_audio.stt.generate \
  --model mlx-community/Ming-3.0-asr-tiny-bf16 \
  --audio speech.wav --output-path transcript
```

Paths and mono or time-major stereo arrays are accepted. Array inputs default to 16 kHz; pass `sample_rate=` for another rate. Files are resampled automatically. Long recordings are encoded in independent chunks of up to 30 seconds; the assembled prompt and generated text must fit the language model's context budget. `max_tokens` defaults to 256 and can be increased for longer transcripts.

BF16, 8-bit and 4-bit checkpoints load through the same API. Quantization preserves the audio encoder and MoE router precision. Runtime inference does not require PyTorch, FLA or remote model code. Streaming, timestamps and speculative MTP decoding are not implemented.
