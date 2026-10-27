# Ming-3.0 Audio Tiny

Native MLX inference for [mlx-community/Ming-3.0-audio-tiny-bf16](https://huggingface.co/mlx-community/Ming-3.0-audio-tiny-bf16): English/Chinese transcription, audio understanding, reasoning, text chat and spoken responses. Its hybrid MoE Thinker understands audio; the bundled Qwen2 Talker generates 44.1 kHz speech and loads only when speech output is requested.

| Precision | Repository |
|-----------|------------|
| BF16 | [mlx-community/Ming-3.0-audio-tiny-bf16](https://huggingface.co/mlx-community/Ming-3.0-audio-tiny-bf16) |
| 8-bit | [mlx-community/Ming-3.0-audio-tiny-8bit](https://huggingface.co/mlx-community/Ming-3.0-audio-tiny-8bit) |
| 4-bit | [mlx-community/Ming-3.0-audio-tiny-4bit](https://huggingface.co/mlx-community/Ming-3.0-audio-tiny-4bit) |

Use any of these repositories with the Python and CLI examples below.

```python
from mlx_audio.sts import load
from mlx_audio.audio_io import write

model = load("mlx-community/Ming-3.0-audio-tiny-bf16", strict=True)
result = model.generate(
    "question.wav", task="spoken_dialogue",
    text="Reply briefly in English.", voice="US_James", seed=42,
)
print(result.text)
if result.audio is not None:
    write("answer.wav", result.audio, result.sample_rate)
```

```bash
python -m mlx_audio.sts.generate \
  --model mlx-community/Ming-3.0-audio-tiny-bf16 \
  --audio question.wav --task spoken_dialogue \
  --text "Reply briefly in English." --voice US_James \
  --output-path answer.wav
```

| Task | Result |
|------|--------|
| `asr` | Plain transcription from audio |
| `audio_understanding` | Text answer about supplied audio and text instruction |
| `speech_to_text` | Text response to spoken input |
| `text` | Text conversation without input audio |
| `spoken_dialogue` | Thinker response followed by Talker speech |
| `tts` | Read `text` directly through Talker |

```python
result = model.generate("recording.wav", task="asr")
print(result.text)

result = model.generate(
    text="What is 12 times 13?", task="text", thinking=True,
    max_new_tokens=256,
)
print(result.reasoning, result.text)

result = model.generate(
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Compare these recordings."},
        {"type": "audio", "audio": "first.wav"},
        {"type": "audio", "audio": "second.wav"},
    ]}],
    task="audio_understanding",
)
```

Messages use `system`, `user` and `assistant` roles and must end with a user turn. Audio content accepts file paths, mono/time-major stereo arrays at `sample_rate` (default 16 kHz), or `(array, sample_rate)` pairs. The shipped chat template is applied without executing remote code. CLI `--messages` accepts a JSON file with the same structure, using paths for audio.

`thinking=True` separates reasoning from the final answer. If the token budget expires during reasoning, the answer stays empty and no reasoning is synthesized. Increase `max_new_tokens` to allow more reasoning. `max_talker_steps` bounds speech generation. Built-in voices include `US_James`, `灵小柔` and `灵楷`; `ref_audio` with optional `ref_text` enables reference speech conditioning. Reference speaker extraction uses native MLX CAMPPlus weights, or a precomputed 192-dimensional `spk_emb`. All listed checkpoints include `talker/speaker_encoder/campplus.safetensors`, so no separate model downloads are required. Neither ONNX nor ONNX Runtime is required.

BF16, 8-bit and 4-bit checkpoints use the same loader. Quantization covers the Thinker language backbone and text head; its audio encoder, MoE routers and bundled Talker remain floating point. Streaming and speculative MTP decoding are not implemented. Flash is outside this port's current scope.
