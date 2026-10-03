# MiMo-Audio

MiMo-Audio supports offline text and speech generation on Apple Silicon using MLX. Both the Base and Instruct checkpoints share one implementation, with a separate audio tokenizer supporting all 20 residual codebooks.

| Model | Tasks |
|---|---|
| [MiMo-Audio-7B-Instruct](https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Instruct) | English/Chinese TTS, voice/style prompting, ASR, audio understanding, spoken and text dialogue, multi-turn history |
| [MiMo-Audio-7B-Base](https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Base) | Few-shot speech transformations and raw text/audio completion |
| [MiMo-Audio-Tokenizer](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer) | 24 kHz mono audio encoding and reconstruction |

```python
from mlx_audio.sts import load
from mlx_audio.audio_io import write

model = load("XiaomiMiMo/MiMo-Audio-7B-Instruct", strict=True)
result = model.generate(task="tts", text="Hello from MLX Audio!")
write("hello.wav", result.audio, result.sample_rate)

transcription = model.generate("speech.wav", task="asr")
print(transcription.text)
```

```bash
python -m mlx_audio.sts.generate \
    --model XiaomiMiMo/MiMo-Audio-7B-Instruct \
    --task spoken_dialogue --audio question.wav --output-path answer.wav
```

Generation is offline and processes one example per call. Streaming and server integration are deferred. The separate tokenizer loads lazily for audio tasks; use `audio_tokenizer_path` in `generate()` or `--audio-tokenizer-path` on the CLI for a local copy. Arrays require their sample rate. Output audio is 24 kHz mono.

See the [complete model guide](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/models/mimo_audio/README.md) for task prompts, Base examples, result fields, and history, the [codec guide](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/codec/models/mimo_audio_tokenizer/README.md) for standalone use, and the [conversion guide](../../guides/quantization.md#mimo-audio) for quantization.
