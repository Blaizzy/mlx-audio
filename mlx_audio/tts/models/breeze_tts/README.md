# Breeze TTS 2

MLX implementation of [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/breeze-tts-2).
The original checkpoint bundles the text encoder and Qwen3-TTS codec used by
this implementation, so loading it does not require the upstream PyTorch
runtime. Its PyTorch safetensors do not load in MLX directly; the examples
use an [unofficial MLX 4-bit
conversion](https://huggingface.co/LunaFox/Breeze-TTS-2-mlx-4bit).

## Voice design

Install `mlx-audio`, then use the `mlx_audio.tts.generate` command with the
model repository:

```bash
mlx_audio.tts.generate \
  --model LunaFox/Breeze-TTS-2-mlx-4bit \
  --text "Welcome aboard. Your journey begins now." \
  --instruction "A warm, thoughtful young woman with a clear voice." \
  --cfg-scale 4
```

The upstream example spellings above are aliases for the established
underscore options `--instruct` and `--cfg_scale`.

## Voice clone

Provide one clean reference clip and its exact transcript. Inline vocal events
such as `(sigh)` are kept in the generated text.

```bash
mlx_audio.tts.generate \
  --model LunaFox/Breeze-TTS-2-mlx-4bit \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --text "(sigh) It is good to hear your voice again." \
  --output_path outputs \
  --file_prefix breeze_clone
```

The aliases `--ref-audio` and `--ref-text` are also available as
`--ref_audio` and `--ref_text`. Breeze accepts one reference pair per
generation and produces 24 kHz mono audio.

## Voice direction

Combine a reference pair with an instruction to preserve speaker identity
while directing tone, emotion, pace, or delivery:

```bash
mlx_audio.tts.generate \
  --model LunaFox/Breeze-TTS-2-mlx-4bit \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --text "(clears throat) We need to discuss what happened." \
  --instruction "Speak slowly with a restrained, serious tone." \
  --cfg-scale 4 \
  --voice S0
```

Voice design works without references. `S0` is the default speaker tag; pass
`--voice S0` explicitly when making the prompt self-documenting.

## Events and streaming

English events use parentheses, for example `(laugh)`, `(cough)`, and
`(clears throat)`. Chinese events use square brackets, for example `[笑]` and
`[叹气]`:

```bash
mlx_audio.tts.generate \
  --model LunaFox/Breeze-TTS-2-mlx-4bit \
  --text "[笑] 欢迎来到今晚的故事时间，让我们一起开始吧。" \
  --instruction "一位温柔自信的年轻女性，声音清晰，语气亲切。" \
  --cfg-scale 4
```

Use `--stream` for incremental chunks. Add `--save` to join the chunks into a
WAV file, and adjust `--streaming_interval` (in seconds) for the requested
cadence:

```bash
mlx_audio.tts.generate \
  --model LunaFox/Breeze-TTS-2-mlx-4bit \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --text "(sigh) This response is streamed as it is generated." \
  --stream \
  --streaming_interval 1.0 \
  --save \
  --output_path outputs \
  --file_prefix breeze_stream
```

The model API yields `GenerationResult` chunks with `sample_rate == 24000` and
mono waveforms:

```python
from mlx_audio.tts import load

model = load("LunaFox/Breeze-TTS-2-mlx-4bit")
for chunk in model.generate(
    text="(laugh) Hello from MLX-Audio.",
    instruct="A bright, friendly voice.",
    cfg_scale=4,
    stream=True,
    streaming_interval=1.0,
):
    assert chunk.sample_rate == 24_000
    print(chunk.audio.shape)
```

The model weights, derivatives, and generated outputs are governed by the
upstream BreezeBlue Research and Non-Commercial License. Review the [original
model card](https://huggingface.co/BreezeBlue/breeze-tts-2) before use.
