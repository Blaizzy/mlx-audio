# Breeze TTS 2

[Breeze TTS 2](https://huggingface.co/BreezeBlue/breeze-tts-2) is an
English/Chinese text-to-speech model for voice design, voice cloning, and voice
direction. The MLX implementation loads the publisher's original checkpoint,
including its bundled Qwen3-TTS audio codec, and emits 24 kHz mono audio.

## Install and load

```bash
pip install -U mlx-audio
```

The model is registered under `breeze`, `breeze-tts`, and `breeze_tts`; the
publisher's `BreezeBlue/breeze-tts-2` repository resolves automatically:

```python
from mlx_audio.tts import load

model = load("BreezeBlue/breeze-tts-2")
```

## Voice design

Create a voice from a natural-language description without reference audio.
The upstream example flags `--instruction` and `--cfg-scale` are accepted;
the established MLX spellings `--instruct` and `--cfg_scale` are equivalent.

```bash
mlx_audio.tts.generate \
  --model BreezeBlue/breeze-tts-2 \
  --text "(sigh) Welcome aboard. Your journey begins now." \
  --instruction "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery." \
  --cfg-scale 4 \
  --output_path outputs \
  --file_prefix breeze_design
```

## Voice clone

Clone one speaker from a clean reference clip and its exact transcript. The
reference aliases `--ref-audio` and `--ref-text` are equivalent to
`--ref_audio` and `--ref_text`.

```bash
mlx_audio.tts.generate \
  --model BreezeBlue/breeze-tts-2 \
  --ref-audio reference_en.wav \
  --ref-text "This is the exact transcript of the English reference audio." \
  --text "(sigh) It is good to hear your voice again after all this time." \
  --voice S0 \
  --output_path outputs \
  --file_prefix breeze_clone
```

Exactly one reference audio/transcript pair is supported per generation. The
transcript should match the clip; incomplete pairs are rejected.

## Voice direction

Combine a reference pair with an instruction to preserve speaker identity
while directing tone, emotion, pace, or delivery:

```bash
mlx_audio.tts.generate \
  --model BreezeBlue/breeze-tts-2 \
  --ref_audio reference.wav \
  --ref_text "This is the exact transcript of the reference audio." \
  --text "(clears throat) We need to discuss what happened last night." \
  --instruct "Speak slowly with a restrained, serious tone." \
  --cfg_scale 4 \
  --voice S0 \
  --output_path outputs \
  --file_prefix breeze_direction
```

`S0` is Breeze's default speaker tag; pass `--voice S0` explicitly when you
want the generated prompt to show it.

## Inline vocal events

Events remain inline in `--text`. Use parentheses for English, such as
`(laugh)`, `(cough)`, `(clears throat)`, and `(sigh)`. Use square brackets for
Chinese, such as `[笑]`, `[咳嗽]`, `[清嗓子]`, and `[叹气]`:

```bash
mlx_audio.tts.generate \
  --model BreezeBlue/breeze-tts-2 \
  --text "[笑] 欢迎来到今晚的故事时间，让我们一起开始吧。" \
  --instruction "一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。" \
  --cfg-scale 4 \
  --output_path outputs \
  --file_prefix breeze_events
```

## Streaming

Pass `--stream` to play incremental codec-decoded chunks. Add `--save` to
join them into a WAV file, and set `--streaming_interval` in seconds to choose
the chunk cadence:

```bash
mlx_audio.tts.generate \
  --model BreezeBlue/breeze-tts-2 \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --text "(sigh) This response is streamed as it is generated." \
  --stream \
  --streaming_interval 1.0 \
  --save \
  --output_path outputs \
  --file_prefix breeze_stream
```

The Python API yields `GenerationResult` objects for the same flow:

```python
from mlx_audio.tts import load

model = load("BreezeBlue/breeze-tts-2")
for result in model.generate(
    text="(laugh) Hello from MLX-Audio.",
    instruct="A bright, friendly voice.",
    cfg_scale=4,
    stream=True,
    streaming_interval=1.0,
):
    assert result.sample_rate == 24_000
    assert result.audio.ndim == 1
```

## License

The Breeze source code is Apache-2.0, but the model weights, derivatives, and
self-hosted output are governed by BreezeBlue's Research and Non-Commercial
License. Review the [original model card](https://huggingface.co/BreezeBlue/breeze-tts-2)
before use.
