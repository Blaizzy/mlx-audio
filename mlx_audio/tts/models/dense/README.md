# Ming Dense TTS (0.5B)

This implementation powers `model_type: "dense"`, the Qwen2.5-0.5B Ming checkpoints:

- [Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B): `mlx-community/Ming-3.0-tts-0.5B-bf16`
- [Ming-omni-tts-0.5B](https://huggingface.co/inclusionAI/Ming-omni-tts-0.5B): `mlx-community/Ming-omni-tts-0.5B-bf16`

Both checkpoints share one config. Ming-3.0 is detected by its one-latent patches (`ditar_config.patch_size == 1`). It gets the Ming-3.0 prompt contract:
- the caption ends with `<|im_end|>`;
- the reference clip is padded at the front;
- text is split into fragments of about 24 words or Chinese characters;
- each fragment's trailing silence is trimmed;
- a `Target language:` line is added for `de`, `es`, `fr`, `ja` and `ko`.

## Ming-3.0-tts-0.5B

Ming-3.0 speaks Chinese, English, German, Spanish, French, Korean and Japanese, plus Sichuanese and Cantonese dialects. It outputs 44.1 kHz audio. On an M4 Pro it runs at about real time and peaks at about 4.4 GB.

### Text to speech

```bash
mlx_audio.tts.generate \
  --model mlx-community/Ming-3.0-tts-0.5B-bf16 \
  --text "The evening wind blew in from the mountain pass, and the distant clouds slowly turned gold." \
  --output_path ./ --file_prefix ming3
```

For German, Spanish, French, Korean or Japanese, pass `--lang_code de` (or `es`, `fr`, `ko`, `ja`).

### Voice cloning

Pass a reference clip with its transcript. The reference audio is only used together with `--ref_text`.

```bash
mlx_audio.tts.generate \
  --model mlx-community/Ming-3.0-tts-0.5B-bf16 \
  --text "I never thought a little machine could sound this much like me." \
  --ref_audio reference.wav \
  --ref_text "Exact transcript of reference.wav." \
  --output_path ./ --file_prefix ming3_clone
```

### Built-in voices

`--voice` selects one of the built-in voices: `US_James`, `US_Emma`, `UK_George`, `UK_Charlotte`, `US_Stella`, or one of the 100+ Chinese voices such as `灵小柔`. The full list is in the [Ming-3.0 voice catalog](https://github.com/inclusionAI/Ming-3.0-tts/blob/main/cookbooks/instruction_en.md#built-in-premium-voices).

```bash
mlx_audio.tts.generate \
  --model mlx-community/Ming-3.0-tts-0.5B-bf16 \
  --text "Good evening, and welcome to the late news." \
  --voice UK_Charlotte \
  --output_path ./ --file_prefix ming3_voice
```

### Freeform instructions

Describe the voice and the delivery in one sentence, with an empty speaker slot:

```bash
mlx_audio.tts.generate \
  --model mlx-community/Ming-3.0-tts-0.5B-bf16 \
  --text "The old radio crackled softly as the storm moved across the valley." \
  --instruct "Speak in a warm, intimate storytelling voice, with a gentle pace and a hint of nostalgia." \
  --use_zero_spk_emb \
  --output_path ./ --file_prefix ming3_freeform
```

### Paralinguistic tags

The tags work inline in Chinese or English:

`[Breathing]`, `[Cough]`, `[Sigh]`, `[Laughter]`, `[Surprise-ah]`, `[Surprise-oh]`, `[Dissatisfaction-hnn]`, `[Uhm]`, `[Confirmation-en]`, `[Question-ei]`, `[Question-ah]`, `[Question-en]`, `[Question-oh]`, `[Question-huh]`

```bash
mlx_audio.tts.generate \
  --model mlx-community/Ming-3.0-tts-0.5B-bf16 \
  --text "[Sigh] The rain still hasn't stopped. [Cough] I guess we'll have to move the party indoors." \
  --output_path ./ --file_prefix ming3_tags
```

### Emotion, speed, pitch, volume, dialect and voice design

These controls are JSON captions with a fixed key order. `ming3_instruction()` builds them, and you pass the result as `instruct`:
- **Emotion:** `高兴 / 悲伤 / 愤怒 / 恐惧 / 厌恶 / 惊讶 / 中性`
- **Speed:** `慢速 / 中速 / 快速`
- **Pitch and volume:** `低 / 中 / 高`
- **Dialect:** `cantonese` or `chuanyu dialect`
- **Voice design (`style`):** a free-text Chinese description, used with `use_zero_spk_emb=True`

```python
from mlx_audio.tts.models.bailingmm.bailingmm import ming3_instruction
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Ming-3.0-tts-0.5B-bf16")

result = next(
    model.generate(
        text="所有测试都顺利通过了，我们终于可以把这个好消息告诉大家！",
        instruct=ming3_instruction(emotion="高兴"),
        ref_audio="reference.wav",
        ref_text="诶，你今天有空吗？我们能不能聊聊这件事啊？",
    )
)

design = next(
    model.generate(
        text="This is the story of a river that forgot where it came from.",
        instruct=ming3_instruction(style="成熟沉稳的男性纪录片旁白，吐字清晰，节奏舒缓，语气理性而有厚度。"),
        use_zero_spk_emb=True,
    )
)
```

### Limitations

- **Speaker embeddings.** Automatic CAM++ speaker embedding extraction is not implemented. Cloning uses the supplied audio and transcript. If you have a precomputed 192-dim embedding, pass it as `spk_emb=`.
- **Freeform and voice-design length.** Freeform and voice-design text is kept in one fragment, capped by the 512-position budget, about 30 seconds of speech.
- **Text normalization.** Numbers, dates and units are not expanded automatically. Write numbers out as words for the most reliable reading.
- **Generation and stopping.** Ming-3.0 defaults to CFG 2.0, sigma 0.25, temperature 0.0, and 10 flow steps. The stop head must exceed 0.5 after the first four patches; its final patch is included. `max_tokens` (default 1000) or `max_decode_steps` sets the requested budget, bounded by the remaining 512-position cache. Trailing silence is trimmed. Some generations can still have long tails; re-run with another seed if that happens.

Pass `raw_text=True` to skip text splitting and punctuation changes. `trim_silence=False` disables the trailing-silence trim.

## Ming-omni-tts-0.5B

```bash
mlx_audio.tts.generate \
  --model mlx-community/Ming-omni-tts-0.5B-bf16 \
  --text "Simply put, this was equivalent to handing over the consumer market to competitors." \
  --ref_audio conversational_a.wav \
  --instruct "Speak quickly, with medium pitch and higher volume." \
  --output_path ./ --file_prefix ming_omni_dense
```

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Ming-omni-tts-0.5B-bf16")

result = next(
    model.generate(
        text="Simply put, this was equivalent to handing over the consumer market to competitors.",
        ref_audio="conversational_a.wav",
        instruct="Speak quickly, with medium pitch and higher volume.",
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=200,
    )
)

out = Path("en_02_basic_000.wav")
audio_write(str(out), np.array(result.audio), result.sample_rate, format="wav")
print(out)
```

## Notes

- `--ref_text` is optional. If you omit it, MLX-Audio transcribes `--ref_audio` automatically. Passing the exact transcript gives more stable cloning.
- For more cookbook examples and advanced options, see the [Ming Omni TTS README](../bailingmm/README.md).
