# Confucius4-TTS

Multilingual, cross-lingual, zero-shot voice-cloning TTS, ported to MLX from
[netease-youdao/Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS)
(Apache-2.0).

Pipeline: w2v-bert semantic features + CAMPPlus speaker embedding → T2S (GPT-2)
→ S2A conditional flow-matching (DiT + WaveNet) → BigVGAN v2 vocoder.

## Voice Cloning

Clone any voice from a short reference clip. Pass the text, the reference wav,
and the target language code:

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Confucius4-TTS-mlx")
results = list(model.generate(
    text="Xin chào, đây là giọng nói được nhân bản.",
    ref_audio="reference.wav",   # any sample rate; decoded + resampled to 16 kHz
    lang="vi",
))

audio = results[0].audio              # mx.array
sample_rate = results[0].sample_rate  # 22050
```

Save the result with the bundled audio I/O helper:

```python
import numpy as np
from mlx_audio.audio_io import write

write("output.wav", np.array(results[0].audio), results[0].sample_rate)
```

## Languages

`lang` accepts: `zh`, `en`, `vi`, `ja`, `ko`, `th`.

## Available Models

| Model | Precision | Description |
|-------|-----------|-------------|
| `mlx-community/Confucius4-TTS-mlx`      | fp32 | Full-quality weights |
| `mlx-community/Confucius4-TTS-mlx-int8` | int8 | ~60% smaller, faster on Apple Silicon |

## Converting Weights

Convert the original PyTorch checkpoint to MLX:

```bash
python -m mlx_audio.tts.models.confucius4.convert --out ./confucius4-model
```

Add `--quantize int8` (or `int4`) to quantize the T2S body matmuls.

## Notes

- No `torch`/`transformers` in the inference path; `torch` is used only in
  `convert.py` for weight conversion.
- Reference audio is decoded and resampled to 16 kHz mono internally via the
  repo's audio I/O and resampling utilities.
```
