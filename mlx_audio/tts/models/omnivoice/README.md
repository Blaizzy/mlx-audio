# OmniVoice

Zero-shot multilingual TTS supporting 646+ languages, with optional voice cloning
from a reference audio file. Based on [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice).

Architecture: bidirectional Qwen3-0.6B backbone with iterative masked diffusion decoding
(non-autoregressive) and HiggsAudioV2 acoustic tokenizer (24 kHz, 25 tokens/sec).

## Model

| Repo ID | Description |
|---------|-------------|
| `k2-fsa/OmniVoice` | Original weights (bfloat16) |

## Usage

### Python API

```python
from transformers import AutoTokenizer
from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer
from mlx_audio.tts.models.omnivoice.omnivoice import Model, OmniVoiceConfig
from mlx_audio.audio_io import write as audio_write
import mlx.core as mx, json, numpy as np

model_path = "path/to/OmniVoice"

# Load model
with open(f"{model_path}/config.json") as f:
    raw = json.load(f)
config = OmniVoiceConfig.from_dict(raw)
model = Model(config)
weights = model.sanitize(dict(mx.load(f"{model_path}/model.safetensors")))
model.load_weights(list(weights.items()))
mx.eval(model.parameters())

# Load tokenizers
tokenizer = HiggsAudioTokenizer.from_pretrained(model_path)
text_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")

# Generate speech
result = next(model.generate(
    text="Hello, this is OmniVoice running on Apple Silicon.",
    language="en",
    duration_s=5.0,
    tokenizer=tokenizer,
    text_tokenizer=text_tokenizer,
))
audio_write("output.wav", np.array(result.audio), result.sample_rate)
```

### Voice cloning

```python
from mlx_audio.audio_io import write as audio_write
import numpy as np

result = next(model.generate(
    text="Hello, this is OmniVoice.",
    language="en",
    duration_s=5.0,
    ref_audio="reference.wav",   # any sample rate, clipped to 10s
    tokenizer=tokenizer,
    text_tokenizer=text_tokenizer,
))
audio_write("output_cloned.wav", np.array(result.audio), result.sample_rate)
```

### CLI

```bash
python -m mlx_audio.tts.generate \
  --model path/to/OmniVoice \
  --text "Hello, this is OmniVoice." \
  --duration_s 5.0 \
  --language en
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `text` | — | Text to synthesize |
| `language` | `"None"` | BCP-47 tag: `"en"`, `"ru"`, `"zh"`, `"ja"` ... `"None"` = auto-detect |
| `instruct` | `"None"` | Style instruction tag (model-specific) |
| `duration_s` | `5.0` | Maximum output duration in seconds |
| `ref_audio` | `None` | Path to reference WAV for voice cloning (any SR) |
| `ref_audio_max_duration_s` | `10.0` | Reference audio clip limit in seconds |
| `num_steps` | `32` | Iterative unmasking steps (4-64; fewer = faster, lower quality) |
| `guidance_scale` | `2.0` | CFG strength (`0` disables CFG) |
| `class_temperature` | `0.0` | Token sampling temperature (`0` = greedy) |
| `position_temperature` | `5.0` | Gumbel noise on position scores |

## Supported languages

646+ languages via language tag. Best quality for:
`en`, `zh`, `ru`, `ja`, `de`, `fr`, `es`, `ko`, `ar`, `pt`, `it`, `pl`, `nl`, `tr`

## Architecture

```
Text tokens  --> backbone (bidirectional Qwen3-0.6B) --> 8 prediction heads
Ref tokens   -->                                              |
MASK tokens  -->                               iterative unmasking (32 steps)
                                                              |
                                             HiggsAudioV2 decoder --> 24kHz WAV
```

- **Backbone**: Qwen3-0.6B with causal mask disabled (bidirectional attention)
- **Audio tokens**: 8 codebooks x 1025 vocab (0-1023 = codec, 1024 = MASK)
- **Generation**: iterative masked diffusion with cosine-shifted schedule and CFG
- **Tokenizer decode**: MLX (fast); encode for voice cloning: PyTorch CPU (~0.5s/1s audio)

## Notes

- Voice cloning requires `torchaudio` and `transformers` (`pip install torchaudio transformers`)
- The encode (voice cloning) path runs on CPU via PyTorch; decode runs in MLX on GPU
- Output quality improves with longer `num_steps`; 16 steps give reasonable speed/quality
