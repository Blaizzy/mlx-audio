# Ming Omni Dense TTS (0.5B)

This implementation powers `model_type: "dense"` for Ming Omni TTS in MLX-Audio.

## Supported model

- `inclusionAI/Ming-omni-tts-0.5B`

## Run with CLI

```bash
uv run mlx_audio.tts.generate \
  --model inclusionAI/Ming-omni-tts-0.5B \
  --text "Simply put, this was equivalent to handing over the consumer market to competitors." \
  --ref_audio /Users/prince_canuma/Downloads/conversational_a.wav \
  --instruct "Speak quickly, with medium pitch and higher volume." \
  --cfg_scale 2.0 \
  --sigma 0.25 \
  --temperature 0.0 \
  --max_tokens 200 \
  --lang_code en \
  --output_path "./" \
  --file_prefix en_02_basic \
  --verbose
```

## Python usage

```python
from pathlib import Path
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

model = load_model("inclusionAI/Ming-omni-tts-0.5B")

result = next(
    model.generate(
        text="Simply put, this was equivalent to handing over the consumer market to competitors.",
        ref_audio="/Users/prince_canuma/Downloads/conversational_a.wav",
        instruct="Speak quickly, with medium pitch and higher volume.",
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        max_tokens=200,
        lang_code="en",
    )
)

out = Path("en_02_basic_000.wav")
audio_write(str(out), np.array(result.audio), result.sample_rate, format="wav")
print(out)
```

## Notes

- `--ref_text` is optional. If omitted, MLX-Audio transcribes `--ref_audio` automatically.
- If you already have exact transcript text for the reference clip, pass `--ref_text` for more stable voice cloning.
- For additional cookbook examples and advanced options, see the [Ming Omni TTS README](../bailingmm/README.md).
