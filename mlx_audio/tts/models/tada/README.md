# TADA

Text-Acoustic Dual Alignment TTS by [HumeAI](https://huggingface.co/collections/HumeAI/tada). A unified speech-language model that synchronizes speech and text into a single stream via 1:1 alignment, enabling high-fidelity speech synthesis with reduced computational overhead. Built on Llama 3.2.

**Paper:** [arXiv 2602.23068](https://arxiv.org/abs/2602.23068)

## Voice Cloning

Clone any voice using a reference audio sample and its transcript:

```python
from mlx_audio.tts.utils import load_model
import sounddevice as sd

model = load_model("HumeAI/mlx-tada-1b")

result = next(model.generate(
    text="Please call Stella. Ask her to bring these things with her from the store.",
    ref_audio="reference.wav",
    ref_text="The examination and testimony of the experts enabled the commission to conclude.",
))
sd.play(result.audio, result.sample_rate)
sd.wait()
```

## Zero-Shot Generation

Generate speech without a reference audio:

```python
result = next(model.generate(
    text="Hello, welcome to TADA text to speech.",
))
```

## Speed Control

Control speech speed with `speed_up_factor` (two-pass generation):

```python
# Faster speech (1.5x)
result = next(model.generate(
    text="This will be spoken quickly.",
    ref_audio="reference.wav",
    ref_text="Reference transcript.",
    speed_up_factor=1.5,
))

# Slower speech (0.75x)
result = next(model.generate(
    text="This will be spoken slowly.",
    ref_audio="reference.wav",
    ref_text="Reference transcript.",
    speed_up_factor=0.75,
))
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.6 | Text sampling temperature |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `acoustic_cfg_scale` | 1.6 | CFG scale for acoustic features |
| `duration_cfg_scale` | 1.0 | CFG scale for duration |
| `noise_temperature` | 0.9 | Initial noise scaling |
| `num_flow_matching_steps` | 20 | Number of ODE solver steps |
| `time_schedule` | `logsnr` | Time schedule (`logsnr`, `cosine`, `linear`) |
| `cfg_schedule` | `cosine` | CFG schedule (`cosine`, `linear`, `constant`) |
| `max_tokens` | 1024 | Maximum generation steps |
| `speed_up_factor` | `None` | Speed control (>1 faster, <1 slower) |

## CLI

```bash
python -m mlx_audio.tts.generate \
  --model HumeAI/mlx-tada-1b \
  --text "Please call Stella." \
  --ref-audio reference.wav \
  --ref-text "The examination and testimony of the experts."
```

## Available Models

| Model | Parameters | Languages |
|-------|-----------|-----------|
| `HumeAI/mlx-tada-1b` | 2B | English |
| `HumeAI/mlx-tada-3b` | 4B | English + ar, zh, de, es, fr, it, ja, pl, pt |

## License

TADA weights are released under the [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE).
