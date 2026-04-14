# ACE-Step for MLX

MLX implementation of [ACE-Step](https://github.com/ace-step/ACE-Step), a 3.5B parameter flow-matching music generation model that can generate high-quality music with vocals from text prompts and lyrics.

## Features

- **Text-to-Music**: Generate music from text descriptions
- **Vocals Support**: Generate songs with lyrics in multiple languages
- **Fast Generation**: Faster than real-time on Apple Silicon (RTF ~0.85x at 20 steps)
- **High Quality**: 48kHz stereo audio output
- **4-bit Quantized Variant**: 2.2GB main model for memory-constrained systems

## Quick Start

```python
from mlx_audio.tts import load

# Load the MLX-converted model (4-bit quantized, 2.2GB)
model = load("mlx-community/ACE-Step1.5-MLX-4bit")

# Generate instrumental music
for result in model.generate(
    text="upbeat electronic dance music with energetic synthesizers",
    duration=30.0,
):
    audio = result.audio  # [samples, 2] stereo audio
    sample_rate = result.sample_rate  # 48000
```

Pre-converted weights are available on Hugging Face:
- [mlx-community/ACE-Step1.5-MLX](https://huggingface.co/mlx-community/ACE-Step1.5-MLX) — fp32 (~9.6GB main model)
- [mlx-community/ACE-Step1.5-MLX-4bit](https://huggingface.co/mlx-community/ACE-Step1.5-MLX-4bit) — 4-bit quantized (~2.2GB main model)

## Generating Music with Vocals

```python
from mlx_audio.tts import load
import scipy.io.wavfile as wavfile
import numpy as np
import mlx.core as mx

model = load("mlx-community/ACE-Step1.5-MLX-4bit")

prompt = "upbeat pop song with female vocals, catchy melody, bright synths"
lyrics = """[Verse 1]
Dance with me tonight
Under the neon lights
Feel the rhythm in your soul
Let the music take control

[Chorus]
We're alive, we're on fire
Dancing higher and higher
Nothing's gonna stop us now
We're shining bright somehow
"""

for result in model.generate(
    text=prompt,
    lyrics=lyrics,
    duration=30.0,
    vocal_language="en",
):
    audio_np = np.array(result.audio.astype(mx.float32))
    audio_int16 = (np.clip(audio_np, -1, 1) * 32767).astype(np.int16)
    wavfile.write("song.wav", result.sample_rate, audio_int16)
```

## How It Works

ACE-Step's turbo model requires a two-stage pipeline:

1. **5Hz Language Model (Planner)**: Takes your text prompt + lyrics and generates an audio code "blueprint" for the song — planning BPM, key signature, structure, and time-aligned audio codes
2. **Diffusion Transformer (DiT)**: Uses these codes as conditioning to denoise random noise into musical latents
3. **VAE Decoder**: Decodes the 25Hz latents into 48kHz stereo audio

The LM planner is **required** for the turbo model — without it, the DiT converges back to silence. `use_lm=True` is the default.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `text` | required | Text description / prompt |
| `lyrics` | `""` | Lyrics (empty for instrumental) |
| `duration` | `30.0` | Target duration in seconds |
| `seed` | `None` | Random seed for reproducibility |
| `num_steps` | `20` | Diffusion steps (20 recommended for vocals, 8 ok for instrumentals) |
| `shift` | `3.0` | Timestep schedule shift (1.0, 2.0, or 3.0) |
| `guidance_scale` | `1.0` | CFG scale (1.0 = no guidance; turbo model is distilled without CFG) |
| `guidance_interval` | `0.5` | Fraction of steps with guidance applied |
| `cfg_type` | `"apg"` | Guidance type: `"apg"` or `"cfg"` |
| `vocal_language` | `"unknown"` | Language code: `"en"`, `"zh"`, `"ja"`, etc. |
| `use_lm` | `True` | Use 5Hz LM planner (required for the turbo model) |
| `lm_model_size` | `"0.6B"` | LM size: `"0.6B"` (fast) or `"4B"` (higher quality, slower) |

## Lyrics Format

Use section markers to structure your lyrics:

```
[Verse 1]
First verse lyrics here
Line by line

[Chorus]
Catchy chorus lyrics
That repeat

[Bridge]
Bridge section
Something different

[Outro]
Final words
```

## Supported Languages

ACE-Step supports lyrics in multiple languages via the 5Hz LM planner:
- English (`en`), Chinese (`zh`), Japanese (`ja`), Korean (`ko`)
- Spanish (`es`), French (`fr`), German (`de`), Italian (`it`)
- And many more (50+ supported by the LM)

Set `vocal_language` to hint the planner — though the 0.6B LM sometimes picks its own language based on caption/lyrics content. Use the 4B LM for more reliable language adherence.

## Performance

Apple Silicon M4 Max, 4-bit quantized model, 20 diffusion steps:

| Duration | Total Time | Diffusion | LM Planning | VAE Decode | RTF |
|----------|-----------|-----------|-------------|------------|-----|
| 30s | ~25s | ~8s | ~12s | ~3.5s | 0.85x |
| 60s | ~40s | ~17s | ~12s | ~6s | 0.67x |

*RTF (Real-Time Factor) < 1.0 means faster than real-time.*

## Tips

1. **Vocals**: Use `num_steps=20` (default) for clearer vocals; `num_steps=8` is fine for instrumentals
2. **Cherry-pick seeds**: Generate a few samples with different seeds and pick the best — LM quality varies
3. **Prompt engineering**: Be specific about genre, instruments, mood, vocal style (e.g., "male vocals", "female choir")
4. **Section markers** in lyrics help the model structure the song (`[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`)
5. **LM size**: 4B model follows language/caption instructions more reliably but takes ~3x longer to plan

## Known Limitations

- The 0.6B LM occasionally ignores the `vocal_language` parameter and picks its own
- Vocal clarity varies by seed — use cherry-picking for final output
- Non-Western genres (e.g., Bollywood) are less represented in training data

## Citation

```bibtex
@article{ace-step,
  title={ACE-Step: A Step Towards Music Generation Foundation Model},
  author={ACE-Step Team},
  year={2024}
}
```

## License

This implementation follows the mlx-audio project license. The ACE-Step model weights are subject to their original license from the ACE-Step team.
