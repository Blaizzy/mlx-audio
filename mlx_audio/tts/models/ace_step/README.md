# ACE-Step for MLX

MLX implementation of [ACE-Step](https://github.com/ace-step/ACE-Step), a 3.5B parameter flow-matching music generation model that can generate high-quality music with vocals from text prompts and lyrics.

## Features

- **Text-to-Music**: Generate music from text descriptions
- **Vocals Support**: Generate songs with lyrics in multiple languages
- **Fast Generation**: Turbo model generates 2 minutes of audio in ~25 seconds on Apple Silicon
- **High Quality**: 48kHz stereo audio output

## Quick Start

```python
from mlx_audio.tts import load

# Load the model
model = load("ACE-Step/ACE-Step1.5")

# Generate instrumental music
for result in model.generate(
    text="upbeat electronic dance music with energetic synthesizers",
    duration=30.0,
):
    audio = result.audio  # [2, samples] stereo audio
    sample_rate = result.sample_rate  # 48000
```

## Generating Music with Vocals

```python
from mlx_audio.tts import load
import scipy.io.wavfile as wavfile
import numpy as np
import mlx.core as mx

model = load("ACE-Step/ACE-Step1.5")

prompt = "upbeat electronic dance music with female vocals, catchy melody"
lyrics = """[verse]
Dance with me tonight
Under the neon lights
Feel the beat so right

[chorus]
Move your body, feel the groove
Nothing left for us to prove
Just the music in our soul
"""

for result in model.generate(
    text=prompt,
    lyrics=lyrics,
    duration=60.0,
    guidance_scale=1.0,
    vocal_language="en",
):
    # Save to WAV file
    audio_np = np.array(result.audio.astype(mx.float32))
    audio_int16 = (audio_np * 32767).astype(np.int16)
    wavfile.write("song.wav", result.sample_rate, audio_int16.T)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `text` | required | Text description/prompt for the music |
| `lyrics` | `""` | Lyrics for vocal generation (empty for instrumental) |
| `duration` | `30.0` | Target duration in seconds |
| `seed` | `None` | Random seed for reproducibility |
| `num_steps` | `8` | Number of diffusion steps (8 for turbo) |
| `guidance_scale` | `1.0` | CFG scale (1.0 = no guidance for turbo model) |
| `guidance_interval` | `0.5` | Fraction of steps with guidance applied |
| `cfg_type` | `"apg"` | CFG type: `"apg"` or `"cfg"` |
| `vocal_language` | `"unknown"` | Language code: `"en"`, `"zh"`, `"ja"`, etc. |

## Recommended Settings

**Note:** The turbo model (ACE-Step1.5) is distilled to work without classifier-free guidance.
The default `guidance_scale=1.0` (no guidance) is recommended for best results.

### For Vocals
```python
model.generate(
    text="...",
    lyrics="...",
    guidance_scale=5.0,      # 4.0-6.0 for clear vocals
    guidance_interval=0.5,   # 0.5-0.7 for lyric adherence
    vocal_language="en",
)
```

### For Instrumental
```python
model.generate(
    text="...",
    lyrics="",               # Empty for instrumental
)
```

### For Better Lyric Following
If the model skips lyrics, try:
```python
model.generate(
    text="...",
    lyrics="...",
    guidance_interval=0.7,   # Higher = stricter lyric following
    num_steps=16,            # More steps for refinement
)
```

## Lyrics Format

Use section markers to structure your lyrics:

```
[verse]
First verse lyrics here
Line by line

[chorus]
Catchy chorus lyrics
That repeat

[bridge]
Bridge section
Something different

[outro]
Final words
```

## Supported Languages

ACE-Step supports lyrics in multiple languages:
- English (`en`)
- Chinese (`zh`)
- Japanese (`ja`)
- Korean (`ko`)
- And more...

Set `vocal_language` to the appropriate code for best results.

## Performance

On Apple Silicon (M-series chips):

| Duration | Generation Time | RTF |
|----------|-----------------|-----|
| 30s | ~12s | 0.4x |
| 60s | ~15s | 0.25x |
| 120s | ~25s | 0.2x |

*RTF (Real-Time Factor) < 1.0 means faster than real-time*

## Tips

1. **Cherry-picking**: Generate multiple samples with different seeds and pick the best one
2. **Prompt engineering**: Be specific about genre, instruments, mood, and vocal style
3. **Lyric length**: Match lyric length to duration (~4 lines per 15-20 seconds)
4. **Turbo model**: Uses no CFG by default (guidance_scale=1.0) as it's distilled for direct generation

## Known Limitations

- May occasionally skip or mix up lyrics in longer songs
- Vocal clarity varies by generation (use cherry-picking)
- Best results with clear, structured lyrics using section markers

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
