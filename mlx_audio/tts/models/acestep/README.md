# ACE-Step

MLX implementation of [ACE-Step](https://github.com/ace-step/ACE-Step-1.5) by StepFun.
ACE-Step is a highly efficient open-source music foundation model capable of generating full songs with vocals, lyrics, and rich instrumentation directly from text prompts.

## Features

- **Text-to-Audio**: Generate high-fidelity music (up to 10 minutes) from text descriptions.
- **Fast Generation**: Uses a native MLX Diffusion Transformer (DiT) combined with a VAE.
- **Lyrics Support**: Conditions music generation on precise lyrics.

## Usage

```python
from mlx_audio.tts import load
import sounddevice as sd
import numpy as np

# Load model natively in MLX
model = load("acestep-mlx")

text_prompt = "A high-energy electronic dance music track with heavy bass and synth drops"

# Generate 30 seconds of music
audio_generator = model.generate(
    text=text_prompt,
    duration=30.0,
    steps=50,
    guidance_scale=4.5
)

# Iterate the generator (ACE-Step generates in one diffusion pass, so it yields once)
for audio_array in audio_generator:
    print(f"Generated {audio_array.shape[-1] / 48000:.1f} seconds of audio!")
    
    # Play using sounddevice (ACE-Step VAE outputs 48kHz audio)
    pcm = np.array(audio_array[0, 0, :])
    sd.play(pcm, samplerate=48000)
    sd.wait()
```

## Model Preparation

ACE-Step weights must be converted to pure MLX format before loading. 

Run the conversion script:
```bash
python -m mlx_audio.tts.models.acestep.convert --model ACE-Step/acestep-v15-turbo --output acestep-mlx
```
This requires `transformers` and `diffusers` installed temporarily just for the conversion step.
