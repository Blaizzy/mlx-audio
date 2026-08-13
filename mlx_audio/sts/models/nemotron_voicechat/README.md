# NVIDIA NemotronLabs VoiceChat

MLX support for `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`, a full-duplex
speech-to-speech model with a 16 kHz input and 22.05 kHz output.

The original checkpoint is a 44 GB NeMo export. Prepare an MLX artifact before
loading it:

```bash
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
  --revision 5631f538c74d1b4a8adfbc0b3a2c4aed6eba4d56 \
  --local-dir ./nemotron-voicechat-source

python -m mlx_audio.sts.models.nemotron_voicechat.convert \
  --source ./nemotron-voicechat-source \
  --output ./nemotron-voicechat-4bit \
  --quantize
```

The 4-bit conversion targets the Nemotron language path while keeping speech
perception, TTS, and codec weights at full precision.

Run offline inference:

```python
from mlx_audio.sts import load

model = load("./nemotron-voicechat-4bit")
output = model.generate("input.wav")
print(output.text)
output.audio
```

The default system prompt and the checkpoint's pre-baked `Aria` speaker latent
match NVIDIA's reference offline inference path.
