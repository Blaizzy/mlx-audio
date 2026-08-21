# NVIDIA NemotronLabs VoiceChat

MLX support for `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`, a full-duplex
speech-to-speech model with a 16 kHz input and 22.05 kHz output.

Pre-converted checkpoints are available on Hugging Face:

- [mlx-community/NemotronLabs-VoiceChat-11B-4bit](https://huggingface.co/mlx-community/NemotronLabs-VoiceChat-11B-4bit)
- [mlx-community/NemotronLabs-VoiceChat-11B-8bit](https://huggingface.co/mlx-community/NemotronLabs-VoiceChat-11B-8bit)
- [mlx-community/NemotronLabs-VoiceChat-11B-bf16](https://huggingface.co/mlx-community/NemotronLabs-VoiceChat-11B-bf16)

Run offline inference:

```python
from mlx_audio.sts import load

model = load("mlx-community/NemotronLabs-VoiceChat-11B-4bit")
output = model.generate("input.wav")
print(output.text)
output.audio
```

## Converting from source

The original checkpoint is a 44 GB NeMo export. To produce a custom
quantization instead of using a pre-converted checkpoint above, prepare an
MLX artifact yourself:

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
perception, TTS, and codec weights at full precision. `load()` accepts this
local output directory the same way it accepts a Hugging Face model id.

For full-duplex inference, keep one streaming session alive and feed mono 16 kHz
PCM as it arrives:

```python
session = model.create_duplex_session()

for chunk in microphone_chunks:
    for event in session.push_audio(chunk, sample_rate=16_000):
        if event.kind == "assistant_text_delta":
            print(event.delta, end="", flush=True)
        elif event.kind == "audio":
            play(event.samples, event.sample_rate)

session.flush()
```

The session buffers arbitrary chunk sizes and emits aligned user transcripts,
assistant text, function tokens, and 22.05 kHz speech on the model's 80 ms
timeline.
