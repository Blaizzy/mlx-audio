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
