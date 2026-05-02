# Silero VAD

MLX port of the small Silero voice activity detector.

## Supported Model

[`mlx-community/silero-vad`](https://huggingface.co/mlx-community/silero-vad)

## Usage

```python
from mlx_audio.vad import load

model = load("mlx-community/silero-vad")
probs = model.predict_proba("audio.wav")
timestamps = model.get_speech_timestamps("audio.wav", return_seconds=True)
print(timestamps)
```

For low-latency streaming, feed 512 samples at 16 kHz or 256 samples at 8 kHz:

```python
state = model.initial_state(sample_rate=16000)
probability, state = model.feed(chunk, state, sample_rate=16000)
```
