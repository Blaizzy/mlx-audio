# Smart Turn v3

Smart Turn endpoint detection for conversational audio turns.

## Supported Models

[`mlx-community/smart-turn-v3`](`https://huggingface.co/mlx-community/smart-turn-v3`)

Original model: [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3)

## What It Does

Smart Turn predicts whether the current user turn is complete (`1`) or incomplete (`0`) from audio.

## Input Requirements

- 16kHz mono audio
- Up to 8 seconds of context
- If shorter than 8s, audio is left-padded with zeros
- If longer than 8s, the last 8s are used

## Quick Start

```python
from mlx_audio.vad import load

model = load("mlx-community/smart-turn-v3", strict=True)

result = model.predict_endpoint("audio.wav")
print("prediction:", result.prediction)   # 1 = complete, 0 = incomplete
print("probability:", result.probability)
```

## With an MLX array

```python
import mlx.core as mx
from mlx_audio.vad import load

model = load("mlx-community/smart-turn-v3")
audio = mx.zeros(16000, dtype=np.float32)  # 1 second at 16kHz
result = model.predict_endpoint(audio, sample_rate=16000, threshold=0.5)
```

## Notes

- Typical usage is to run Smart Turn after silence is detected by a lightweight VAD.
- Use full current-turn context instead of very short snippets when possible.

## License

This code is licensed under the BSD 2-Clause "Simplified" License. See `LICENSE` for more information.
