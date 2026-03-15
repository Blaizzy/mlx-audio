# Whisper

MLX implementation of [Whisper](https://github.com/openai/whisper) by OpenAI. Also natively supports distilled variants like [Distil-Whisper](https://huggingface.co/distil-whisper/distil-large-v3).

## Usage

```python
from mlx_audio.stt import load

# Standard Whisper
model = load("mlx-community/whisper-large-v3-turbo-asr-fp16")

# Distil-Whisper
# model = load("distil-whisper/distil-large-v3")

result = model.generate("audio.wav")
print(result.text)
```
