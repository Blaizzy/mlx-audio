# MiMo-V2.5-ASR

MLX support for Xiaomi's `MiMo-V2.5-ASR`.

## Available Model

- [MiMo-V2.5-ASR](https://huggingface.co/mlx-community/MiMo-V2.5-ASR-MLX)

## Python Usage

```python
from mlx_audio.stt import load

model = load("mlx-community/MiMo-V2.5-ASR-MLX")
result = model.generate("audio.wav", language="en")
print(result.text)
```

## Local Usage

```python
from mlx_audio.stt import load

model = load("/path/to/MiMo-V2.5-ASR-MLX")
result = model.generate("audio.wav")
print(result.text)
```
