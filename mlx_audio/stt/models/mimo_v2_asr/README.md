# MiMo-V2.5-ASR

MLX support for Xiaomi's `MiMo-V2.5-ASR`.

## Available Model

- [mlx-community/MiMo-V2.5-ASR-MLX](https://huggingface.co/mlx-community/MiMo-V2.5-ASR-MLX)

The model repo resolves its audio tokenizer from `mlx-community/MiMo-Audio-Tokenizer`
via `mlx_manifest.json`, so the default Hugging Face path works without extra
arguments.

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

If you want to override the tokenizer location for a local checkout:

```python
from mlx_audio.stt import load

model = load(
    "/path/to/MiMo-V2.5-ASR-MLX",
    audio_tokenizer_dir="/path/to/MiMo-Audio-Tokenizer",
)
```
