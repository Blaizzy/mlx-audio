# DeepFilterNet (MLX)

DeepFilterNet speech enhancement in pure MLX with support for model versions 1, 2, and 3.

Pretrained weights: [mlx-community/DeepFilterNet-mlx](https://huggingface.co/mlx-community/DeepFilterNet-mlx)

## Quick Start

```python
from mlx_audio.sts import load

# Load v3 (default) through the standard STS loader
model = load("mlx-community/DeepFilterNet-mlx")
model.enhance_file("noisy.wav", "clean.wav")
```

Or use the model-specific initializer directly when you need explicit version
selection:

```python
from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

# Load v3 (default)
model = DeepFilterNetModel.from_pretrained()
model.enhance_file("noisy.wav", "clean.wav")

# Load a specific version
model = DeepFilterNetModel.from_pretrained(version=2)

# Or specify the subfolder directly
model = DeepFilterNetModel.from_pretrained(subfolder="v1")
```

Streaming/chunked mode (true per-hop stateful processing for v2/v3):

```python
streamer = model.create_streamer(pad_end_frames=3, compensate_delay=True)
out_1 = streamer.process_chunk(chunk_a)
out_2 = streamer.process_chunk(chunk_b)
out_tail = streamer.flush()
```

## Model Selection

Model architecture is selected automatically from `config.json` (`model_version` field).
