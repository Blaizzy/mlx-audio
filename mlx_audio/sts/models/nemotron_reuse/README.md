# NVIDIA NemotronLabs RE-USE

Enhance noisy speech recordings on Apple Silicon with
[NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE).
The model supports audio from 8 to 48 kHz, including mono and multichannel files.

Follow the [MLX Audio installation instructions](../../../../README.md#installation)
before running the examples below.

## Python

Load the model and save an enhanced recording:

```python
from mlx_audio import audio_io
from mlx_audio.sts import load

model = load("nvidia/NVIDIA-NemotronLabs-RE-USE", strict=True)
result = model.generate("noisy.wav")
audio_io.write("enhanced.wav", result.audio, result.sample_rate)
```

The output keeps the input sample rate and number of channels. For a NumPy or
MLX array, provide its sample rate:

```python
audio, sample_rate = audio_io.read("noisy.wav")
result = model.generate(audio, sample_rate=sample_rate)
```

Arrays use `[samples]` for mono or `[samples, channels]` for multichannel audio.
Each channel is enhanced independently.

## Command line

```bash
python -m mlx_audio.sts.generate \
  --model nvidia/NVIDIA-NemotronLabs-RE-USE \
  --audio noisy.wav --output-path enhanced.wav
```

## Options and long recordings

| Python option | Default | Purpose |
| --- | --- | --- |
| `sample_rate` | Read from the file | Required for array input; must be between 8000 and 48000 Hz. |
| `target_sample_rate` | Keep the input rate | Resample to this rate before enhancement; must be between 8000 and 48000 Hz. |
| `chunk_seconds` | `5.0` | Process long recordings in chunks. Smaller chunks reduce memory use; `None` processes the whole recording at once. |
| `overlap_seconds` | Half the chunk duration | Overlap between chunks, in seconds. Must be nonnegative and less than the chunk duration. |

Changing the chunk duration can change the enhanced audio. Processing an entire
long recording at once uses more memory.

To extend lower-rate speech to 48 kHz, set `target_sample_rate`:

```python
result = model.generate("noisy.wav", target_sample_rate=48000)
audio_io.write("enhanced_48k.wav", result.audio, result.sample_rate)
```

Pass these options through `--gen-kwargs` on the command line:

```bash
python -m mlx_audio.sts.generate \
  --model nvidia/NVIDIA-NemotronLabs-RE-USE --audio noisy.wav \
  --output-path enhanced_48k.wav \
  --gen-kwargs '{"target_sample_rate": 48000, "chunk_seconds": 5, "overlap_seconds": 2.5}'
```

The model processes recorded audio offline and does not support live streaming.
When speakers overlap, it may suppress secondary voices.

## Save a local model

The original checkpoint loads directly, so conversion is optional. To save a
local MLX checkpoint:

```bash
python -m mlx_audio.convert \
  --hf-path nvidia/NVIDIA-NemotronLabs-RE-USE \
  --mlx-path ./RE-USE-mlx --dtype float32
```

Then use the local directory in place of the model ID:

```python
model = load("./RE-USE-mlx", strict=True)
```

Keep the weights in float32. For full float32 computation on M5, prefix inference
commands with `MLX_ENABLE_TF32=0`.

## License

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. See NVIDIA's [upstream license and third-party notices](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/blob/main/LICENSE).
