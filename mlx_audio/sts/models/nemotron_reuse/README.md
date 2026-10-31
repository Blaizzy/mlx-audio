# NVIDIA NemotronLabs RE-USE

Native MLX implementation of [nvidia/NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE), a 9.6M-parameter universal speech enhancement model. It uses a convolutional encoder, 30 bidirectional time/frequency Mamba blocks, and magnitude/phase decoders. Runtime inference requires no PyTorch, NeMo, or CUDA.

## Inference

The original float32 checkpoint loads directly:

```python
from mlx_audio import audio_io
from mlx_audio.sts import load

model = load("nvidia/NVIDIA-NemotronLabs-RE-USE", strict=True)
result = model.generate("noisy.wav")
audio_io.write("enhanced.wav", result.audio, result.sample_rate)
```

```bash
python -m mlx_audio.sts.generate \
  --model nvidia/NVIDIA-NemotronLabs-RE-USE \
  --audio noisy.wav --output-path enhanced.wav
```

- Files retain their sample rate. Arrays use `[samples]` or `[samples, channels]` layout and require `sample_rate`. Channels are enhanced independently.
- Supports 8–48 kHz by scaling the STFT sizes, including the source model's integer rounding at 22.05 and 44.1 kHz.
- Uses five-second chunks with 50% overlap and Hann overlap-add by default, following the upstream long-audio recipe. Set `chunk_seconds=None` for one full-context pass. Chunked and full-context results can differ because the model is bidirectional and uses instance normalization.
- Set `target_sample_rate=48000` to resample before enhancement for bandwidth extension. This port uses SciPy polyphase resampling; upstream uses librosa's `kaiser_best`, so that optional resampling step is not numerically identical.
- Preserves the upstream nonpositive-magnitude sweep suppression and `1e-8` tail padding to the original length. Very short inputs are zero-padded before analysis.
- This is offline enhancement, not causal streaming. Overlapping speech may suppress secondary speakers.

Additional CLI options can be passed through `--gen-kwargs`:

```bash
python -m mlx_audio.sts.generate \
  --model nvidia/NVIDIA-NemotronLabs-RE-USE --audio noisy.wav \
  --gen-kwargs '{"target_sample_rate": 48000, "chunk_seconds": 5, "overlap_seconds": 2.5}'
```

## Conversion

```bash
python -m mlx_audio.convert \
  --hf-path nvidia/NVIDIA-NemotronLabs-RE-USE \
  --mlx-path ./RE-USE-mlx --dtype float32
```

Conversion transposes convolution weights to MLX layout and writes a standalone native checkpoint. The loader accepts both original and converted layouts. Float32 is the validated precision; keep the small model unquantized.

For full float32 computation on hardware such as M5, launch inference with
`MLX_ENABLE_TF32=0`. MLX otherwise permits reduced-precision matrix operations
even with float32 weights; see [MLX environment variables](https://ml-explore.github.io/mlx/build/html/usage/environment_variables.html#mlx-enable-tf32).
The port leaves this process-wide backend setting under the caller's control.
This setting does not guarantee identical waveforms across FFT backends.

```bash
MLX_ENABLE_TF32=0 python -m mlx_audio.sts.generate \
  --model ./RE-USE-mlx --audio noisy.wav --output-path enhanced.wav
```

## Numerical validation

All 1,416 converted tensors preserve the source float32 values exactly after
layout conversion. Against the NVIDIA network with the upstream Mamba CPU
reference scan, identical spectral inputs produced relative complex-output
errors of `4.4e-6` to `8.8e-6` on five non-silent clips, and `3.6e-4` on silence,
with `MLX_ENABLE_TF32=0` on Apple M5 Max.

End-to-end waveform agreement is input-dependent. Small FFT rounding differences
can select opposite sides of the `-pi`/`+pi` phase boundary. The network consumes
raw phase angles, so this affects predictions, particularly in the first
reflected frame of a chunk. Relative waveform errors against that CPU reference
were 2.43% on a one-second 48 kHz coffee-shop excerpt, 2.99% on half a second of
16 kHz noise, and 0.37% on 1.2 seconds of speech using half-second chunks.
These are numerical differences, not measurements of perceptual quality.
Parity with NVIDIA's CUDA implementation has not yet been established.

## License

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. The adapted architecture and checkpoint are covered by [OpenMDW-1.1](LICENSE). See the included license and third-party notices.
