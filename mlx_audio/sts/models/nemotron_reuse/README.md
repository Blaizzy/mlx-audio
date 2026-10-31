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

## Experimental stable analysis

`frontend="stable"` uses NumPy CPU float64 spectral analysis, an exactly
symmetric periodic Hann window, and a defined phase convention for real FFT
bins. The neural network still runs in MLX with the original weights. This is
an experimental alternative to the default `frontend="native"`:

```python
result = model.generate("noisy.wav", frontend="stable")
```

```bash
MLX_ENABLE_TF32=0 python -m mlx_audio.sts.generate \
  --model nvidia/NVIDIA-NemotronLabs-RE-USE --audio noisy.wav \
  --gen-kwargs '{"frontend": "stable"}'
```

The stable mode changes the model's input convention and therefore its output.
It is intended for comparison with a reference using the same analysis; it
does not reproduce NVIDIA's unmodified CUDA preprocessing. It sets imaginary
parts to zero only at DC, Nyquist, and frames centered exactly on reflected
boundaries, with zero phase for exactly zero coefficients. Other bins are not
thresholded. It also uses the shared window for
synthesis. Keep `MLX_ENABLE_TF32=0` when checking numerical agreement.

### Stable frontend validation

The experimental mode was compared on Apple M5 Max and NVIDIA A30 using the
same float32 audio samples, original weights, and shared NumPy analysis.
Both backends used full float32 network computation. The CUDA reference used
PyTorch 2.8.0/CUDA 12.8 and native Mamba 2.3.1 kernels.

| Input | Native frontend error | Shared stable frontend error |
| --- | ---: | ---: |
| Coffee shop, 48 kHz | 1.392% | 0.000408% |
| Noise, 16 kHz | 1.265% | 0.000388% |
| Speech, 8 kHz, 0.5 s chunks | 0.406% | 0.000476% |
| Held-out coffee excerpt, 48 kHz | 2.614% | 0.000444% |

Error is `norm(MLX - CUDA) / norm(CUDA)`, before audio encoding. Each column
compares its own frontend on both backends: the stable column does **not**
measure agreement with NVIDIA's original preprocessing. All 47 cases passed
the preset 0.1% error target. The worst non-silent error was 0.000726%; silence
remained at 0.0263% (maximum absolute sample error `1.19e-5`).

The shared analysis is not guaranteed bit-identical across CPU platforms.
On macOS/NumPy 2.5.4 and Linux/NumPy 2.2.6, 36 of 47 cases had identical
frontend hashes. The remaining cases differed by at most `1.46e-11` in
compressed magnitude and `3.58e-7` radians in phase, with identical windows
and no phase-wrap mismatches.

Quality was screened separately on 36 six-second inputs from 12
[LibriSpeech test-clean](https://www.openslr.org/12) speakers: clean controls,
additive [coffee-shop noise](https://freesound.org/people/ericdhunter/sounds/236336/),
and white noise at -5, 0, or +5 dB input SNR. Sample rates spanned 8–48 kHz;
the original speech was 16 kHz, so upsampling adds no high-frequency speech
content. Stable-minus-native changes against the same clean references were:

| Backend | Median SI-SDR change | Worst SI-SDR change | Median STOI change | Worst STOI change |
| --- | ---: | ---: | ---: | ---: |
| MLX | +0.0104 dB | -0.1014 dB | +0.00000011 | -0.0000204 |
| CUDA | +0.0090 dB | -0.1113 dB | -0.00000010 | -0.0001427 |

Both passed preset engineering screens: median/worst SI-SDR loss below
0.1/1.0 dB and median/worst STOI loss below 0.005/0.02. This small synthetic
mixture set is not a listening study or proof of perceptual equivalence.
The native frontend remains the default.

## Numerical validation (native frontend)

All 1,416 converted tensors preserve the source float32 values exactly after
layout conversion. The unchanged NVIDIA network was validated on an NVIDIA L4
with PyTorch 2.8.0/CUDA 12.8, Mamba 2.3.1, and native CUDA Mamba/causal-convolution
kernels. The MLX comparison used Apple M5 Max, MLX 0.32.4, and
`MLX_ENABLE_TF32=0`; CUDA TF32 was also disabled. The source checkpoint revision
was `31f82f2efad1095034492f236e8dac9239ba713b`.

With identical spectral inputs, relative complex-output error was `4.0e-6` to
`6.8e-6` on eight non-silent clips, and `3.0e-4` on silence. End-to-end waveform
agreement was input-dependent:

| Input | Duration | Relative waveform error |
| --- | ---: | ---: |
| Speech, 8 kHz | 5.1 s | 0.00047% |
| Speech, 44.1 kHz | 1 s | 0.342% |
| Speech, 22.05 kHz | 0.6 s | 0.469% |
| Coffee shop, 48 kHz | 1 s | 1.39% |
| Noise, 16 kHz | 0.5 s | 1.27% |
| Silence, 8 kHz | 0.25 s | 0.0263% |
| Speech, 8 kHz, 0.5 s chunks | 1.2 s | 0.406% |
| Held-out coffee channel/excerpt, 48 kHz | 1 s | 2.61% |
| Held-out speech, 16 kHz | 1 s | 0.224% |
| Held-out noise, 32 kHz | 0.5 s | 1.44% |
| Speech, 8 kHz, default 5 s chunks | 6.3 s | 0.0287% |

Relative error is `norm(MLX - CUDA) / norm(CUDA)` on the same float32 input
samples, before audio encoding. These are numerical differences, not
measurements of perceptual quality. Optional bandwidth-extension resampling
was excluded from this comparison.

Small FFT rounding differences can select opposite sides of the `-pi`/`+pi`
phase boundary. The network consumes raw phase angles, so this affects
predictions, particularly in the first reflected frame and at real FFT
endpoints. Keeping NVIDIA's CUDA network fixed and changing only the STFT to
PyTorch CPU changed its waveform by 1.86% on the coffee clip and 4.70% on noise.
Full float32 therefore does not guarantee waveform parity across FFT backends.

## License

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. The adapted architecture and checkpoint are covered by [OpenMDW-1.1](LICENSE). See the included license and third-party notices.
