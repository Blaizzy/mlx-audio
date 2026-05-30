# FSMN-VAD

MLX port of the [FunASR FSMN-VAD](https://github.com/modelscope/FunASR) voice activity detector.

FSMN-VAD is a lightweight, low-latency VAD model based on Feedforward Sequential Memory Networks (FSMN), widely used in Chinese speech processing pipelines.

## Supported Model

[mlx-community/fsmn-vad](https://huggingface.co/mlx-community/fsmn-vad)

## Usage

### Load via framework (recommended)

```python
from mlx_audio.vad import load

model = load("mlx-community/fsmn-vad")
```

### Load from model class directly

```python
from mlx_audio.vad.models.fsmn.model import Model

# From HuggingFace
model = Model.from_pretrained("mlx-community/fsmn-vad")

# Or from a local directory
model = Model.from_pretrained("/path/to/fsmn-vad-mlx")
```

### Detect speech segments

From a WAV file:

```python
segments = model.detect("audio.wav")
print(segments)  # [[start_ms, end_ms], ...]
```

From a numpy waveform:

```python
from mlx_audio.audio_io import read

waveform, sr = read("audio.wav")
segments = model.detect(waveform, sample_rate=sr)
print(segments)  # [[270, 3790], [4460, 6900], ...]
```

## Requirements

The FSMN-VAD frontend uses `mlx_audio.dsp.compute_fbank_kaldi`, `mlx_audio.audio_io.read`, and `mlx_audio.utils.resample_audio`, so no extra torch/torchaudio/soundfile dependencies are required.

## Architecture

- **Frontend**: Kaldi fbank (80-dim) → LFR (5×1) → CMVN → 400-dim input
- **Encoder**: 4-layer FSMN with causal depthwise convolution
- **Output**: 248-class softmax (speech/silence/noise posteriors)
- **Post-processing**: Configurable VAD state machine with silence/speech thresholds

## Model Conversion

To convert your own FSMN-VAD weights from FunASR (PyTorch) to MLX:

```python
from mlx_audio.vad.models.fsmn.convert import convert

convert("path/to/funasr/fsmn-vad", output_dir="fsmn-vad-mlx")
```

## Reference

- [FunASR](https://github.com/modelscope/FunASR) — Alibaba DAMO Academy
- [FSMN paper](https://arxiv.org/abs/1803.05030) — Feedforward Sequential Memory Networks
