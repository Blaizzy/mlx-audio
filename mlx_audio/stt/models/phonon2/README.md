# Phonon-2 and Phonon-2.5

Speech recognition with Fermion Research's five-value derivatives of NVIDIA
Parakeet TDT. All three checkpoints load through the standard MLX Audio API;
no Fermion runtime is needed.

| Model | Languages |
| --- | --- |
| [FermionResearch/Phonon-2.5](https://huggingface.co/FermionResearch/Phonon-2.5) | 25 European languages |
| [FermionResearch/Phonon-2.5-English](https://huggingface.co/FermionResearch/Phonon-2.5-English) | English |
| [mlx-community/Phonon-2](https://huggingface.co/mlx-community/Phonon-2) | English |

Install with `pip install 'mlx-audio[stt]'`. The first load of a v2.5 release
downloads its compressed archive, verifies its checksums, and caches an MLX
checkpoint in `~/.cache/mlx_audio/phonon2`. Set `MLX_AUDIO_CACHE_DIR` to change
the cache root. Later loads reuse the cached checkpoint.

```python
from mlx_audio.stt import load

model = load("FermionResearch/Phonon-2.5")
result = model.generate("audio.wav", chunk_duration=30, overlap_duration=2)
print(result.text)
```

Use `load("FermionResearch/Phonon-2.5-English")` for the English-only variant.
The multilingual model recognizes the spoken language automatically; no language
prompt or separate tokenizer download is needed.

`result.sentences` contains timestamped token pieces, which may need joining
into words. Inputs can be file paths or mono 16 kHz MLX arrays. File inputs
are resampled automatically; short nonempty inputs are padded to 320 samples.
Empty inputs are rejected.

## CLI and streaming

```bash
mlx_audio.stt.generate \
  --model FermionResearch/Phonon-2.5 --audio audio.wav --output-path transcript
```

`stream=True` or CLI `--stream` emits overlapping chunks from a recording.
Live-input streaming, translation, diarization, and hotword biasing are not supported.

Generic conversion and requantization are unsupported because they can alter
packed codes and row palettes. Load the published checkpoints directly.

## Attribution

Packed layer integration and container loading are adapted from fermion-research
0.2.11 and 0.2.13 (Copyright 2026 Fermion Research), under the
Apache License 2.0. The reader maps container records to MLX Audio's
parameter names and preserves the packed encoder planes. Weights derive from
NVIDIA Parakeet TDT 0.6B v3 and are CC-BY-4.0.
