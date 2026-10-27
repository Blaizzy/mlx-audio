---
title: Phonon-2 and Phonon-2.5
---

# Phonon-2 and Phonon-2.5

Speech recognition using Fermion Research's five-value Parakeet TDT models.
[Phonon-2.5](https://huggingface.co/FermionResearch/Phonon-2.5) supports 25 European
languages; [Phonon-2.5 English](https://huggingface.co/FermionResearch/Phonon-2.5-English)
and [Phonon-2](https://huggingface.co/mlx-community/Phonon-2) support English.
All load directly without a Fermion runtime.

```bash
pip install 'mlx-audio[stt]'
```

```python
from mlx_audio.stt import load

model = load("FermionResearch/Phonon-2.5")
result = model.generate("audio.wav", chunk_duration=30, overlap_duration=2)
print(result.text)
```

Use `load("FermionResearch/Phonon-2.5-English")` for English-only recognition.
The first v2.5 load verifies and unpacks the release into a cached MLX checkpoint.
Later loads reuse it. Requires the `stt` extra for archive decompression.

`result.sentences` contains sentences and timestamped token pieces. `stream=True`
emits overlapping chunks from a recording; live-input streaming, translation,
diarization, and hotwords are not supported.

Weights are CC-BY-4.0; adapted code is Apache-2.0.
See the [model README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/phonon2/README.md)
for input requirements and CLI usage.
