# Phonon-1

MLX support for Fermion Research's English Phonon-1 speech recognition family.
The public Hugging Face transport archive is downloaded, SHA-256 checked, and
materialized automatically on first use.

## Models

| Model | Download | Description |
|---|---:|---|
| `FermionResearch/Phonon-1-Micro` | 285 MB | Smallest build |
| `FermionResearch/Phonon-1` | 415 MB | Default accuracy/size balance |
| `FermionResearch/Phonon-1-Big` | 581 MB | Full-precision audio tower and lowest latency |

Install the STT dependencies, including the archive decoder:

```bash
pip install 'mlx-audio[stt]'
```

```python
from mlx_audio.stt import load

model = load("FermionResearch/Phonon-1")
result = model.generate("audio.wav", language="English")
print(result.text)
```

```bash
mlx_audio.stt.generate \
  --model FermionResearch/Phonon-1 \
  --audio audio.wav \
  --output-path transcript \
  --language English
```

Phonon-1 is English-only and emits segment timestamps at chunk boundaries. It
does not provide word timestamps or forced alignment.
