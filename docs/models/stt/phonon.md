---
title: Phonon-1
---

# Phonon-1

Phonon-1 is Fermion Research's English speech-recognition family based on the
Qwen3-ASR 0.6B architecture. Its decoder is trained with a five-value weight
representation and runs through native MLX quantized matrix multiplication.

## Available models

| Model | Download | LibriSpeech clean / other | Notes |
|---|---:|---:|---|
| [`FermionResearch/Phonon-1-Micro`](https://huggingface.co/FermionResearch/Phonon-1-Micro) | 285 MB | 3.002 / 6.511 | Smallest build |
| [`FermionResearch/Phonon-1`](https://huggingface.co/FermionResearch/Phonon-1) | 415 MB | 2.640 / 5.699 | Default accuracy/size balance |
| [`FermionResearch/Phonon-1-Big`](https://huggingface.co/FermionResearch/Phonon-1-Big) | 581 MB | 2.667 / 5.722 | Full-precision audio tower and lowest latency |

Word-error rates are the provider's published full-test-set measurements.

## Installation

Install the STT extra so the first-run archive materializer is available:

```bash
pip install 'mlx-audio[stt]'
```

The first load downloads the selected Hugging Face archive, verifies its
published SHA-256, reconstructs the original safetensors, and verifies every
materialized file against the archive manifest. Later loads reuse the local
cache. Set `MLX_AUDIO_CACHE_DIR` to relocate that cache.

## Usage

=== "Python"

    ```python
    from mlx_audio.stt import load

    model = load("FermionResearch/Phonon-1")
    result = model.generate(
        "audio.wav",
        language="English",
        temperature=0.0,
        repetition_penalty=1.05,
        repetition_context_size=96,
    )
    print(result.text)
    ```

=== "CLI"

    ```bash
    mlx_audio.stt.generate \
      --model FermionResearch/Phonon-1 \
      --audio audio.wav \
      --output-path transcript \
      --language English \
      --gen-kwargs '{"repetition_penalty": 1.05, "repetition_context_size": 96}'
    ```

Use `stream=True` in Python or `--stream` in the CLI for token streaming.

## Boundaries

The released family is English-only. It returns chunk-level segments, not word
timestamps, forced alignment, speaker diarization, or translation.
