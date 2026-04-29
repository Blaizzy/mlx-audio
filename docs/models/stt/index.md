---
title: Speech-to-Text Models
---

# Speech-to-Text (STT) Models

MLX Audio provides a range of speech-to-text models optimized for Apple Silicon, from lightweight English-only models to large multilingual systems with translation capabilities.

## Model Comparison

| Model | Provider | Parameters | Languages | Streaming | Timestamps | Repo |
|-------|----------|-----------|-----------|-----------|------------|------|
| [**Whisper**](whisper.md) | OpenAI | Various | 99+ | -- | Segment + Word | [mlx-community/whisper-large-v3-turbo-asr-fp16](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-fp16) |
| **Distil-Whisper** | HuggingFace | Various | EN | -- | Segment | [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) |
| [**Parakeet**](parakeet.md) | NVIDIA | 0.6B | EN (v2), 25 EU (v3) | Yes | Sentence + Word | [mlx-community/parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) |
| [**Voxtral Realtime**](voxtral-realtime.md) | Mistral | 4B | Multiple | Yes | -- | [4bit](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit), [fp16](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16) |
| **Qwen3-ASR** | Alibaba | 0.6B / 1.7B | ZH, EN, JA, KO + more | Yes | Segment | [mlx-community/Qwen3-ASR-1.7B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit) |
| **Qwen3-ForcedAligner** | Alibaba | 0.6B | ZH, EN, JA, KO + more | -- | Word-level | [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) |
| **VibeVoice-ASR** | Microsoft | 9B | Multiple | Yes | Segment | [mlx-community/VibeVoice-ASR-bf16](https://huggingface.co/mlx-community/VibeVoice-ASR-bf16) |
| **Voxtral** | Mistral | 3B | Multiple | -- | -- | [mlx-community/Voxtral-Mini-3B-2507-bf16](https://huggingface.co/mlx-community/Voxtral-Mini-3B-2507-bf16) |
| **Cohere Transcribe** | Cohere | 2B | 14 languages | -- | Segment | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) |
| [**Qwen2-Audio**](qwen2-audio.md) | Alibaba | 7B | Multiple | -- | -- | [mlx-community/Qwen2-Audio-7B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2-Audio-7B-Instruct-4bit) |
| **Canary** | NVIDIA | ~1B | 25 EU + RU, UK | -- | -- | [README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/canary/README.md) |
| **SenseVoice** | Alibaba DAMO | ~234M | 50+ | -- | -- | [mlx-community/SenseVoiceSmall](https://huggingface.co/mlx-community/SenseVoiceSmall) |
| **FireRedASR2** | Xiaohongshu | ~1.18B | ZH, EN | -- | -- | [mlx-community/FireRedASR2-AED-mlx](https://huggingface.co/mlx-community/FireRedASR2-AED-mlx) |
| **Granite Speech** | IBM | ~1B | EN, FR, DE, ES, PT, JA | Yes | -- | [README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/granite_speech/README.md) |
| **Moonshine** | Useful Sensors | 27M / 61M | EN | -- | -- | [README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/moonshine/README.md) |
| **MMS** | Meta | 1B | 1000+ | -- | -- | [README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/README.md) |

!!! note "Cohere quantized local checkpoints"
    Cohere Transcribe can be loaded from local MLX-converted checkpoints, including 8-bit and 4-bit variants. To generate these variants explicitly, use `--q-bits 8` or `--q-bits 4` together with `--quantize`.

    - 8-bit: `python -m mlx_audio.convert --model-domain stt --quantize --q-bits 8`
    - 4-bit: `python -m mlx_audio.convert --model-domain stt --quantize --q-bits 4`

    You can also pass `--q-group-size` to control the quantization group size when needed.

## Unified API

All STT models share the same loading interface:

=== "Python"

    ```python
    from mlx_audio.stt import load

    model = load("mlx-community/whisper-large-v3-turbo-asr-fp16")
    result = model.generate("audio.wav")
    print(result.text)
    ```

=== "CLI"

    ```bash
    mlx_audio.stt.generate \
      --model mlx-community/whisper-large-v3-turbo-asr-fp16 \
      --audio audio.wav \
      --verbose
    ```

!!! tip "Choosing a model"
    - **Best multilingual coverage**: Whisper (99+ languages) or MMS (1000+ languages)
    - **Best accuracy for English**: Parakeet v2 or Whisper large-v3-turbo
    - **Best for European languages**: Parakeet v3 (25 languages) or Canary
    - **Lowest latency / streaming**: Voxtral Realtime (4bit variant)
    - **Smallest footprint**: Moonshine tiny (27M parameters)
    - **Speaker diarization built-in**: VibeVoice-ASR
    - **Word-level alignment**: Qwen3-ForcedAligner
    - **Emotion / event detection**: SenseVoice
