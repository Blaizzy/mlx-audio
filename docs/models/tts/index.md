# Text-to-Speech Models

MLX-Audio supports a wide range of TTS models optimized for Apple Silicon. Each model offers different tradeoffs between speed, quality, languages, and features.

## Model Comparison

| Model | Size | Languages | Voice Cloning | Streaming | Key Features |
|-------|------|-----------|:---:|:---:|--------------|
| [**Kokoro**](kokoro.md) | 82M | EN, JA, ZH, FR, ES, IT, PT, HI | -- | -- | Fast, 54 voice presets, speed control |
| [**Qwen3-TTS**](qwen3-tts.md) | 0.6B / 1.7B | ZH, EN, JA, KO, + more | Yes | Yes | Voice cloning, emotion control, voice design, batch generation |
| [**Voxtral TTS**](voxtral-tts.md) | 4B | EN, FR, ES, DE, IT, PT, NL, AR, HI | -- | Yes | 20 voice presets, 9 languages, chunked streaming output |
| [**CSM**](csm.md) | 1B | EN | Yes | Yes | Conversational speech, voice cloning, multi-turn context |
| [**Dia**](dia.md) | 1.6B | EN | -- | -- | Dialogue with `[S1]`/`[S2]` speaker tags |
| [**Chatterbox**](chatterbox.md) | -- | EN + 15 languages | Yes | -- | Expressive, emotion exaggeration control |
| [KugelAudio](kugelaudio.md) | 7B | 24 European languages | -- | -- | VibeVoice-based multilingual TTS with diffusion decoding |
| [Spark](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/spark) | 0.5B | EN, ZH | -- | -- | SparkTTS model |
| [OuteTTS](https://huggingface.co/mlx-community/OuteTTS-1.0-0.6B-fp16) | 0.6B | EN | -- | -- | Efficient TTS |
| [Soprano](https://huggingface.co/mlx-community/Soprano-1.1-80M-bf16) | 80M | EN | -- | -- | High-quality TTS |
| [Ming Omni TTS](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/bailingmm/README.md) | 16.8B (A3B) / 0.5B | EN, ZH | Yes | -- | Voice cloning, style/emotion control, music & sound FX generation |
| [TADA](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/tada/README.md) | 1B / 3B | EN (1B), EN + 9 langs (3B) | Yes | -- | HumeAI, speed control, flow matching |
| [Echo TTS](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/echo_tts/README.md) | -- | EN | Yes | -- | Diffusion-based, fast voice cloning |
| [Irodori TTS](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/irodori_tts/README.md) | 500M | JA | Yes | -- | Japanese-only, DiT + DACVAE |
| [Fish Speech](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/fish_qwen3_omni/README.md) | -- | EN | Yes | -- | Inline control tags, multi-speaker, long-form batching |

## Quick Start

All TTS models share a common interface:

=== "CLI"

    ```bash
    mlx_audio.tts.generate \
        --model <model-id> \
        --text "Hello, world!" \
        --voice <voice-name>
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("<model-id>")

    for result in model.generate(text="Hello, world!"):
        audio = result.audio  # mx.array waveform
    ```

!!! tip "Choosing a model"
    - **Fastest / smallest:** Kokoro (82M) -- great for quick generation with many voice presets.
    - **Voice cloning:** CSM or Qwen3-TTS -- clone any voice from a short reference clip.
    - **Multilingual:** Voxtral TTS (9 languages, 20 voices) or Chatterbox (16 languages).
    - **Dialogue:** Dia -- built-in support for multi-speaker conversations.
    - **Emotion / style control:** Qwen3-TTS CustomVoice or VoiceDesign variants.
