---
hide:
  - navigation
  - toc
---

# MLX Audio

**Fast, efficient audio processing on Apple Silicon.**

MLX Audio is the premier audio library built on Apple's MLX framework, delivering high-performance text-to-speech (TTS), speech-to-text (STT), and speech-to-speech (STS) on M-series chips.

---

<div class="grid cards" markdown>

-   :material-microphone:{ .lg .middle } **Text-to-Speech**

    ---

    Generate natural speech with models like Kokoro, Qwen3-TTS, Voxtral TTS, CSM, Dia, and more. Multilingual support, voice cloning, and speed control.

    [:octicons-arrow-right-24: TTS Models](models/tts/index.md)

-   :material-ear-hearing:{ .lg .middle } **Speech-to-Text**

    ---

    Transcribe audio with Whisper, Parakeet, Voxtral Realtime, Qwen3-ASR, VibeVoice, and more. Streaming support and word-level timestamps.

    [:octicons-arrow-right-24: STT Models](models/stt/index.md)

-   :material-swap-horizontal:{ .lg .middle } **Speech-to-Speech**

    ---

    Source separation with SAM-Audio, speech enhancement with MossFormer2 and DeepFilterNet, and conversational AI with Liquid2.5-Audio.

    [:octicons-arrow-right-24: STS Models](models/sts/index.md)

-   :material-lightning-bolt:{ .lg .middle } **Optimized for Apple Silicon**

    ---

    Built on MLX for native M1/M2/M3/M4 acceleration. Quantization support (3-bit to 8-bit) for smaller models and faster inference.

    [:octicons-arrow-right-24: Quantization Guide](guides/quantization.md)

-   :material-api:{ .lg .middle } **OpenAI-Compatible API**

    ---

    Drop-in REST API server with a modern web UI featuring 3D audio visualization. Compatible with existing OpenAI client libraries.

    [:octicons-arrow-right-24: Web UI & API Guide](guides/web-ui-api-server.md)

-   :material-apple:{ .lg .middle } **Swift / iOS Support**

    ---

    On-device TTS for macOS and iOS via the companion [mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift) package.

    [:octicons-arrow-right-24: Swift Package](https://github.com/Blaizzy/mlx-audio-swift)

</div>

---

## Get Started

<div class="grid cards" markdown>

-   [:octicons-download-24: **Installation**](getting-started/installation.md) -- pip, uv, and development setup
-   [:octicons-terminal-24: **CLI Quick Start**](getting-started/quickstart-cli.md) -- Generate and transcribe from the command line
-   [:octicons-code-24: **Python Quick Start**](getting-started/quickstart-python.md) -- Use mlx-audio in your Python projects
-   [:octicons-book-24: **All Models**](models/index.md) -- Browse every supported model

</div>
