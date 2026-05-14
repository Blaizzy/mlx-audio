# Models

MLX Audio supports a wide range of audio models across four categories, all optimized for Apple Silicon.

Many hosted MLX checkpoints referenced in these docs live under
[mlx-community](https://huggingface.co/mlx-community) on Hugging Face, the shared org
for ready-to-use MLX model weights across projects like `mlx-lm`, `mlx-vlm`, and
`mlx-audio`. If you are adding a new model, prefer publishing it there when possible
so users can find MLX models in one consistent place.

## Text-to-Speech (TTS)

Generate natural-sounding speech from text. Multiple models with multilingual support, voice cloning, and style control.

| Model | Description | Languages | Repo |
|-------|-------------|-----------|------|
| **Kokoro** | Fast, high-quality multilingual TTS | EN, JA, ZH, FR, ES, IT, PT, HI | [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) |
| **Qwen3-TTS** | Alibaba's multilingual TTS with voice design | ZH, EN, JA, KO, + more | [mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16) |
| **Voxtral TTS** | Mistral's 4B multilingual TTS (20 voices, 9 languages) | EN, FR, ES, DE, IT, PT, NL, AR, HI | [mlx-community/Voxtral-4B-TTS-2603-mlx-bf16](https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-bf16) |
| **CSM** | Conversational Speech Model with voice cloning | EN | [mlx-community/csm-1b](https://huggingface.co/mlx-community/csm-1b) |
| **Dia** | Dialogue-focused TTS | EN | [mlx-community/Dia-1.6B-fp16](https://huggingface.co/mlx-community/Dia-1.6B-fp16) |
| **Chatterbox** | Expressive multilingual TTS | EN, ES, FR, DE, IT, PT, + more | [mlx-community/chatterbox-fp16](https://huggingface.co/mlx-community/chatterbox-fp16) |
| **KugelAudio** | 7B multilingual TTS for 24 European languages | 24 European languages | [kugelaudio/kugelaudio-0-open](https://huggingface.co/kugelaudio/kugelaudio-0-open) |
| **Soprano** | High-quality TTS | EN | [mlx-community/Soprano-1.1-80M-bf16](https://huggingface.co/mlx-community/Soprano-1.1-80M-bf16) |
| **OuteTTS** | Efficient TTS model | EN | [mlx-community/OuteTTS-1.0-0.6B-fp16](https://huggingface.co/mlx-community/OuteTTS-1.0-0.6B-fp16) |
| **Spark** | SparkTTS model | EN, ZH | [mlx-community/Spark-TTS-0.5B-bf16](https://huggingface.co/mlx-community/Spark-TTS-0.5B-bf16) |
| **Ming Omni TTS (BailingMM)** | Multimodal generation with voice cloning and style control | EN, ZH | [mlx-community/Ming-omni-tts-16.8B-A3B-bf16](https://huggingface.co/mlx-community/Ming-omni-tts-16.8B-A3B-bf16) |
| **Ming Omni TTS (Dense)** | Lightweight dense Ming Omni variant | EN, ZH | [mlx-community/Ming-omni-tts-0.5B-bf16](https://huggingface.co/mlx-community/Ming-omni-tts-0.5B-bf16) |

[:octicons-arrow-right-24: Browse TTS Models](tts/index.md)

---

## Speech-to-Text (STT)

Transcribe and understand speech with state-of-the-art accuracy. Streaming support, word-level timestamps, and speaker diarization.

| Model | Description | Languages | Repo |
|-------|-------------|-----------|------|
| **Whisper** | OpenAI's robust STT model | 99+ languages | [mlx-community/whisper-large-v3-turbo-asr-fp16](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-fp16) |
| **Distil-Whisper** | Distilled fast Whisper variants | EN | [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) |
| **Qwen3-ASR** | Alibaba's multilingual ASR | ZH, EN, JA, KO, + more | [mlx-community/Qwen3-ASR-1.7B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit) |
| **MiMo-V2.5-ASR** | Xiaomi's speech recognition model | ZH, EN | [MiMo-V2.5-ASR](https://huggingface.co/mlx-community/MiMo-V2.5-ASR-MLX) |
| **Qwen3-ForcedAligner** | Word-level audio alignment | ZH, EN, JA, KO, + more | [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit) |
| **Parakeet** | NVIDIA's accurate STT | EN (v2), 25 EU languages (v3) | [mlx-community/parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) |
| **Voxtral** | Mistral's speech model | Multiple | [mlx-community/Voxtral-Mini-3B-2507-bf16](https://huggingface.co/mlx-community/Voxtral-Mini-3B-2507-bf16) |
| **Voxtral Realtime** | Mistral's 4B streaming STT | Multiple | [4bit](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit) / [fp16](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16) |
| **VibeVoice-ASR** | Microsoft's 9B ASR with diarization | Multiple | [mlx-community/VibeVoice-ASR-bf16](https://huggingface.co/mlx-community/VibeVoice-ASR-bf16) |
| **Qwen2-Audio** | Audio-language model for transcription, translation, and audio understanding | Multiple | [mlx-community/Qwen2-Audio-7B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2-Audio-7B-Instruct-4bit) |
| **Canary** | NVIDIA's multilingual ASR with translation | 25 EU + RU, UK | -- |
| **Moonshine** | Useful Sensors' lightweight ASR | EN | -- |
| **MMS** | Meta's massively multilingual ASR | 1000+ languages | -- |
| **Granite Speech** | IBM's ASR + speech translation | EN, FR, DE, ES, PT, JA | -- |

[:octicons-arrow-right-24: Browse STT Models](stt/index.md)

---

## Voice Activity Detection / Speaker Diarization (VAD)

Detect speech segments and identify speakers in audio.

| Model | Description | Repo |
|-------|-------------|------|
| **Sortformer v1** | NVIDIA's end-to-end speaker diarization (up to 4 speakers) | [mlx-community/diar_sortformer_4spk-v1-fp32](https://huggingface.co/mlx-community/diar_sortformer_4spk-v1-fp32) |
| **Sortformer v2.1** | Streaming speaker diarization with AOSC compression | [mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32](https://huggingface.co/mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32) |
| **Smart Turn** | Endpoint detection for conversational turn-taking | [mlx-community/smart-turn-v3](https://huggingface.co/mlx-community/smart-turn-v3) |

[:octicons-arrow-right-24: VAD Models](vad/index.md)

---

## Speech-to-Speech (STS)

Transform, separate, and enhance audio.

| Model | Description | Use Case | Repo |
|-------|-------------|----------|------|
| **SAM-Audio** | Text-guided source separation | Extract specific sounds | [mlx-community/sam-audio-large](https://huggingface.co/mlx-community/sam-audio-large) |
| **Liquid2.5-Audio** | Speech-to-Speech, TTS, and STT | Speech interactions | [mlx-community/LFM2.5-Audio-1.5B-8bit](https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-8bit) |
| **MossFormer2 SE** | Speech enhancement | Noise removal | [starkdmi/MossFormer2_SE_48K_MLX](https://huggingface.co/starkdmi/MossFormer2_SE_48K_MLX) |
| **DeepFilterNet (1/2/3)** | Speech enhancement | Noise suppression | [mlx-community/DeepFilterNet-mlx](https://huggingface.co/mlx-community/DeepFilterNet-mlx) |

[:octicons-arrow-right-24: STS Models](sts/index.md)
