# Contributions

This file acknowledges the original authors and contributors of models ported to mlx-audio.

## MiniMax Music 3 (Song Generation)

- **Original**: [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
- **Copyright**: MiniMax
- **License**: [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE)
- **MLX Port**: Adapted from [mikolaj92/minimax-music3-mlx](https://github.com/mikolaj92/minimax-music3-mlx) (Apache-2.0)

## MossFormer2 SE (Speech Enhancement)

- **Original**: [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
- **Copyright**: Speech Lab, Alibaba Group
- **License**: Apache License 2.0
- **MLX Port**: Dmitry Starkov ([@starkdmi](https://github.com/starkdmi))

## DeepFilterNet (Speech Enhancement)

- **Original**: [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- **Copyright**: Hendrik Schröter and contributors
- **License**: MIT / Apache-2.0
- **MLX Port**: Kyle Howells ([@kylehowells](https://github.com/kylehowells))

## Nemotron 3.5 ASR Streaming (Speech-to-Text)

- **Original**: [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- **Copyright**: NVIDIA Corporation
- **License**: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- **MLX Port**: [@ARahim3](https://github.com/ARahim3)

## Streaming audio encoder

- **Original**: [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)
- **Copyright**: remsky
- **License**: Apache License 2.0
- **MLX Port**: `mlx_audio/streaming_encoder.py` adapts the incremental
  container-encoding approach from `api/src/services/streaming_audio_writer.py`,
  including its ordering fix for Ogg muxers (Kokoro-FastAPI #497).
