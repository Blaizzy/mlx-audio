# Contributions

This file acknowledges the original authors and contributors of models ported to mlx-audio.

## StepAudio2 Cached Streaming Codec

- **Original streaming algorithms**: StepAudio2/CosyVoice2, including encoder code modified from ESPnet; reviewed from the `minicpmo-utils` 1.0.6 source bundle.
- **Copyright notices retained**: 2021 Mobvoi Inc (Binbin Zhang, Di Wu), 2022 Xingchen Song, 2024 Alibaba Inc (Xiang Lyu, Zhihao Du).
- **License**: Apache-2.0 for the derived `codec/models/stepaudio2/streaming.py`; the existing MLX Audio weighted network definitions remain unchanged.
- **MLX adaptation**: Explicit prompt/sequence cache containers, lookahead-aware chunk decoding and functional reset. The API does not introduce a language model, speech-token generator, audio device or application integration.
- **License copy**: `mlx_audio/codec/models/stepaudio2/STREAMING-LICENSE.txt` accompanies the source and packaged codec.

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

## rumik-oss 1 (Text-to-Speech)

- **Original**: [rumik-ai/rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1)
- **Copyright**: rumik.ai
- **License**: [CC-BY-NC 4.0 with Cohere Labs acceptable-use addendum](https://cohere.com/cohere-labs-cc-by-nc-license) (weights); Mimi codec CC-BY-4.0
- **MLX Port**: Suryansh Shakya ([@nullHawk](https://github.com/nullHawk))
