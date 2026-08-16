# Music Generation

MLX-Audio music models generate complete songs from a musical caption and
structured lyrics.

| Model | Output | Models | Key features |
|---|---|---|---|
| [MiniMax Music 3](minimax-music3.md) | 44.1 kHz stereo | [BF16](https://huggingface.co/mlx-community/MiniMax-Music3-bf16), [MXFP8](https://huggingface.co/mlx-community/MiniMax-Music3-mxfp8), [MXFP4](https://huggingface.co/mlx-community/MiniMax-Music3-mxfp4) | Multilingual lyrics, structural tags, long-form generation |

```bash
python -m mlx_audio.music.generate \
  --model mlx-community/MiniMax-Music3-mxfp8 \
  --caption "Warm acoustic pop with intimate female vocals" \
  --lyrics $'[verse]\nMorning light\n[chorus]\nSing with me' \
  --output song.wav
```
