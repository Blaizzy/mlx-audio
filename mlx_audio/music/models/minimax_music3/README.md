# MiniMax Music 3

Native MLX inference for
[`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
including the hierarchical autoregressive generator, RVQ depth decoder,
flow-matching transformer, condition encoder, and stereo 44.1 kHz vocoder.

## Models

| Model | Use |
|---|---|
| [`mlx-community/MiniMax-Music3-bf16`](https://huggingface.co/mlx-community/MiniMax-Music3-bf16) | Dense reference conversion |
| [`mlx-community/MiniMax-Music3-8bit`](https://huggingface.co/mlx-community/MiniMax-Music3-8bit) | Affine 8-bit conversion |
| [`mlx-community/MiniMax-Music3-6bit`](https://huggingface.co/mlx-community/MiniMax-Music3-6bit) | Affine 6-bit conversion |
| [`mlx-community/MiniMax-Music3-4bit`](https://huggingface.co/mlx-community/MiniMax-Music3-4bit) | Affine 4-bit conversion |
| [`mlx-community/MiniMax-Music3-mxfp8`](https://huggingface.co/mlx-community/MiniMax-Music3-mxfp8) | Recommended balance of memory and lyric fidelity |
| [`mlx-community/MiniMax-Music3-mxfp4`](https://huggingface.co/mlx-community/MiniMax-Music3-mxfp4) | Experimental lower-memory conversion |
| [`mlx-community/MiniMax-Music3-nvfp4`](https://huggingface.co/mlx-community/MiniMax-Music3-nvfp4) | Experimental NVFP4 conversion |

MXFP4 may alter or omit more requested words than BF16 or MXFP8. Prefer MXFP8
when lyric fidelity matters.

## Generate

```bash
python -m mlx_audio.music.generate \
  --model mlx-community/MiniMax-Music3-mxfp8 \
  --caption "Warm acoustic pop, 96 BPM, intimate female vocal" \
  --lyrics $'[verse]\nMorning light across the room\n[chorus]\nSing with me' \
  --duration 30 \
  --steps 30 \
  --seed 7 \
  --output song.wav
```

Section tags such as `[verse]` and `[chorus]` should be placed on their own
lines. Use `--lyrics-file lyrics.txt` for longer songs.

```python
from mlx_audio.music import load

model = load("mlx-community/MiniMax-Music3-mxfp8")
result = next(
    model.generate(
        text="Warm acoustic pop, 96 BPM, intimate female vocal",
        lyrics="[verse]\nMorning light across the room\n[chorus]\nSing with me",
        duration=30,
        steps=30,
        seed=7,
    )
)
```

Lyrics are required by the checkpoint contract. Use `[instrumental]` explicitly
for instrumental generation. The maximum requested duration is 360 seconds;
the autoregressive stage may emit its end token earlier. Caption controls such
as instrumentation and tempo are probabilistic rather than strict.

## Manual conversion

The general converter can read the official modular checkpoint directly:

```bash
python -m mlx_audio.convert \
  --hf-path MiniMaxAI/MiniMax-Music3 \
  --mlx-path ./MiniMax-Music3-mxfp8 \
  --quantize \
  --q-mode mxfp8
```

The MLX implementation is adapted from
[`mikolaj92/minimax-music3-mlx`](https://github.com/mikolaj92/minimax-music3-mlx)
under Apache-2.0. MiniMax Music 3 weights and tokenizer are released under the
[`MiniMax-Music3 Community License`](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE).
