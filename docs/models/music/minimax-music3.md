# MiniMax Music 3

MiniMax Music 3 is a multilingual song-generation model that combines a Qwen3
autoregressive planner, an RVQ depth decoder, a flow-matching transformer, and
a stereo 44.1 kHz vocoder. The original checkpoint is
[`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3).

## Models

| Model | Recommendation |
|---|---|
| [`mlx-community/MiniMax-Music3-bf16`](https://huggingface.co/mlx-community/MiniMax-Music3-bf16) | Dense reference conversion |
| [`mlx-community/MiniMax-Music3-8bit`](https://huggingface.co/mlx-community/MiniMax-Music3-8bit) | Affine 8-bit conversion |
| [`mlx-community/MiniMax-Music3-6bit`](https://huggingface.co/mlx-community/MiniMax-Music3-6bit) | Affine 6-bit conversion |
| [`mlx-community/MiniMax-Music3-4bit`](https://huggingface.co/mlx-community/MiniMax-Music3-4bit) | Affine 4-bit conversion |
| [`mlx-community/MiniMax-Music3-mxfp8`](https://huggingface.co/mlx-community/MiniMax-Music3-mxfp8) | Recommended balance of memory and lyric fidelity |
| [`mlx-community/MiniMax-Music3-mxfp4`](https://huggingface.co/mlx-community/MiniMax-Music3-mxfp4) | Experimental lower-memory conversion |
| [`mlx-community/MiniMax-Music3-nvfp4`](https://huggingface.co/mlx-community/MiniMax-Music3-nvfp4) | Experimental NVFP4 conversion |

!!! warning "Choose MXFP8 when lyric fidelity matters"
    MXFP4 uses less memory, but may alter or omit more requested words than
    BF16 or MXFP8.

## Generate

=== "CLI"

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

    For longer lyrics, pass a UTF-8 file with `--lyrics-file lyrics.txt`.

=== "Python"

    ```python
    from mlx_audio.music import load

    model = load("mlx-community/MiniMax-Music3-mxfp8")

    for result in model.generate(
        text="Warm acoustic pop, 96 BPM, intimate female vocal",
        lyrics="[verse]\nMorning light across the room\n[chorus]\nSing with me",
        duration=30,
        steps=30,
        seed=7,
    ):
        print(result.audio.shape, result.sample_rate)
    ```

Place section tags such as `[verse]`, `[chorus]`, and `[bridge]` on their own
lines. Lyrics are required by the checkpoint contract; use `[instrumental]`
explicitly for instrumental generation.

Generation supports up to 360 requested seconds, but the autoregressive model
may emit its end token earlier. Caption controls such as tempo and
instrumentation are probabilistic rather than strict.

## Manual conversion

The general converter reads the official modular checkpoint directly:

```bash
python -m mlx_audio.convert \
    --hf-path MiniMaxAI/MiniMax-Music3 \
    --mlx-path ./MiniMax-Music3-mxfp8 \
    --quantize \
    --q-mode mxfp8
```

The MLX implementation is adapted from
[`mikolaj92/minimax-music3-mlx`](https://github.com/mikolaj92/minimax-music3-mlx)
under Apache-2.0. The model weights and tokenizer use the
[`MiniMax-Music3 Community License`](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE).
