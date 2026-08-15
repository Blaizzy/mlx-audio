# MiniMax Music 3

MiniMax Music 3 is a multilingual song-generation model that combines a Qwen3
autoregressive planner, an RVQ depth decoder, a flow-matching transformer, and
a stereo 44.1 kHz vocoder. The original checkpoint is
[`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3).

## Convert

MLX-Audio reads the official modular Diffusers checkpoint directly. The
conversion keeps embeddings, output heads, convolutions, and the vocoder dense,
while quantizing the large generation linears.

=== "MXFP8 (recommended)"

    ```bash
    python -m mlx_audio.convert \
        --hf-path MiniMaxAI/MiniMax-Music3 \
        --mlx-path ./MiniMax-Music3-mxfp8 \
        --quantize \
        --q-mode mxfp8
    ```

=== "MXFP4 (experimental)"

    ```bash
    python -m mlx_audio.convert \
        --hf-path MiniMaxAI/MiniMax-Music3 \
        --mlx-path ./MiniMax-Music3-mxfp4 \
        --quantize \
        --q-mode mxfp4
    ```

Both modes use a group size of 32. `mxfp4` stores quantized linear weights at 4
bits, while `mxfp8` stores them at 8 bits.

!!! warning "Choose MXFP8 when lyric fidelity matters"
    MXFP8 is the recommended quantized format. In controlled checks against a
    dense BF16 conversion over multiple seeds, MXFP8 retained more of the
    requested lyrics. MXFP4 uses less memory, but may alter or omit words and
    should be treated as an experimental quality tradeoff.

## Generate

=== "CLI"

    ```bash
    python -m mlx_audio.tts.generate \
        --model ./MiniMax-Music3-mxfp8 \
        --text "Warm acoustic pop, 96 BPM, intimate female vocal" \
        --lyrics $'[verse]\nMorning light across the room\n[chorus]\nSing with me' \
        --gen_duration 30 \
        --steps 30 \
        --seed 7
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("./MiniMax-Music3-mxfp8")

    for result in model.generate(
        text="Warm acoustic pop, 96 BPM, intimate female vocal",
        lyrics="[verse]\nMorning light across the room\n[chorus]\nSing with me",
        duration=30,
        steps=30,
        seed=7,
    ):
        print(result.audio.shape, result.sample_rate)
    ```

Lyrics are required by the checkpoint contract. Use `[instrumental]` explicitly
for instrumental generation. Generation supports up to 360 requested seconds;
the autoregressive model may emit its end token earlier.

The caption conditions the requested style, tempo, instruments, and vocal
character, but these controls are probabilistic rather than strict. A named
instrument may be subtle or absent in a particular seed; try another seed or a
more explicit arrangement description when exact instrumentation matters.

!!! warning "License"
    MiniMax Music 3 is released under the
    [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE),
    which includes acceptable-use and commercial terms. Review it before using
    or redistributing converted weights.

## Links

- [:octicons-link-external-16: Original model](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
- [:octicons-mark-github-16: In-repo README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/minimax_music3/README.md)
