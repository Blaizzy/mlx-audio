# MiniMax Music 3

Native MLX inference for
[`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
including the hierarchical autoregressive generator, RVQ depth decoder,
flow-matching transformer, condition encoder, and stereo 44.1 kHz vocoder.

## Convert

The converter reads the official modular Diffusers repository directly. It
quantizes the large generation linears while keeping embeddings, output heads,
convolutions, and the vocoder dense.

```bash
python -m mlx_audio.convert \
  --hf-path MiniMaxAI/MiniMax-Music3 \
  --mlx-path ./MiniMax-Music3-mxfp8 \
  --quantize \
  --q-mode mxfp8

python -m mlx_audio.convert \
  --hf-path MiniMaxAI/MiniMax-Music3 \
  --mlx-path ./MiniMax-Music3-mxfp4 \
  --quantize \
  --q-mode mxfp4
```

The mode defaults are a group size of 32 with 4 bits for `mxfp4`, and a group
size of 32 with 8 bits for `mxfp8`. MXFP8 is recommended when lyric fidelity
matters. MXFP4 uses less memory, but controlled comparisons with dense BF16 and
MXFP8 over multiple seeds found that it may alter or omit more words.

## Generate

```bash
python -m mlx_audio.tts.generate \
  --model ./MiniMax-Music3-mxfp8 \
  --text "Warm acoustic pop, 96 BPM, intimate female vocal" \
  --lyrics $'[verse]\nMorning light across the room\n[chorus]\nSing with me' \
  --gen_duration 30 \
  --steps 30 \
  --seed 7
```

```python
from mlx_audio.tts.utils import load_model

model = load_model("./MiniMax-Music3-mxfp8")
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

## Official implementation parity

The opt-in differential tests instantiate the official PyTorch Qwen3 and
Diffusers components, pass their parameters through the MLX conversion path,
and compare Qwen logits, RVQ depth output, conditioning, flow velocity, the
Euler schedule, and vocoder audio numerically:

```bash
git clone https://github.com/huggingface/diffusers.git /tmp/diffusers
git -C /tmp/diffusers checkout dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d
uv pip install torch==2.8.0
MLX_AUDIO_MINIMAX_MUSIC3_DIFFUSERS=/tmp/diffusers \
  pytest mlx_audio/tts/tests/test_minimax_music3_official_parity.py
```

The implementation includes adapted and modified Apache-2.0 work from
`mikolaj92/minimax-music3-mlx`; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
