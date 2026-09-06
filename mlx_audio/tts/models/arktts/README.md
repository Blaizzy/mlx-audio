# Audio8 TTS (arktts)

Multilingual zero-shot voice cloning with a bundled 44.1 kHz neural codec.

## Supported Models

| | slow backbone | LM params | languages |
|---|---|---|---|
| `mlx-community/Audio8-TTS-Preview-0.6b-bf16` | pure attention | 601M | 11 |
| `mlx-community/Audio8-TTS-Preview-0.1b-bf16` | Falcon-H1 hybrid | 170M | 8 |

Both are selected automatically from `slow_backbone` in `config.json`; there is one model
class. Based on [Audio8/Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
and [Audio8/Audio8-TTS-Preview-0.1b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.1b).

**Licences differ.** The 0.6b is Apache-2.0. The 0.1b is the Audio8 Community License v1.0
(revenue-capped: free non-commercial, free commercial under US$2M annual revenue, written
licence at or above that). The bundled codec is byte-identical between the two upstream
repos, so the converted codec tensors are shared and carry the 0.6b's Apache-2.0 terms.

## Usage

Python API:

```python
import soundfile as sf
from mlx_audio.tts.utils import load

model = load("mlx-community/Audio8-TTS-Preview-0.6b-bf16")

# Zero-shot cloning. The reference transcript must match what the clip says —
# the model conditions on the (audio, text) pair, not on the audio alone.
for result in model.generate(
    text="Welcome to Audio8 TTS, running on Apple Silicon.",
    ref_audio="reference.wav",
    ref_text="Transcript of the reference clip.",
):
    sf.write("output.wav", result.audio, result.sample_rate)
```

Without `ref_audio` the model synthesizes with its own default voice.

CLI (default voice — the CLI has no flags for a reference clip, so cloning is Python-API only):

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Audio8-TTS-Preview-0.6b-bf16 \
  --text "Welcome to Audio8 TTS, running on Apple Silicon."
```

## Languages

Cantonese, Chinese, Dutch, English, French, German, Italian, Japanese, Korean, Polish, Spanish.

Language coverage is intentionally limited in this preview release; results are best within the
list above.

## Options

| | default | |
|---|---|---|
| `temperature` | 0.7 | |
| `top_p` | 0.9 | |
| `top_k` | 50 | |
| `max_tokens` | 512 | frames; one frame ≈ 46 ms of audio, so 512 ≈ 23.8 s |
| `do_sample` | `True` | set `False` for deterministic argmax decoding |
| `seed` | — | reproducible sampling |

The upstream demo Space runs hotter (`0.8 / 0.95 / 1024`) than the model card's defaults, which
are used here.

## Architecture

DualAR, in the style of Fish Audio S2 Pro. The 0.1b (`slow_backbone: falcon_h1`) swaps the
slow stack for a Falcon-H1 hybrid — every layer carries a Mamba-2 mixer, attention, and an
MLP — consumed from `mlx_lm.models.falcon_h1`, and adds a dedicated `semantic_output` head
emitting COMPACT logits of width `codebook_size + 1` (index `i` = semantic token
`semantic_begin_id + i`; index `codebook_size` = EOS) instead of full-vocabulary logits tied
to the input embedding. Its fast AR, prompt layout, sampling, and codec are unchanged.

One trap is worth knowing before touching the falcon path: `embedding_multiplier` is applied
to the COMPOSITE embedding (text + the ten codebook embeddings), not to the token lookup.
`sanitize()` therefore deliberately does NOT fold it into `embed_tokens.weight` the way
`mlx_lm`'s own FalconH1 sanitize does. Folding it scales the text half and leaves the
codebook half at 1.0 — measured against the PyTorch reference that is 77% relative error on
the embedding and 38% on the final hidden state, while still producing plausible audio of
the right length.

- **Slow AR** (0.6b) — 24 layers, width 896, 14 heads / 2 KV heads. Emits one semantic token
  per audio frame.
- **Fast AR** — 4 layers, same width. Emits that frame's 10 residual codec codebooks, conditioned
  on the slow hidden state and the preceding codebooks.
- **Codec** — 44.1 kHz, 2048 samples per frame (~21.5 frames/s). DAC-style encoder/decoder with a
  split semantic + residual RVQ and windowed transformer pre/post modules. Handles both
  reference-audio encoding and waveform decoding, so no separate codec checkpoint is needed.

Sampling reproduces the reference implementation exactly: semantic-range logit filtering, the
legacy top-k/top-p order (filter *before* temperature), exponential-race sampling, and RAS
repetition rescue.

## Conversion notes

The upstream repository ships its codec as a PyTorch `.pth` with unfused weight norm. The
converted `mlx-community` weights are **pre-sanitized** — weight norm folded, conv weights in
MLX's channels-last layout, Snake alphas reshaped — so `sanitize()` passes them through
unchanged. It stays able to consume a raw upstream checkpoint as well.

## Parity

Verified against the PyTorch reference on CPU in fp32, for both checkpoints:

| | 0.6b | 0.1b |
|---|---|---|
| unit / block outputs | max-abs 1e-6 … 1e-4 | max-abs 1e-6 … 3e-5 |
| prefill final hidden | max-abs 6.1e-5 | max-abs 2.1e-5 |
| reference-audio codec encode | **100% code-exact** | **100% code-exact** |
| greedy generation | **100% token-exact** (102 frames) | **100% token-exact** (107 frames) |
| decoded waveform | max-abs 7.5e-6 | max-abs 1.2e-5 |

Steady-state throughput, same text and reference clip, deterministic decoding, first run
discarded as warm-up:

| | 0.6b | 0.1b |
|---|---|---|
| RTF | 0.33 | **0.28** |
| peak memory | 8.17 GB | **7.59 GB** |

The gains are modest relative to the 3.5x LM shrink because the codec — identical in both —
dominates both throughput and peak activation.
