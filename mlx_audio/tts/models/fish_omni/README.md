# Fish Audio Drama 3

MLX port of [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3): a
35B-parameter (3B active) MoE text-to-speech model with up to 16 speakers,
voice cloning, inline cues such as `[laughs]`, and editing by continuation.

Built with Fish Audio. The weights are under the
[Fish Audio Community License](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE).

## Architecture

- **Backbone:** a Qwen3.5-MoE hybrid of Gated DeltaNet and gated full attention,
  with 256 experts (8 active), from the vendored `mlx_audio.lm.models.qwen3_5`.
  Its vocabulary is extended with 4,096 semantic tokens.
- **Codebook tower** (`tower.py`): 4 shared layers plus 9 residual layers that
  predict codes 1–9 of each frame. Unlike the S2 Pro fast decoder, every tower
  layer attends over *time* and keeps a KV cache across frames.
- **Codec:** the S2 Pro codec (`fish_s1_dac`); `codec.pth` is byte-identical.
  It runs at 44.1 kHz, 21.5 frames/s.

## Pre-converted weights

| Repo | Experts | Size | Peak memory |
|---|---|---|---|
| [cguldogan/drama-3-mlx-q4](https://huggingface.co/cguldogan/drama-3-mlx-q4) | 4-bit | 24 GB | ~30 GB |
| [cguldogan/drama-3-mlx-q8](https://huggingface.co/cguldogan/drama-3-mlx-q8) | 8-bit | 40 GB | ~46 GB |

Pass either id as `--model` (or to `load_model`). To build your own, see Convert.

## Convert

The release checkpoint stores its experts in block-FP8, so it has to be converted
first. The converter streams one layer at a time and peaks around 10 GB of
memory. Each conversion takes under a minute:

```bash
# ~24 GB: 4-bit routed experts
python -m mlx_audio.tts.models.fish_omni.convert --output drama-3-mlx-q4 --expert-bits 4
# ~40 GB: 8-bit routed experts
python -m mlx_audio.tts.models.fish_omni.convert --output drama-3-mlx-q8 --expert-bits 8
```

Only the routed experts (`--expert-bits`) and the token embedding
(`--embed-bits`, default 8) are quantized by default. The attention projections
and shared experts (`--dense-bits`), the tower (`--tower-bits`), the router and
the sliced `lm_head` stay in bf16. See "Precision notes" for why.

## Use

```bash
python -m mlx_audio.tts.generate --model drama-3-mlx-q4 \
  --text "<|speaker:0|> [sighs] Fine. <|speaker:1|> [laughs] Well, maybe." \
  --ref_audio voice.wav --ref_text "Transcript of voice.wav." \
  --temperature 1.0 --top_p 0.9 --top_k 30
```

`model.generate` defaults to the official server's sampling: temperature 1.0,
top_p 0.9, top_k 30. The CLI's own defaults (temperature 0.7, top_k 50) override
those, so pass them explicitly as above. Lines heavy with cues such as
`[laughs]` still miss on roughly 15–20% of seeds at these settings, running on
past the text; re-roll the seed when that happens.

```python
from mlx_audio.tts.utils import load_model

model = load_model("drama-3-mlx-q4")
for result in model.generate(text, ref_audio=audio_44k, ref_text=transcript):
    ...  # one GenerationResult per text chunk
```

Long text is chunked the way the hosted server chunks it: about 300 bytes and at
most 3 speakers per chunk. Each chunk's prompt carries the first chunk and the
latest one as history. Pre-encoded references can be passed as
`ref_codes=[codes]` and `ref_texts=[transcript]`.

To edit by continuation, pass earlier takes back in as history. Each result
carries its `text` and `codes`:

```python
take = next(model.generate("Due on Wednesday, by noon.", ref_audio=a, ref_text=t))
fixed = next(model.generate("Due on Thursday, by noon.", ref_audio=a, ref_text=t,
                            previous_takes=[(take.text, take.codes)]))
```

## Test UI

A local page for trying voices, scripts and sampling settings (needs the `server`
extra):

```bash
python -m mlx_audio.tts.models.fish_omni.app \
  --models drama-3-mlx-q4 drama-3-mlx-q8 \
  --voice ref.wav "Transcript of ref.wav."      # optional, repeatable
# then open http://127.0.0.1:7860
```

The page has:

- a model switcher (one model is loaded at a time)
- reference voices, each with a transcript and a speaker number
- a script editor with buttons for speaker tags and `[laughs]`/`[sighs]`
- sampling sliders and a seed field
- chunks that play as soon as they finish
- a list of takes, each with WAV download and "Edit from this take", which
  continues from it via `previous_takes`

## Reference audio

Pass references as file paths, or resample them with `load_reference`. Don't
use `mlx_audio.utils.load_audio(..., sample_rate=44100)`: when it upsamples
(e.g. from a 24 kHz clip) it uses miniaudio's linear resampler. The imaging
noise that leaves gets encoded into the voice codes, and in testing it made
generations ramble past the text in most seeds (1–2 of 6 correct, against
5–6 of 6 with a clean resample).

## Precision notes

Parity was measured against the fish-speech PyTorch reference, with the backbone
truncated to 4 layers and the full tower, teacher-forced on real codec codes. The
yardstick is how far the production configuration (bf16 with FP8 W8A8 experts)
drifts from fp32:

- **The tower amplifies small errors.** Its deep codebook heads move noticeably
  even in the reference's own bf16. Main-token logits, and therefore
  intelligibility and WER, barely move. Timbre and detail live in the heads.
- Quantizing the routed experts to 8 bits stays at production level; at 4 bits,
  head 0 drifts about 4x further. 8-bit embeddings are harmless.
- 8-bit attention projections, shared experts or tower weights each made the
  heads 5–30x worse than production. That's why they stay bf16.
- The tower's zero-centered RMSNorms keep `w` as stored and compute `1 + w` in
  fp32, like the reference. Storing `1 + w` in bf16 wrecked the deep heads.

## Speed

Steady-state decode on an M4 Pro with 64 GB, where real time is 21.5 frames/s:

- **q4:** about 31 frames/s
- **q8:** about 28 frames/s

Long chunks with history (prompts of around 800 rows) run closer to 20 frames/s
once prefill and codec decode are counted. The first generation in a process
also pays for Metal kernel compilation and the lazy weight load.

Not implemented yet:

- streaming output within a chunk
- overlapping one decode step's graph build with the previous step's GPU work.
  Graph building is only about 2 ms of the ~32 ms frame, so the gain would be small.
