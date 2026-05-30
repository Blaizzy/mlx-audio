# Higgs Audio v2

Higgs Audio v2 is a Llama-3.2-3B-backed TTS with multi-codebook acoustic tokens and delay-pattern streaming. The MLX port targets the 3B open-weights release from Boson AI and reuses the in-tree HiggsAudio acoustic tokenizer (originally added for OmniVoice).

## Highlights

- Real-time voice cloning on Apple Silicon (RTF ≈ 0.6× bf16 / 0.36× q8 / 0.33× q6 on M5 Max)
- Reference-audio voice cloning via ChatML prompt format
- Full `AUDIO_INIT` + delay-pattern ramp-in/out state machine
- Repetition-avoidance sampling (RAS) for stable long-form output
- MLX native 4/6/8-bit quantization with optional per-layer protection

## Basic usage

### Top-level CLI

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/higgs-audio-v2-3B-mlx-q8 \
  --text "Hello from Higgs Audio on MLX." \
  --ref_audio path/to/reference.wav \
  --ref_text "Transcript of the reference clip."
```

The `Model` class conforms to the standard mlx-audio interface, so the
existing `mlx_audio.tts.generate` CLI and `mlx_audio.server` both work
unchanged against Higgs.

### Python API (standard)

```python
from mlx_audio.tts.utils import load
from mlx_audio.audio_io import write as audio_write

model = load("mlx-community/higgs-audio-v2-3B-mlx-q8")

for result in model.generate(
    text="Hello from Higgs Audio on MLX.",
    ref_audio="path/to/reference.wav",   # optional; strongly recommended
    ref_text="Transcript of the reference clip.",
    temperature=0.7,
    top_p=0.95,
    max_new_frames=1200,
    fade_in_ms=30.0,
):
    audio_write("output.wav", result.audio, result.sample_rate)
```

Without `ref_audio`, generation runs in "smart voice" mode (random voice
per sample). This works but is less reliable than voice cloning — the
sampling occasionally collapses to `stream_eos` early and produces silent
output. If that happens, rerun (each call draws fresh noise) or pass
`ref_audio`. For production use, a reference voice is strongly recommended.

### Python API (Higgs-specific kwargs)

For direct access to the full Higgs parameter surface (RAS windowing,
sampling warmup, pre-loaded codec override, etc.), use `HiggsAudioServer`:

```python
from mlx_audio.tts.models.higgs_audio import HiggsAudioServer
from mlx_audio.audio_io import write as audio_write

server = HiggsAudioServer.from_pretrained(
    model_path="bosonai/higgs-audio-v2-generation-3B-base",     # bf16 base
    codec_path="mlx-community/higgs-audio-v2-tokenizer",        # acoustic tokenizer
)

result = server.generate(
    target_text="Hello from Higgs Audio on MLX.",
    temperature=0.7,
    top_p=0.95,
    max_new_frames=1200,
    fade_in_ms=30.0,
)
audio_write("output.wav", result.pcm, result.sampling_rate)
```

### Recommended parameters

- `temperature=0.7`, `top_p=0.95` — proven stable across prompt lengths during the M5 benchmark
- `max_new_frames=1200` — generous cap; generation stops naturally at the EOS ramp
- `fade_in_ms=30.0`, `fade_out_ms=15.0` — suppresses the first-frame transient that the 5ms default occasionally lets through

## Voice cloning

Pass `ref_audio` (path or pre-loaded mx.array at 24 kHz mono) together with
`ref_text` (the transcript of that clip). Reference audio is encoded through
the in-tree `HiggsAudioTokenizer` and stitched into the assistant turn of a
ChatML prompt — the transcript is required for stable alignment between the
cloned voice and the target text.

```python
from mlx_audio.audio_io import write as audio_write

for result in model.generate(
    text="Hello, this is a cloned voice.",
    ref_audio="reference.wav",
    ref_text="Transcript of the reference clip.",
    temperature=0.7,
    top_p=0.95,
    max_new_frames=1200,
    fade_in_ms=30.0,
):
    audio_write("output.wav", result.audio, result.sample_rate)
```

Best results come from 5–15 seconds of clean reference speech.

### Bundled sample voices

Three drop-in reference voices ship in `examples/voice_prompts/`, generated via Higgs smart-voice mode so they're license-clean:

- `en_woman.wav` — English, feminine register
- `en_man.wav` — English, masculine register
- `en_man_deep.wav` — English, masculine register, lower pitch

Each `.wav` is paired with a matching `.txt` transcript. See `examples/voice_prompts/README.md` for the usage snippet.

## Streaming

For chunked streaming output (e.g. Pipecat pipelines), use
`HiggsAudioServer.generate_stream`:

```python
for pcm_chunk in server.generate_stream(
    target_text="Generating in chunks for live playback.",
    reference_audio_path="reference.wav",
    reference_text="...",
    chunk_ms=640.0,
):
    # emit or resample pcm_chunk (float32 at 24 kHz)
    ...
```

Current shape: full generate, then chunk the resulting PCM. Per-chunk quality matches non-streaming exactly. Mid-generation streaming (emit-as-you-go) is not yet supported because the neural-vocoder codec produces subtly different PCM at the same sample position when called with different accumulated lengths — boundary discontinuities become audible. Proper overlap-add streaming is follow-up work.

## Quantization

MLX native 4/6/8-bit quantization works on the Llama backbone. The audio head and audio codebook embeddings benefit from staying at bf16 — quantizing them introduces voice-character drift (pitch register shifts at q6, trajectory instability at q4).

Already-quantized checkpoints load transparently via `load(...)` — config.json carries a `quantization` block that the framework applies before weight load. To quantize in place on a fresh bf16 load, use `model.model_quant_predicate`:

```python
import mlx.core as mx
import mlx.nn as nn
from mlx_audio.tts.utils import load

model = load("bosonai/higgs-audio-v2-generation-3B-base")
nn.quantize(model, group_size=64, bits=8, class_predicate=model.model_quant_predicate)
mx.eval(model.parameters())
```

Benchmark on M5 Max (warm), long-prompt RTF:

| variant | RTF   | weights size | notes                                       |
|---------|-------|--------------|---------------------------------------------|
| bf16    | 0.60× | 6.8 GB       | `bosonai/higgs-audio-v2-generation-3B-base` (authoritative) |
| q8      | 0.36× | 6.18 GB      | `mlx-community/higgs-audio-v2-3B-mlx-q8`    |
| q6      | 0.33× | 4.75 GB      | `mlx-community/higgs-audio-v2-3B-mlx-q6`    |
| q4      | 0.26× | 3.32 GB      | deferred — seed-sensitive, follow-up PR     |

bf16 is served directly from the authoritative `bosonai/*` upload — no need for a redundant mlx-community re-host. q8 and q6 are MLX-specific selectively-quantized variants.

## Sampling controls

- `temperature=0.7`, `top_p=0.95` are the Higgs defaults.
- `ras_win_len=7`, `ras_max_repeat=2` enables repetition-avoidance sampling (catches near-tie mispicks that compound into loops). Set `ras_win_len=None` to disable.
- `sampling_warmup_frames=N` uses greedy sampling for the first N frames, then switches to temperature. Exposed for experimentation; not helpful at default settings.
- `fade_in_ms=5.0`, `fade_out_ms=5.0` applies a short linear fade to the decoded PCM boundaries. Below onset perception threshold on bf16/q8; masks rounding-click transients on quantized variants.

## Implementation notes

The generation state machine is the non-obvious piece of this port. See source at `mlx_audio/tts/models/higgs_audio/higgs_audio.py:HiggsAudioModel._generate_raw_frames`. The first audio frame is **synthetic all `audio_stream_bos_id`** (AUDIO_INIT) — not sampled from audio_logits at the `<|audio_out_bos|>` text position, because those logits were never trained for direct audio prediction. Without this, the model emits the stream-EOS token on half the codebooks at step 1 and output collapses to a stuck pitch.

Codebook `i` is emitted with `i`-frame delay, so the first K frames are a progressive ramp-in (cb₀ sampled at frame 1, cb₁ at frame 2, etc.; the rest forced to BOS). On any codebook emitting EOS, a K-frame ramp-out begins — trailing codebooks forced to EOS before termination. After `revert_delay_pattern`, the first and last aligned columns are dropped (BOS-seed and EOS-seal — they decode to arbitrary codec token 1023 and produce audible clicks otherwise).

## References

- Original repo: <https://github.com/boson-ai/higgs-audio>
- Paper / blog: <https://boson.ai/blog/higgs-audio-v2>
- HF model (reference): <https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base>
- HF model (MLX q8): <https://huggingface.co/mlx-community/higgs-audio-v2-3B-mlx-q8>
- HF model (MLX q6): <https://huggingface.co/mlx-community/higgs-audio-v2-3B-mlx-q6>
- HF codec: <https://huggingface.co/mlx-community/higgs-audio-v2-tokenizer>
