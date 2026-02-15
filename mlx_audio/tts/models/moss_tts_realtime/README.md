# MOSS-TTS-Realtime (MLX Runtime)

User guide for the dedicated realtime runtime (`model_type = moss_tts_realtime`).

This runtime is optimized for turn-based, incremental text ingestion and chunked audio emission.

## When To Use This Runtime

Use `moss_tts_realtime` when you need:

- incremental text streaming (delta/token push)
- explicit turn lifecycle control
- bounded decode buffering with overlap crossfade
- cache reuse across turns

For non-realtime family variants (Delay/Local/TTSD/VoiceGenerator/SoundEffect), use:
`mlx_audio/tts/models/moss_tts/README.md`

## Quick Start

### High-level `model.generate(...)`

```python
from mlx_audio.tts.utils import load_model

model = load_model("OpenMOSS-Team/MOSS-TTS-Realtime")

for chunk in model.generate(
    text="Realtime check from MLX.",
    preset="realtime",
    stream=True,
    chunk_frames=40,
    overlap_frames=4,
):
    audio = chunk.audio
```

### CLI

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Realtime \
  --text "Realtime check from MLX." \
  --preset realtime \
  --stream \
  --output_path ./outputs/moss_realtime
```

## Session API (Recommended for Interactive Pipelines)

The explicit lifecycle API provides better control than one-shot `generate(...)` for low-latency apps.

```python
from mlx_audio.tts.models.moss_tts_realtime import MossTTSRealtimeInference, RealtimeSession
from mlx_audio.tts.utils import load_model

model = load_model("OpenMOSS-Team/MOSS-TTS-Realtime")

inferencer = MossTTSRealtimeInference(
    model=model.model,
    tokenizer=model.tokenizer,
    config=model.config,
    max_length=1200,
)

session = RealtimeSession(
    inferencer=inferencer,
    processor=model.processor,
    chunk_frames=40,
    overlap_frames=4,
)

try:
    session.reset_turn(user_text="", include_system_prompt=True, reset_cache=False)

    # Push text incrementally.
    chunks = []
    chunks.extend(session.push_text("Hello realtime "))
    chunks.extend(session.push_text("world."))

    # Explicit end-of-text + drain.
    chunks.extend(session.end_text())
    chunks.extend(session.drain())
finally:
    session.close()
```

## Delta-Bridge API

For LLM token/delta streams, use `bridge_text_stream(...)` on an active session:

```python
for chunk in session.bridge_text_stream(["Hello ", "realtime ", "world"], hold_back=0):
    handle(chunk)
```

See runnable reference: `../../../../examples/moss_tts_realtime_text_deltas.py`.

## Main Controls

| Field | Default | Purpose |
|---|---|---|
| `include_system_prompt` | `True` | Include built-in realtime system prompt at turn start |
| `reset_cache` | `True` in high-level request | Drop cache on turn reset |
| `chunk_frames` | `40` | Decoder chunk size in audio-token frames |
| `overlap_frames` | `4` | Crossfade overlap in frames |
| `decode_chunk_duration` | `0.32` | Codec decode chunk override |
| `max_pending_frames` | `4096` | Backpressure guard for queued token frames |

Sampling defaults come from preset `realtime`:

- `temperature=0.8`
- `top_p=0.6`
- `top_k=30`
- `repetition_penalty=1.1`

## Reference Audio Priming

- High-level API: pass `ref_audio` (path or `mx.array`) to `model.generate(...)`.
- Session API: pass `user_audio_tokens` as a waveform `mx.array` or pre-encoded tokens.
  - To prime from a file path, pre-encode first: `prompt_tokens = model.processor.encode_prompt_audio("ref.wav")`, then pass `user_audio_tokens=prompt_tokens`.

Runtime packs prompt audio into reference-audio rows and continues generation from the same turn context.

## Notes

- `stream=True` emits chunks incrementally, with a one-chunk lookahead so only the last chunk is marked `is_final_chunk=True`.
- Pre-encoded prompt audio tokens are accepted in either `(T, RVQ)` or `(NQ, T)` layouts; for ambiguous square ties (`rvq x rvq`), inputs are interpreted as codebook-major `(NQ, T)` and transposed to time-major `(T, RVQ)`.
- If you need strict lifecycle control, prefer the explicit `RealtimeSession` API.

## Additional Docs

- Technical internals: `TECHNICAL_OVERVIEW.md`
- Family runtime docs: `../moss_tts/README.md`
- Family technical docs: `../moss_tts/TECHNICAL_OVERVIEW.md`
- Preset/field artifacts:
  - `../../../../PLANS/MOSS-TTS-PLANS/artifacts/phase7/p7_effective_field_matrix.md`
  - `../../../../PLANS/MOSS-TTS-PLANS/artifacts/phase7/p7_quality_taxonomy_contract.md`
