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
    repetition_window=50,
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

## Capability Recipes (No External Scripts)

### 1) Session API (recommended for interactive pipelines)

```python
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.models.moss_tts_realtime import MossTTSRealtimeInference, RealtimeSession
from mlx_audio.tts.utils import load_model

output_dir = Path("outputs/moss_realtime_session")
output_dir.mkdir(parents=True, exist_ok=True)

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
    repetition_window=50,
)

chunks = []
try:
    # Optional persistent voice prompt for this conversation:
    session.set_voice_prompt_audio("voice_prompt.wav")

    session.reset_turn(
        user_text="",
        user_audio_tokens=None,
        include_system_prompt=True,
        reset_cache=False,
    )
    chunks.extend(session.push_text("Hello realtime "))
    chunks.extend(session.push_text("world."))
    chunks.extend(session.end_text())
    chunks.extend(session.drain())
finally:
    session.close()

if not chunks:
    raise RuntimeError("No realtime audio chunks generated")
merged = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)
audio_write(
    str(output_dir / "session_turn.wav"),
    np.array(merged),
    model.sample_rate,
    format="wav",
)
```

### 2) Delta-bridge API for LLM token/delta streams

Use `session.bridge_text_stream(...)` or the top-level `bridge_text_stream(...)`
helper:

```python
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.models.moss_tts_realtime import (
    MossTTSRealtimeInference,
    RealtimeSession,
    bridge_text_stream,
)
from mlx_audio.tts.utils import load_model

output_dir = Path("outputs/moss_realtime_session")
output_dir.mkdir(parents=True, exist_ok=True)

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
    repetition_window=50,
)
deltas = ["Hello ", "realtime ", "world."]

chunks = []
try:
    session.reset_turn(
        user_text="",
        include_system_prompt=True,
        reset_cache=False,
    )
    chunks.extend(bridge_text_stream(session, deltas, hold_back=0, drain_step=1))
finally:
    session.close()

if not chunks:
    raise RuntimeError("No realtime chunks emitted from delta bridge")
merged = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)
audio_write(
    str(output_dir / "delta_bridge.wav"),
    np.array(merged),
    model.sample_rate,
    format="wav",
)
```

### 3) Multiturn flow (persistent voice prompt + per-turn user audio)

```python
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.models.moss_tts_realtime import MossTTSRealtimeInference, RealtimeSession
from mlx_audio.tts.utils import load_model

output_dir = Path("outputs/moss_realtime_session")
output_dir.mkdir(parents=True, exist_ok=True)

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
    repetition_window=50,
)

turns = [
    {
        "user_text": "I have six hours in Paris. What should I prioritize?",
        "user_audio_tokens": "REFERENCE/MOSS-TTS-GitHub/moss_tts_realtime/audio/user1.wav",
        "assistant_deltas": ["Start near the Seine. ", "Visit one museum. ", "Leave buffer for transit."],
    },
    {
        "user_text": "I prefer a relaxed outdoor route.",
        "user_audio_tokens": "REFERENCE/MOSS-TTS-GitHub/moss_tts_realtime/audio/user2.wav",
        "assistant_deltas": ["Great. ", "Walk Notre-Dame to the gardens. ", "End by the river at sunset."],
    },
]

session.set_voice_prompt_audio("REFERENCE/MOSS-TTS-GitHub/moss_tts_realtime/audio/user1.wav")

try:
    for turn_idx, turn in enumerate(turns):
        session.reset_turn(
            user_text=turn["user_text"],
            user_audio_tokens=turn["user_audio_tokens"],
            include_system_prompt=(turn_idx == 0),
            reset_cache=False,
        )
        chunks = list(
            session.bridge_text_stream(
                turn["assistant_deltas"],
                hold_back=0,
                drain_step=1,
            )
        )
        if not chunks:
            raise RuntimeError(f"No chunks for turn {turn_idx}")
        merged = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)
        audio_write(
            str(output_dir / f"turn_{turn_idx:02d}.wav"),
            np.array(merged),
            model.sample_rate,
            format="wav",
        )
finally:
    session.close()
```

### 4) CLI recipes

One-shot realtime:

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Realtime \
  --text "Realtime check from MLX." \
  --preset realtime \
  --output_path ./outputs/moss_realtime_cli
```

Streaming realtime with explicit realtime controls via `model_kwargs_json`:

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Realtime \
  --text "Stream this realtime response." \
  --preset realtime \
  --stream \
  --model_kwargs_json '{"chunk_frames":40,"overlap_frames":4,"decode_chunk_duration":0.32,"max_pending_frames":4096,"repetition_window":50}' \
  --output_path ./outputs/moss_realtime_cli_stream
```

## Main Controls

| Field | Default | Purpose |
|---|---|---|
| `include_system_prompt` | `True` | Include built-in realtime system prompt at turn start |
| `reset_cache` | `True` in high-level request | Drop cache on turn reset |
| `chunk_frames` | `40` | Decoder chunk size in audio-token frames |
| `overlap_frames` | `4` | Crossfade overlap in frames |
| `decode_chunk_duration` | `0.32` | Codec decode chunk override |
| `max_pending_frames` | `4096` | Backpressure guard for queued token frames |
| `repetition_window` | `50` | Repetition-penalty history window (`None`/`<=0` = unbounded) |

Sampling defaults come from preset `realtime`:

- `temperature=0.8`
- `top_p=0.6`
- `top_k=30`
- `repetition_penalty=1.1`
- `repetition_window=50`

## Stream Assembly Guarantees

- Emitted streaming audio is sample-conservative relative to decoded chunk totals.
- Overlap crossfade uses the current chunk overlap window and adapts when the
  final flush chunk is shorter than steady chunks.
- Final flush clears overlap state, preventing stale-tail carryover into the
  next turn/session.

## Voice Prompt vs Turn Audio (Upstream Parity)

Realtime now mirrors upstream separation between:

- persistent **voice prompt** timbre (system prompt rows): `set_voice_prompt_tokens(...)`, `set_voice_prompt_audio(...)`, `clear_voice_prompt_tokens()`
- per-turn **user audio conditioning** (user prompt rows): `reset_turn(..., user_audio_tokens=...)`

High-level one-shot generation keeps user ergonomics simple:

- `model.generate(..., ref_audio=...)` maps `ref_audio` into the persistent voice-prompt path for that request.

Session API behavior:

- call voice-prompt setter once per conversation (or whenever timbre changes)
- pass `user_audio_tokens` only for turn-local conditioning
- `reset_turn(input_ids=...)` remains available as an escape hatch for advanced integrations

## Prompt Packing Contract

The default session path uses upstream-compatible prompt builders:

- `make_ensemble(...)`: system prompt with `<|audio_pad|>` placeholder rows filled by voice-prompt tokens
- `make_user_prompt(...)`: user turn with explicit `delay_tokens_len` alignment + channel-1 BOS/EOS placement
- always appends `"<|im_end|>\n<|im_start|>assistant\n"` boundary rows before streaming response tokens

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
