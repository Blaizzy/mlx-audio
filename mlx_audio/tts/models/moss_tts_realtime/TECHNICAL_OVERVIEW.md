# MOSS-TTS-Realtime Technical Overview

This document explains internals for the dedicated realtime runtime in `mlx_audio/tts/models/moss_tts_realtime/`.

## Module Map

| File | Responsibility |
|---|---|
| `model.py` | Loadable runtime wrapper, sanitize remapping, high-level `generate(...)` |
| `config.py` | Realtime global/local config and token IDs |
| `request.py` | `RealtimeNormalizedRequest` with user-facing defaults/validation |
| `processor.py` | Turn input packing, prompt-audio encode/decode helpers |
| `inference.py` | Prefill/step/finish inferencer, session lifecycle, decode bridge |

## Architecture Shape

`MossTTSRealtimeCore` uses:

- `embedding_list[0]` for text tokens
- `embedding_list[1..rvq]` for audio channels
- shared global backbone (`MossTTSBackbone`)
- local autoregressive decoder (`MossTTSLocalTransformer`)
- per-channel heads (`lm_heads`) + per-head RMSNorm

Generation is two-stage per frame:

1. Global hidden state from multimodal prompt/history.
2. Local channel-by-channel decoding for `rvq` audio tokens.

## Input/Token Contracts

Configured in `ModelConfig`:

- `channels = 1 + rvq`
- audio tokens: `audio_pad_token`, `audio_bos_token`, `audio_eos_token`
- text/reference markers: `text_pad`, `reference_audio_pad`
- prompt alignment: `delay_tokens_len`

`MossTTSRealtimeProcessor.build_turn_input_ids(...)` packs a turn as `[B, T, channels]` with:

- channel 0: text/control tokens
- channels 1..rvq: audio tokens or `audio_pad_token`

## Prompt Packing Parity

Realtime prompt assembly now mirrors upstream split builders:

- `make_ensemble(prompt_audio_tokens=...)`
  - builds system rows
  - fills `<|audio_pad|>` placeholder rows with voice-prompt token frames
- `make_user_prompt(text, audio_tokens)`
  - applies `delay_tokens_len` text/audio alignment
  - inserts channel-1 BOS/EOS at upstream offsets
  - appends `"<|im_end|>\n<|im_start|>assistant\n"` boundary rows

This keeps `RealtimeSession.reset_turn(input_ids=...)` as an escape hatch while making the default path upstream-compatible.

## Inferencer Lifecycle

`MossTTSRealtimeInference` is explicit and stateful:

1. `prefill(...)`: consume turn input + initial text prefix, produce first frame.
2. `step(text_token)`: one frame per new text token (or `text_pad`).
3. `finish(max_steps)`: continue until EOS/cap.
4. `reset_turn(...)` / `reset_generation_state(...)`: clear turn state and optionally cache.

Sampling controls (`temperature`, `top_p`, `top_k`, `repetition_penalty`, `repetition_window`) are threaded through all three generation steps.
`repetition_window` applies windowed penalty over the most recent generated frames per channel.

Cache growth is bounded by `_ensure_cache_capacity(...)`; cache is rebuilt when context cap is exceeded.

## Session Lifecycle and Invariants

`RealtimeSession` wraps inferencer + decoder and enforces sequencing:

- Required order: `reset_turn` -> `push_text`/`push_text_tokens` -> `end_text` -> `drain`.
- `reset_turn`/`reset` during active undrained turns raises.
- `close()` drains active turns before shutdown.
- Session-level voice prompt API is persistent until explicit clear:
  - `set_voice_prompt_tokens(...)`
  - `set_voice_prompt_audio(...)`
  - `clear_voice_prompt_tokens()`

This prevents orphaned buffered tokens/audio between turns.

## Text Ingestion Paths

- `push_text_tokens(...)`: direct token path.
- `push_text(...)`: text fragments are segmented via punctuation/whitespace heuristics.
- `RealtimeTextDeltaBridge`: delta stream adapter using `TextDeltaTokenizer`.

`TextDeltaTokenizer` keeps a full-text retokenization state and emits stable suffix tokens, with optional `hold_back` for tokenizer stability.

## Decode and Backpressure

`AudioStreamDecoder` handles buffered token rows and waveform chunk emission.

Key behaviors:

- bounded pending token frames (`max_pending_frames`)
- chunked decode (`chunk_frames`)
- overlap crossfade (`overlap_frames`)
- explicit final flush behavior

Backpressure is fail-fast: exceeding pending-frame cap raises instead of silently growing memory.

## Checkpoint Loading and Sanitization

`Model.sanitize(...)` remaps multiple upstream key families into runtime parameter names, including:

- global language model to `model.backbone.*`
- embed-token families to `model.embedding_list.*`
- local decoder/head norms to local runtime names

`num_batches_tracked` tensors are dropped.

Quantization guardrails in `model_quant_predicate(...)` block embeddings and output heads from quantization.

## High-Level `Model.generate(...)`

Wrapper flow:

1. Resolve preset (`realtime`) and request defaults (`RealtimeNormalizedRequest`).
2. Build `RealtimeSession` with decode/backpressure controls.
3. If `ref_audio` is provided, map it to session voice-prompt state (`set_voice_prompt_tokens(...)`).
4. Reset turn with turn-local user conditioning (`user_audio_tokens` path remains separate).
5. Push text tokens, `end_text`, then `drain`.
6. In `stream=True` mode, emit chunks incrementally as each stage produces audio (with one-chunk lookahead so `is_final_chunk` is correct).
7. In `stream=False` mode, merge all chunks into one final `GenerationResult`.

For strict lifecycle/latency control, call session APIs directly.

## Streaming Contract: Incremental vs Buffered

`stream=True` must mean "yield as soon as audio exists," not "yield after synthesis ends."

The main pitfall is generator structure:

- Correct streaming: stage work -> yield available chunks -> continue stage work.
- Fake streaming: do all stage work first, then yield from a fully buffered list.

Why this matters:

- Time-to-first-audio: buffered behavior delays first playback until full turn completion.
- Memory: buffered behavior scales with total turn output instead of near-current decode window.
- Client semantics: realtime consumers expect progressive playback and chunk cadence.

Current `Model.generate(stream=True)` behavior is incremental and protected by regression tests.

## Audio Token Layout Normalization Cheat Sheet

Realtime pre-encoded prompt tokens can arrive in either:

- `(T, RVQ)` time-major
- `(NQ, T)` codebook-major

Normalization goal is always internal `(T, RVQ)`.

### Why this is tricky

Shape ties can be ambiguous, especially square matrices (`rows == cols == rvq`).
Both orientations are shape-valid, so incorrect interpretation may silently pass shape checks.

### Rule of thumb

- Prefer explicit axis matches first.
- For ambiguous small, RVQ-aligned ties, prefer canonical codebook-major interpretation (`(NQ, T)`), then transpose to `(T, RVQ)`.

### Current normalization decisions (`_normalize_preencoded_audio_tokens`)

| Input shape signal | Interpretation | Normalized output |
|---|---|---|
| `rows == rvq and cols != rvq` | likely `(NQ, T)` | transpose (`tokens[:rvq, :].T`) |
| `cols == rvq and rows != rvq` | likely `(T, NQ)` | keep (`tokens[:, :rvq]`) |
| `rows >= rvq and cols < rvq` | likely codebook-major prefix | transpose |
| `cols >= rvq and rows < rvq` | likely time-major prefix | keep |
| `rows % rvq == 0 and rows <= 64` | ambiguous RVQ-aligned small tie, bias codebook-major | transpose |
| `cols % rvq == 0 and cols <= 64` | ambiguous RVQ-aligned small tie, bias time-major | keep |
| `rows <= 64 and cols > 64` | small-leading-axis heuristic | transpose |
| `cols <= 64 and rows > 64` | small-trailing-axis heuristic | keep |
| fallback | backward-compatible default | keep (`tokens[:, :rvq]`) |

### Square tie example (important)

If `rvq = 16` and input shape is `16 x 16`, shape alone cannot distinguish axes.
This is why square ties were a frequent source of silent conditioning corruption:

- Wrong orientation does not crash.
- Wrong orientation can still decode.
- Audio quality/conditioning fidelity degrades without obvious runtime errors.

## Why This Area Is Error-Prone

This boundary has repeatedly produced subtle bugs because:

- Raw arrays do not carry explicit layout metadata.
- Multiple modules normalize similar data (codec path + realtime path).
- "Looks valid" shape checks can hide semantic axis flips.
- Realtime performance bugs (buffering) can be masked in short test inputs.

When modifying this area, treat layout + streaming semantics as load-bearing contracts.

## Contract Tests

Primary regressions:

- `mlx_audio/tts/tests/test_moss_tts_realtime_runtime.py`
- `mlx_audio/tts/tests/test_generate_stream_contracts.py`
- `tests/tts/models/moss_tts_realtime/test_realtime_regressions.py`

These tests define current contracts for lifecycle transitions, decode flow control, stream ordering, and ambiguous pre-encoded token normalization.
