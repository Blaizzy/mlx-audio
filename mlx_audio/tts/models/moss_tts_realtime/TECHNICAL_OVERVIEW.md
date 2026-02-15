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

`MossTTSRealtimeProcessor.build_turn_input_ids(...)` packs a turn as `[B, T, channels]` with:

- channel 0: text/control tokens
- channels 1..rvq: audio tokens or `audio_pad_token`

## Inferencer Lifecycle

`MossTTSRealtimeInference` is explicit and stateful:

1. `prefill(...)`: consume turn input + initial text prefix, produce first frame.
2. `step(text_token)`: one frame per new text token (or `text_pad`).
3. `finish(max_steps)`: continue until EOS/cap.
4. `reset_turn(...)` / `reset_generation_state(...)`: clear turn state and optionally cache.

Cache growth is bounded by `_ensure_cache_capacity(...)`; cache is rebuilt when context cap is exceeded.

## Session Lifecycle and Invariants

`RealtimeSession` wraps inferencer + decoder and enforces sequencing:

- Required order: `reset_turn` -> `push_text`/`push_text_tokens` -> `end_text` -> `drain`.
- `reset_turn`/`reset` during active undrained turns raises.
- `close()` drains active turns before shutdown.

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
3. Reset turn, optionally encode prompt audio.
4. Push text tokens, `end_text`, then `drain`.
5. Build `GenerationResult` chunks or merged final audio.

For strict lifecycle/latency control, call session APIs directly.

## Known Follow-Ups (Tracker Date: 2026-02-15)

The progress tracker currently lists:

- `P5-STAB-04`: stream-first-yield behavior in high-level `Model.generate(stream=True)`
- `P5-STAB-05`: square tie normalization for pre-encoded realtime audio tokens

Reference: `../../../../PLANS/MOSS-TTS-PLANS/moss_tts_master_plan_progress_tracker.md`

## Contract Tests

Primary regressions:

- `mlx_audio/tts/tests/test_moss_tts_realtime_runtime.py`
- `mlx_audio/tts/tests/test_generate_stream_contracts.py`

These tests define the current runtime contracts for transitions, decode flow control, and parity behaviors.
