# MOSS-TTS Runtime Technical Overview

This document describes how the `moss_tts` MLX runtime is wired internally for Delay, Local, TTSD, VoiceGenerator, and SoundEffect variants.

## Module Map

| File | Responsibility |
|---|---|
| `config.py` | Unified config, Local-vs-Delay detection, Qwen3/local transformer config synthesis |
| `model.py` | High-level generation orchestration, variant routing, sanitize, long-form integration |
| `request.py` | Narrow-waist request normalization (`MossNormalizedRequest`) |
| `processor.py` | Prompt packing, message normalization, reference encode/decode, delay pattern transforms |
| `delay_model.py` | Delay architecture core (global backbone + per-channel heads) |
| `local_model.py` | Local architecture core (global backbone + local transformer autoregression) |
| `backbone.py` | Shared Qwen3-style backbone with KV cache support |
| `inference_utils.py` | Delay scheduler state machine helpers |
| `long_form.py` | Segmentation planner, continuity state, seam metrics |
| `sampling.py` | Per-channel sampling config + repetition penalty logic |
| `presets.py` | Variant preset catalog and alias resolution |

## Runtime Entry Points

- `Model.generate(...)` in `model.py` is the main contract.
- `Model.post_load_hook(...)` initializes tokenizer + codec + processor.
- `Model.sanitize(...)` handles checkpoint key remapping across Local/Delay layouts.

## Request Normalization and Precedence

`Model.generate(...)` normalizes user kwargs through `_resolve_generation_request(...)` into `MossNormalizedRequest`:

- Alias handling: `instruction` and `instruct` must not conflict.
- `ref_text` is appended as instruction context (`"Reference transcript: ..."`).
- `duration_s`/`seconds` convert to tokens at 12.5 Hz only when `tokens` is absent.
- `reference` and `ref_audio` are exclusive.

This keeps the upstream `build_user_message(...)` contract stable at one boundary.

## Prompt Packing and Reference Flow

`MossTTSProcessor.prepare_generation_inputs(...)` constructs unified `(B, T, 1 + NQ)` model inputs.

### Message normalization

- Accepts `Message` objects or dict payloads.
- `mode="generation"` requires final message role `user`.
- `mode="continuation"` requires final message role `assistant`.

### Audio reference path

- References can be waveform paths, waveforms (`mx.array`), or pre-encoded token matrices.
- Pre-encoded layout normalization accepts both `(T, NQ)` and `(NQ, T)` patterns.
- Waveforms are encoded through `MossAudioTokenizer.batch_encode(...)`.

### Delay vs Local packing

- Local keeps direct `(T, NQ)` audio-code alignment.
- Delay applies channel-offset delay coding via `apply_delay_pattern(...)` before packing.

## Generation Loops

### Local Loop

`_generate_local(...)`:

1. Prefill backbone with packed prompt.
2. For each step, sample next channel bundle via `MossTTSLocalModel.sample_next_channels(...)`.
3. Append generated audio rows when channel-0 emits `audio_assistant_gen_slot_token_id`.
4. Stop at `audio_end_token_id` or `effective_max_tokens`.
5. Decode rows through codec and stream/finalize according to `stream`.

Local-only depth control is enforced by `_resolve_local_n_vq_for_inference(...)`.

### Delay Loop

`_generate_delay(...)` uses explicit scheduler state (`DelaySchedulerState`):

1. Initialize continuation/flush state from prompt tail.
2. Build forced text tokens (`build_delay_forced_text_tokens(...)`) and sampling masks.
3. Sample text/audio channels with mask-aware constraints.
4. Update scheduler state (`update_delay_scheduler_state(...)`).
5. Decode complete delay rows through `extract_complete_delay_rows(...)`.

This avoids branch ladders in `model.py`; phase logic stays in `inference_utils.py`.

## Long-Form Orchestration

When `long_form=True`, `Model.generate(...)` routes to `_generate_long_form(...)`:

1. Segment plan from `plan_text_segments(...)` under min/target/max char budgets.
2. For each segment:
   - Merge continuity prefix audio/text.
   - Run single-segment generation.
   - Evaluate boundary seam (`evaluate_segment_boundary(...)`).
   - Advance continuity (`advance_continuity_state(...)`).
3. Emit streaming segment chunks or one merged waveform.
4. Store metrics in `_last_long_form_segment_metrics` and `_last_long_form_boundary_metrics`.

Memory/cache pressure is bounded by explicit `mx.clear_cache()` boundaries per segment attempt.

## Checkpoint Loading and Quantization Guardrails

### Sanitize

`Model.sanitize(...)` has separate key-remap paths:

- Local remaps `model.language_model.*`, `local_transformer.*`, `local_to_speech_embedding_mlps.*`, etc.
- Delay remaps `language_model.*`, `emb_ext.*`, `lm_heads.*`, and drops Local-only groups.

`num_batches_tracked` tensors are skipped.

### Quantization Predicate

`model_quant_predicate(...)` blocks quantization for embeddings and sensitive audio-head paths (plus variant-specific groups).

## Integration Boundaries

`moss_tts` intentionally treats these as edges:

- Tokenizer + chat template behavior (`transformers` tokenizer)
- Audio codec encode/decode (`MossAudioTokenizer`)
- CLI/server wiring (`mlx_audio/tts/generate.py`, `mlx_audio/server.py`)

Core generation flow remains in pure MLX model/runtime modules.

## Contract Tests

Primary regression anchors:

- `mlx_audio/tts/tests/test_moss_tts_local_runtime.py`
- `mlx_audio/tts/tests/test_moss_tts_delay_runtime.py`
- `mlx_audio/tts/tests/test_moss_tts_long_form_runtime.py`
- `mlx_audio/tts/tests/test_moss_tts_bootstrap_safety.py`
- `mlx_audio/tts/tests/test_generate_stream_contracts.py`

Use these when updating runtime contracts or docs.
