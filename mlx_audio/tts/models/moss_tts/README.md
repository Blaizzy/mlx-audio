# MOSS-TTS Family (MLX Runtime)

Unified MLX runtime support for the OpenMOSS MOSS-TTS family.

Supported checkpoints:

- `OpenMOSS-Team/MOSS-TTS` (Delay)
- `OpenMOSS-Team/MOSS-TTS-Local-Transformer` (Local)
- `OpenMOSS-Team/MOSS-TTSD-v1.0` (TTSD)
- `OpenMOSS-Team/MOSS-Voice-Generator` (VoiceGenerator)
- `OpenMOSS-Team/MOSS-SoundEffect` (SoundEffect)

Realtime has a dedicated runtime package: `mlx_audio/tts/models/moss_tts_realtime/`.

## Model Variants

| Variant | HF ID | Runtime | Preset | Primary Use |
|---|---|---|---|---|
| Delay | `OpenMOSS-Team/MOSS-TTS` | `moss_tts` | `moss_tts` | General high-capability TTS |
| Local | `OpenMOSS-Team/MOSS-TTS-Local-Transformer` | `moss_tts` | `moss_tts_local` | Lower-memory TTS, inference depth override |
| TTSD | `OpenMOSS-Team/MOSS-TTSD-v1.0` | `moss_tts` | `ttsd` | Multi-speaker dialogue |
| VoiceGenerator | `OpenMOSS-Team/MOSS-Voice-Generator` | `moss_tts` | `voice_generator` | Voice design from text instruction |
| SoundEffect | `OpenMOSS-Team/MOSS-SoundEffect` | `moss_tts` | `soundeffect` | Text-to-sound-event synthesis |

## Quick Start

### Python API

```python
from mlx_audio.tts.utils import load_model

model = load_model("OpenMOSS-Team/MOSS-TTS-Local-Transformer")

results = list(
    model.generate(
        text="Hello from MOSS on MLX.",
        preset="moss_tts_local",
        input_type="text",  # text | pinyin | ipa
        duration_s=6.0,       # mapped to tokens at 12.5 Hz when tokens is omitted
        max_tokens=240,
    )
)

audio = results[0].audio
```

### CLI

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --text "Hello from MOSS on MLX." \
  --preset moss_tts_local \
  --duration_s 6 \
  --output_path ./outputs/moss_local
```

More complete runnable examples are in:

- `examples/moss_tts_basic.py`
- `examples/moss_tts_voice_cloning.py`
- `examples/moss_ttsd_dialogue.py`
- `examples/moss_voice_design.py`
- `examples/moss_sound_effects.py`
- `examples/moss_tts_long_form.py`
- `examples/moss_tts_pronunciation_control.py`

## Generation Controls

### Common fields

| Field | Meaning |
|---|---|
| `text` | Primary user text |
| `ref_audio`, `ref_text` | Voice/reference conditioning |
| `instruct` | Style or voice instruction |
| `tokens` | Explicit target token budget |
| `duration_s` / `seconds` | Convenience duration mapped at 12.5 tokens/sec |
| `quality` | Quality hint string (variant-dependent behavior) |
| `input_type` | `text`, `pinyin`, or `ipa` |
| `preset` | Variant sampling defaults |
| `stream`, `streaming_interval` | Chunked output controls |
| `repetition_window` | Realtime-only repetition-penalty history window |

### `quality` hint (advisory)

This integration treats `quality` as a user hint string. Recommended values:

- `draft`, `balanced`, `high`, `max`, or `custom:<label>`

For Delay-family variants in this package, `quality` is passed through verbatim into the prompt (unknown values are allowed).

### Precedence and validation rules

- `tokens` wins over `duration_s`/`seconds` if both are provided.
- `duration_s`/`seconds` must be positive.
- `ref_audio` and raw `reference` cannot be provided together.
- `ref_audio` may be a path, waveform (`mx.array`), or pre-encoded codec tokens (`(T, NQ)` or `(NQ, T)`).
- `n_vq_for_inference` is Local-only (`1..config.n_vq`).
- `conversation` and `dialogue_speakers` are mutually exclusive.
- `long_form=True` cannot be combined with `conversation` or `dialogue_speakers`.

### `input_type` pronunciation semantics

- `input_type` is a mlx-audio affordance for validation and explicit helper workflows.
- The upstream user prompt continues to carry only text/reference-style fields; `input_type`
  itself is not injected as an upstream message field.
- `input_type="text"`: no pronunciation-specific validation.
- `input_type="pinyin"`: requires tone-numbered, whitespace-separated pinyin syllables
  (for example, `ni3 hao3`); obvious non-pinyin payloads fail fast.
- `input_type="ipa"`: requires one or more balanced `/.../` IPA spans; malformed slash
  delimiters fail fast.
- No silent conversion is performed during generation. Conversion helpers are opt-in.

Optional helper entrypoints:

```python
from mlx_audio.tts.models.moss_tts import (
    convert_text_to_tone_numbered_pinyin,
    convert_text_to_ipa,
)

pinyin_text = convert_text_to_tone_numbered_pinyin("您好，请问您来自哪座城市？")
ipa_text = convert_text_to_ipa("Hello, may I ask which city you are from?")
```

Optional helper dependencies can be installed separately (for example,
`pip install pypinyin phonemizer-fork deep-phonemizer`).

### Variant-focused controls

| Variant | Key fields |
|---|---|
| Delay / Local | `text`, `input_type`, `instruct`, `ref_audio`, `ref_text`, `tokens`/`duration_s` |
| TTSD | `text` + `dialogue_speakers` schema |
| VoiceGenerator | `instruct` (+ optional `normalize_inputs`) |
| SoundEffect | `ambient_sound`, `sound_event`, optional `quality` |
| Realtime | `chunk_frames`, `overlap_frames`, `decode_chunk_duration`, `max_pending_frames`, `repetition_window` + session voice-prompt API |

For full effective-field contracts see:
`../../../../PLANS/MOSS-TTS-PLANS/artifacts/phase7/p7_effective_field_matrix.md`

## TTSD `dialogue_speakers` Schema

`dialogue_speakers_json` should contain a JSON list of speaker mappings:

```json
[
  {
    "speaker_id": 1,
    "ref_audio": "path/to/speaker1.wav",
    "ref_text": "Speaker one reference transcript"
  },
  {
    "speaker_id": 2,
    "ref_audio": "path/to/speaker2.wav",
    "ref_text": "Speaker two reference transcript"
  }
]
```

Notes:

- `speaker_id` may be one-based (`1..N`) or zero-based (`0..N-1`); runtime normalizes both.
- `ref_text` can be supplied as `text` in each speaker entry.

## Long-Form Mode

Enable segmented long-form synthesis with `long_form=True`.

Main controls:

- `long_form_min_chars`, `long_form_target_chars`, `long_form_max_chars`
- `long_form_prefix_audio_seconds`, `long_form_prefix_audio_max_tokens`
- `long_form_prefix_text_chars`
- `long_form_retry_attempts`

Runtime emits segment/boundary metrics into model attributes after generation:

- `_last_long_form_segment_metrics`
- `_last_long_form_boundary_metrics`

## Preset Catalog

| Preset | Runtime | Defaults |
|---|---|---|
| `moss_tts` | `moss_tts` | `temperature=1.7`, `top_p=0.8`, `top_k=25`, `repetition_penalty=1.0` |
| `moss_tts_local` | `moss_tts` | `temperature=1.0`, `top_p=0.95`, `top_k=50`, `repetition_penalty=1.1` |
| `ttsd` | `moss_tts` | `temperature=1.1`, `top_p=0.9`, `top_k=50`, `repetition_penalty=1.1` |
| `voice_generator` | `moss_tts` | `temperature=1.5`, `top_p=0.6`, `top_k=50`, `repetition_penalty=1.1` |
| `soundeffect` | `moss_tts` | `temperature=1.5`, `top_p=0.6`, `top_k=50`, `repetition_penalty=1.2` |

## Practical Notes

- Streaming CLI mode requires an output sink (`--output_path`) or `--play`.
- VoiceGenerator defaults to normalized prompt inputs unless overridden (`normalize_inputs=False`).
- CLI convenience: if you provide `--ref_audio` without `--ref_text`, the default `mlx_audio.tts.generate` flow will transcribe the reference audio using Whisper; provide `--ref_text` to avoid extra downloads/latency.
- SoundEffect can synthesize from `ambient_sound` even when `text` is omitted.
- The shared codec is mandatory at runtime; see codec docs below.

## Technical Docs and Artifacts

- Architecture details: `TECHNICAL_OVERVIEW.md`
- Realtime runtime docs: `../moss_tts_realtime/README.md`
- Realtime architecture: `../moss_tts_realtime/TECHNICAL_OVERVIEW.md`
- Codec contracts: `../../../codec/models/moss_audio_tokenizer/README.md`
- Canonical model IDs / aliases:
  `../../../../PLANS/MOSS-TTS-PLANS/artifacts/phase7/p7_canonical_model_ids_and_aliases.md`
- `quality` taxonomy contract:
  `../../../../PLANS/MOSS-TTS-PLANS/artifacts/phase7/p7_quality_taxonomy_contract.md`
- Schema/watchlist contract:
  `../../../../PLANS/MOSS-TTS-PLANS/artifacts/phase7/p7_schema_versioned_request_contract_and_watchlist.md`
