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
import mlx.core as mx

from mlx_audio.tts.utils import load_model

mx.random.seed(1234)
model = load_model("OpenMOSS-Team/MOSS-TTS-Local-Transformer")

results = list(
    model.generate(
        text="Hello and good morning and good evening from the moss model on MLX.",
        instruct="Calm, clear narrator, medium pace.",
        preset="moss_tts_local",
        input_type="text",    # text | pinyin | ipa
        do_samples=[False],   # optional explicit default for channel-0
        # recommended natural-stop baseline: no tokens/duration_s required
        max_tokens=240,
    )
)

audio = results[0].audio
```

### CLI

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --text "Hello and good morning and good evening from the moss model on MLX." \
  --preset moss_tts_local \
  --seed 1234 \
  --model_kwargs_json '{"do_samples":[false]}' \
  --instruct "Calm, clear narrator, medium pace." \
  --output_path ./outputs/moss_local
```

`--model_kwargs_json '{"do_samples":[false]}'` is optional here and equivalent to
the Local preset default for channel-0; keep it only if you want explicit per-channel
sampling overrides.

## Capability Recipes (No External Scripts)

All recipes below run directly through `mlx_audio.tts.generate`.

### 1) Base TTS (Local variant)

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --preset moss_tts_local \
  --text "Hello from MOSS Local on MLX." \
  --duration_s 6 \
  --output_path outputs/moss_local_base
```

### 2) Base TTS (Delay variant)

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS \
  --preset moss_tts \
  --text "Hello from the Delay runtime." \
  --tokens 120 \
  --seed 1234 \
  --instruct "Calm, clear narrator, medium pace." \
  --output_path outputs/moss_delay_base
```

### 3) Voice cloning (reference-conditioned Delay/Local)

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --preset moss_tts_local \
  --text "This line should follow the reference timbre and cadence." \
  --ref_audio REFERENCE/MOSS-Audio-Tokenizer/demo/demo_gt.wav \
  --ref_text "Demo reference transcript." \
  --tokens 180 \
  --output_path outputs/moss_voice_clone
```

### 4) TTSD multi-speaker dialogue

```bash
cat > /tmp/moss_ttsd_speakers.json <<'JSON'
[
  {
    "speaker_id": 1,
    "ref_audio": "REFERENCE/MOSS-Audio-Tokenizer/demo/demo_gt.wav",
    "ref_text": "Speaker one prompt."
  },
  {
    "speaker_id": 2,
    "ref_audio": "REFERENCE/MOSS-Audio-Tokenizer/demo/demo_gt.wav",
    "ref_text": "Speaker two prompt."
  }
]
JSON

uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --preset ttsd \
  --text "[S1] Thanks for joining. [S2] Happy to help." \
  --dialogue_speakers_json /tmp/moss_ttsd_speakers.json \
  --tokens 220 \
  --output_path outputs/moss_ttsd_dialogue
```

### 5) Voice design (VoiceGenerator)

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-Voice-Generator \
  --preset voice_generator \
  --text "Welcome to the voice design demo." \
  --instruct "Calm premium narrator, warm tone, medium pace." \
  --quality high \
  --tokens 180 \
  --output_path outputs/moss_voice_generator
```

### 6) SoundEffect generation (ambient + event cues)

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-SoundEffect \
  --preset soundeffect \
  --ambient_sound "Thunder and rain over a city street at night." \
  --sound_event storm \
  --quality high \
  --tokens 140 \
  --output_path outputs/moss_soundeffect
```

### 7) Long-form segmented synthesis

```bash
cat <<'TXT' | uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --preset moss_tts_local \
  --long_form \
  --long_form_min_chars 160 \
  --long_form_target_chars 320 \
  --long_form_max_chars 520 \
  --long_form_prefix_audio_seconds 2.0 \
  --long_form_prefix_audio_max_tokens 25 \
  --long_form_retry_attempts 1 \
  --max_tokens 260 \
  --output_path outputs/moss_long_form
Long-form synthesis in MOSS-TTS plans bounded text segments, carries a short
audio tail for continuity, and emits segment/boundary metrics after generation.
TXT
```

### 8) Streaming chunks (non-realtime variants)

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --preset moss_tts_local \
  --text "Stream this response in chunks." \
  --stream \
  --streaming_interval 0.5 \
  --output_path outputs/moss_streamed
```

### 9) Deterministic hybrid decoding (recommended reproducibility mode)

For Local/Delay MOSS runtime, hybrid decoding is already the default behavior:
channel-0 (text/control) is greedy while audio channels remain sampled.

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --preset moss_tts_local \
  --text "Deterministic hybrid run." \
  --duration_s 6 \
  --seed 1234 \
  --output_path outputs/moss_deterministic_hybrid
```

Use `do_samples` only when you want to override per-channel defaults.

For explicit turn lifecycle control and multiturn realtime usage, see
`../moss_tts_realtime/README.md`.

## Advanced Python Recipes

### Continuation prompting (assistant-prefix audio, no `ref_audio` argument)

```python
from pathlib import Path

import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

output_dir = Path("outputs/moss_continuation")
output_dir.mkdir(parents=True, exist_ok=True)

model = load_model("OpenMOSS-Team/MOSS-TTS-Local-Transformer")
processor = model.processor

user_message = processor.build_user_message(
    text="Continue in the same voice and cadence, then close with one sentence.",
    instruction="Warm narrator, steady pace.",
    tokens=180,
    input_type="text",
)
assistant_message = processor.build_assistant_message(
    audio_codes_list=["REFERENCE/MOSS-Audio-Tokenizer/demo/demo_gt.wav"],
    content="<|audio|> Continue in the same voice and cadence.",
)

results = list(
    model.generate(
        conversation=[user_message, assistant_message],
        preset="moss_tts_local",
        max_tokens=320,
    )
)
audio_write(
    str(output_dir / "continuation.wav"),
    np.array(results[0].audio),
    results[0].sample_rate,
    format="wav",
)
```

### Long-form with metric capture

```python
from dataclasses import asdict

from mlx_audio.tts.utils import load_model

model = load_model("OpenMOSS-Team/MOSS-TTS-Local-Transformer")
results = list(
    model.generate(
        text="Long text goes here...",
        preset="moss_tts_local",
        long_form=True,
        long_form_min_chars=160,
        long_form_target_chars=320,
        long_form_max_chars=520,
        long_form_prefix_audio_seconds=2.0,
        long_form_prefix_audio_max_tokens=25,
        long_form_retry_attempts=1,
        max_tokens=260,
    )
)

segment_metrics = [asdict(item) for item in model._last_long_form_segment_metrics]
boundary_metrics = [asdict(item) for item in model._last_long_form_boundary_metrics]
```

### Deterministic full-greedy (all channels)

```python
import mlx.core as mx

from mlx_audio.tts.utils import load_model

mx.random.seed(1234)
model = load_model("OpenMOSS-Team/MOSS-TTS-Local-Transformer")
all_greedy = [False] * int(model.config.channels)

results = list(
    model.generate(
        text="Full-greedy deterministic run.",
        preset="moss_tts_local",
        duration_s=6.0,
        do_samples=all_greedy,
        max_tokens=260,
    )
)
```

`full-greedy` is deterministic but can produce silent clips on some prompts/checkpoints.

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
| `do_samples`, `layers` | Per-channel deterministic/sampling overrides |
| `natural_stop_min_audio_rows` | Local natural-stop guard floor (`0` disables; default auto) |
| `long_form` (+ `long_form_*`) | Segmented long-form synthesis planner/continuity controls |

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
| Delay / Local | `text`, `input_type`, `instruct`, `ref_audio`, `ref_text`, `tokens`/`duration_s`, `natural_stop_min_audio_rows` |
| TTSD | `text` + `dialogue_speakers` schema |
| VoiceGenerator | `instruct` (+ optional `normalize_inputs`) |
| SoundEffect | `ambient_sound`, `sound_event`, optional `quality` |
| Realtime | use dedicated runtime docs: `../moss_tts_realtime/README.md` |

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
- Local natural stopping now includes an adaptive EOS guard derived from prompt length.
- Tune/override with `natural_stop_min_audio_rows` in `model_kwargs` (`0` disables the guard).
- Delay natural stopping can miss EOS on short prompts and run until `max_tokens`; for Delay checkpoints, set `tokens` or `duration_s` explicitly.
- For fixed output length, explicitly set `tokens` or `duration_s`.
- VoiceGenerator defaults to normalized prompt inputs unless overridden (`normalize_inputs=False`).
- CLI convenience: if you provide `--ref_audio` without `--ref_text`, the default `mlx_audio.tts.generate` flow will transcribe the reference audio using Whisper; provide `--ref_text` to avoid extra downloads/latency.
- SoundEffect can synthesize from `ambient_sound` even when `text` is omitted.
- The shared codec is mandatory at runtime; see codec docs below.

## Server Escape Hatch Safety (`model_kwargs`)

`/v1/audio/speech` supports `model_kwargs` for advanced controls that do not
already have first-class request fields.

Reserved keys are rejected with HTTP 400:

- `text`
- `input`
- `input_text`

Use top-level `input` for synthesis text instead of passing these keys inside
`model_kwargs`.

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
