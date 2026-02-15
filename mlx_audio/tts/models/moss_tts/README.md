# MOSS-TTS Family

MLX runtime support for the OpenMOSS MOSS-TTS family:

- Delay: `OpenMOSS-Team/MOSS-TTS`
- Local: `OpenMOSS-Team/MOSS-TTS-Local-Transformer`
- TTSD: `OpenMOSS-Team/MOSS-TTSD-v1.0`
- VoiceGenerator: `OpenMOSS-Team/MOSS-Voice-Generator`
- SoundEffect: `OpenMOSS-Team/MOSS-SoundEffect`
- Realtime: `OpenMOSS-Team/MOSS-TTS-Realtime`

## Presets

| Preset | Runtime | Defaults |
|---|---|---|
| `moss_tts` | Delay | `temperature=1.7`, `top_p=0.8`, `top_k=25`, `repetition_penalty=1.0` |
| `moss_tts_local` | Local | `temperature=1.0`, `top_p=0.95`, `top_k=50`, `repetition_penalty=1.1` |
| `ttsd` | TTSD | `temperature=1.1`, `top_p=0.9`, `top_k=50`, `repetition_penalty=1.1` |
| `voice_generator` | VoiceGenerator | `temperature=1.5`, `top_p=0.6`, `top_k=50`, `repetition_penalty=1.1` |
| `soundeffect` | SoundEffect | `temperature=1.5`, `top_p=0.6`, `top_k=50`, `repetition_penalty=1.2` |
| `realtime` | Realtime | `temperature=0.8`, `top_p=0.6`, `top_k=30`, `repetition_penalty=1.1` |

## CLI Examples

### Delay / Local text generation

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --text "Hello from MLX MOSS Local." \
  --preset moss_tts_local \
  --tokens 120 \
  --output_path ./outputs/moss_local
```

### Voice cloning

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS \
  --text "This is a cloned style sample." \
  --ref_audio REFERENCE/MOSS-Audio-Tokenizer/demo/demo_gt.wav \
  --ref_text "Demo reference transcript." \
  --preset moss_tts \
  --output_path ./outputs/moss_clone
```

### TTSD multi-speaker dialogue

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --text "[S1] Hi there. [S2] Good to meet you." \
  --dialogue_speakers_json PLANS/MOSS-TTS-PLANS/artifacts/phase4/real_model_smokes/ttsd/dialogue_speakers_demo.json \
  --preset ttsd \
  --output_path ./outputs/moss_ttsd
```

### SoundEffect generation

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-SoundEffect \
  --ambient_sound "Thunder and rain over a city street." \
  --sound_event storm \
  --preset soundeffect \
  --tokens 120 \
  --output_path ./outputs/moss_soundeffect
```

### Realtime one-shot generation

```bash
uv run python -m mlx_audio.tts.generate \
  --model OpenMOSS-Team/MOSS-TTS-Realtime \
  --text "Realtime confirmation clip." \
  --preset realtime \
  --stream \
  --output_path ./outputs/moss_realtime
```

## Phase 7 Parity Docs

- Multilingual smoke matrix:
  - `PLANS/MOSS-TTS-PLANS/artifacts/phase7/p7_multilingual_smoke_matrix.md`
- Effective-field matrix:
  - `PLANS/MOSS-TTS-PLANS/artifacts/phase7/p7_effective_field_matrix.md`
- Canonical IDs and alias table:
  - `PLANS/MOSS-TTS-PLANS/artifacts/phase7/p7_canonical_model_ids_and_aliases.md`
- `quality` taxonomy contract:
  - `PLANS/MOSS-TTS-PLANS/artifacts/phase7/p7_quality_taxonomy_contract.md`
- Schema-versioned request contract and watchlist:
  - `PLANS/MOSS-TTS-PLANS/artifacts/phase7/p7_schema_versioned_request_contract_and_watchlist.md`
