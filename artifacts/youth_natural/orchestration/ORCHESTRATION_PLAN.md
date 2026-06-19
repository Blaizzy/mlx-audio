# YouthNaturalLoRA Orchestration Plan

## Scope

Build a supervised-only, reversible ZONOS2 adapter system named
`YouthNaturalLoRA` on the existing MLX ZONOS2 path. This branch must keep Studio
unchanged, use explicit adapter enablement only, preserve rights-lane isolation,
and never train on or publish raw minor audio from Git-tracked paths.

## Initial Source Contract

- Active implementation path: `mlx_audio/tts/models/zonos2/`.
- Base model type: `zonos2`.
- Default codebooks: nine DAC codebooks plus one text column per frame.
- DAC sample rate: 44.1 kHz with 512 samples per audio frame in the local code.
- Prompt builder emits optional speaker marker rows, byte-text rows, then a
  short sheared silence prefix.
- Speaker conditioning replaces the speaker slot hidden state after
  `multi_embedder` and before `emb_norm`.
- Attention owns `wq`, combined `wkv`, `wo`, gating, and RoPE. Runtime splits
  `wkv` into key and value halves; LoRA may target only the value half.
- MoE routers, experts, embeddings, norms, output heads, DAC, and speaker
  encoder are frozen for the first adapter topology.

## Workstreams

| task_id | resource_class | owner type | mutable paths | dependencies | integration tests |
|---|---|---|---|---|---|
| yn-00-model-audit | read_only_analysis | explorer | none | none | source manifest, prompt/model audit |
| yn-00-rights-audit | io_or_network | explorer | none | none | rights report, blocked/allowed evidence |
| yn-00-hardware-audit | read_only_analysis | explorer | none | none | environment manifest, memory policy |
| yn-00-repo-conventions | read_only_analysis | explorer | none | none | packaging/test command notes |
| yn-00-test-gap-audit | read_only_analysis | explorer | none | none | test matrix |
| yn-01-data-pipeline | cpu_light | worker | `mlx_audio/research/zonos2_youth/data*`, `research/specs/youth_dataset_item.schema.json`, data tests | Wave 1 contracts | schema, split, duplicate, privacy tests |
| yn-01-mlx-training | cpu_light, mlx_metal_training | worker | `mlx_audio/research/zonos2_youth/adapter*`, `train*`, `export*`, adapter tests | model audit | LoRA parity, gradient, save/reload tests |
| yn-01-eval-harness | cpu_light | worker | `mlx_audio/research/zonos2_youth/eval*`, `metrics*`, eval tests | schemas | generation record, anti-studio, bandwidth tests |
| yn-01-tests | independent_review | worker | `tests/research/*` only | data/training contracts | adversarial fixture tests |
| yn-01-repro-report | cpu_light | worker | `research/youth_natural/*`, reports | Wave 1 contracts | docs and command smoke tests |

## Concurrency Rules

- Read-only agents may inspect the whole repository.
- Write-capable agents must use isolated branches or forked workspaces and may
  modify only assigned paths.
- Only the lead writes `task_ledger.jsonl`, `session_registry.jsonl`,
  `file_ownership.json`, and `integration_log.jsonl` in the integration branch.
- Raw datasets, waveform caches, speaker embeddings, and model checkpoints stay
  outside Git-tracked paths unless they are synthetic test fixtures.
- Full-model MLX inference/training is serialized on this Mac until the memory
  probe proves otherwise.

## Integration Tests By Gate

- Orchestration: schema validation, resource class validation, dependency graph,
  no active write ownership overlap, every accepted commit has a log entry.
- Data: rights-lane isolation, Common Voice metadata filtering, speaker/session
  split isolation, duplicate detection, age metadata preservation, no raw audio.
- Model: prompt/target layout, shear/unshear, zero LoRA initialization,
  value-slice-only `wkv`, strength-zero parity, gradient filtering,
  save/reload/resume, manifest validation.
- Evaluation: deterministic generation records, bandwidth estimation,
  anti-studio metrics that penalize noise/bandwidth loss, future-RL records.
- Reproducibility: one-command CLI smoke tests on synthetic fixtures, immutable
  run/config IDs, truthful `blocked`, `partial`, and `not_run` receipts.
