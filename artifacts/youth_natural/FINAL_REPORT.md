# YouthNaturalLoRA Final Report

## Status

Branch: `feature/zonos2-youth-natural-sft`

Source/evidence commit: `8b554858e9ea57e68cded2eb57de9a41119bd2b7`

Result: complete and tested synthetic-first training/data/evaluation scaffold with truthful `not_run` receipts. No real youth-data SFT run was executed, no real YouthNaturalLoRA checkpoint was promoted, and no raw audio or speaker embeddings are included in Git.

## Implemented And Tested

- Orchestration artifacts, task ledger, session registry, file ownership map, handoff schema, and integration log under `artifacts/youth_natural/orchestration/`.
- Research policy docs under `research/youth_natural/` for dataset rights, training, bandwidth, evaluation, and future RL handoff.
- JSON schema emit/validation for handoffs, dataset items, adapter manifests, and generation records.
- Dataset utilities for Common Voice metadata filtering, provided age-band preservation, speaker/session split isolation, duplicate detection, transcript preservation, and reference/target separation on synthetic records.
- Target utilities for prompt-plus-target teacher-forcing batches and masked nine-codebook CE metrics on synthetic token fixtures.
- Portable YouthNaturalLoRA manifest and synthetic adapter Safetensors helpers, including exact-zero initialization, strength-zero behavior, and value-slice-only combined `wkv` guards.
- Bandwidth tiering and conservative narrowband policy scaffolding. The DAC contribution experiment is recorded as not run.
- Evaluation record schema, future preference-ready fields, and transparent anti-studio metric scaffolding that penalizes bandwidth loss.
- CLI commands for `audit`, `prepare-data`, `overfit`, `train`, `evaluate`, `export-adapter`, and `test-merge`.

## Verification

Commands run and passing:

```bash
<venv>/bin/python -m unittest discover mlx_audio/research/zonos2_youth/tests
<venv>/bin/python -m unittest mlx_audio.tts.tests.test_zonos2
<venv>/bin/python -m mlx_audio.research.zonos2_youth validate-orchestration
rg -n "\\/Users\\/|\\/var\\/folders" artifacts/youth_natural research/youth_natural research/configs || true
git diff --check
```

Observed results:

- YouthNaturalLoRA synthetic tests: 34 passed.
- Existing ZONOS2 tests: 20 passed.
- Orchestration validation: `status=ok`, 7 persisted handoffs.
- Absolute local path scan: no matches in tracked YouthNaturalLoRA docs/config/artifacts.
- Whitespace check: clean for the working diff.

## Real MLX Results

No real MLX ZONOS2 training, full-model deterministic baseline generation, decoded audio evaluation, merge parity, or promoted checkpoint exists from this run.

The branch records this through:

- `artifacts/youth_natural/training_runs/prepare-data-not-run/receipt.json`
- `artifacts/youth_natural/training_runs/youth-not-run/receipt.json`
- `artifacts/youth_natural/training_runs/natural-not-run/receipt.json`
- `artifacts/youth_natural/training_runs/evaluate-not-run/receipt.json`
- `artifacts/youth_natural/training_runs/export-adapter-not-run/receipt.json`
- `artifacts/youth_natural/training_runs/test-merge-not-run/receipt.json`

## Synthetic And Mock Coverage

Synthetic tests cover schemas, orchestration validity, Common Voice local filtering, rights-lane isolation, split leakage prevention, duplicate detection, age metadata preservation without inference, transcript preservation, reference/target separation, target masks, masked CE metrics, bandwidth estimation, adapter manifest validation, zero LoRA behavior, value-slice-only `wkv`, save/reload, future generation records, and private-artifact guards.

The synthetic overfit command writes a synthetic adapter manifest only. It is not evidence of model trainability on real ZONOS2 weights.

## Dataset And Rights State

Datasets actually acquired: none.

Permissive lane records:

- Mozilla Common Voice Scripted Speech English, pinned in the rights report as `v26.0 / cv-corpus-26.0-2026-06-12`.
- Mozilla Common Voice Spontaneous Speech English, pinned as `v4.0 / sps-corpus-4.0-2026-06-12`.

Research-only or blocked records:

- MyST Children's Conversational Speech: research/noncommercial lane, not acquired, blocked for release without license acceptance and review.
- Expresso: research/noncommercial lane, not acquired, blocked for release-lane adapters.
- CSLU Kids Speech and CMU Kids Corpus: separately licensed lane, not acquired, blocked until an executed agreement permits the intended use.

No final legal decision is asserted by this branch. The rights report is a provenance and gating artifact, not legal advice.

## Training Stages

Youth-stage effect: not measured.

Natural-continuation effect: not measured.

Reason: no rights-checked local dataset snapshot and no real model/DAC training run were available in this branch run. The CLI refuses to report success and writes `not_run` receipts instead.

## Metrics And Limitations

Automatic metric production was scaffolded but not run against real generated audio. No WER, CER, speaker similarity, F0, energy, pause, bandwidth, MoE, or free-running regression numbers are claimed.

Known evaluation limitation: child and teen speech metrics can be biased by ASR and speaker encoders trained mostly on adult or studio-quality speech. Future promotion must report worst-speaker and worst-window failures, not only averages.

## Private Artifacts Excluded

Git ignore rules exclude YouthNaturalLoRA raw media, generated waveform caches, NumPy/NPZ arrays, and embedding artifacts under `artifacts/youth_natural/`. Dataset paths, raw minor audio, speaker embeddings, access tokens, and identifying metadata were not placed in subagent prompts or tracked artifacts.

## Reproduction Commands

```bash
python -m mlx_audio.research.zonos2_youth audit
python -m mlx_audio.research.zonos2_youth prepare-data --config research/configs/youth_natural_data.yaml
python -m mlx_audio.research.zonos2_youth overfit --config research/configs/youth_natural_train.yaml
python -m mlx_audio.research.zonos2_youth train --stage youth --config research/configs/youth_natural_train.yaml
python -m mlx_audio.research.zonos2_youth train --stage natural --config research/configs/youth_natural_train.yaml
python -m mlx_audio.research.zonos2_youth evaluate --config research/configs/youth_natural_eval.yaml
python -m mlx_audio.research.zonos2_youth export-adapter <checkpoint>
python -m mlx_audio.research.zonos2_youth test-merge <adapter>
```

Without a rights-checked dataset snapshot and serialized model resource gate, these commands intentionally emit synthetic or `not_run` artifacts.

## Checkpoints And Hashes

- Real adapter checkpoints: none.
- Youth-stage checkpoint: none.
- Natural-stage checkpoint: none.
- Synthetic adapter manifest: `artifacts/youth_natural/training_runs/synthetic-overfit/adapter_manifest.json`.
- Dataset snapshot hashes: none, because no dataset was acquired.
- Source manifest: `artifacts/youth_natural/source_manifest.json`.

## Subagents And Review

Wave 0 read-only subagents:

- `yn-00-model-audit`, Peirce: model/source audit handoff persisted.
- `yn-00-rights-audit`, Noether: rights audit handoff persisted.
- `yn-00-hardware-audit`, Raman: hardware/resource audit handoff persisted.
- `yn-00-repo-conventions`, Singer: repository conventions handoff persisted.
- `yn-00-test-gap-audit`, Beauvoir: test-gap handoff persisted.

Independent review subagents:

- `yn-02-correctness-repro-review`, Einstein: requested fixes for stale manifests, chunked LoRA guards, CLI output roots, and whitespace.
- `yn-02-rights-privacy-review`, Copernicus: requested fixes for absolute local paths, separately licensed dataset records, restricted-lane release validation, and broader media/embedding ignore rules.

Resolution:

- Review repairs landed in `320f14aaa62e67c2c0f3d8ffef9e522fcbca1bf5`.
- Audit provenance cleanup landed in `8b554858e9ea57e68cded2eb57de9a41119bd2b7`.
- The lead session inspected the findings, patched the integration branch directly, reran tests, and recorded the decisions in `integration_log.jsonl`.

## Delegation Limits

Native subagents were available and used for read-only audit and independent review. Additional write-capable worktrees were not used for the synthetic scaffold because the available subagent pool was consumed by audit/review, the high-risk implementation was intentionally kept small, and no real dataset or model run was available to justify parallel write sessions. This is recorded as a capability and dependency limit, not hidden as completed training.

## Conflicts And Local State

The branch was created from `4ee95391967e6ff802970a300d68c67bd8809158`. Pre-existing local edits in `mlx_audio/tts/generate.py` and `mlx_audio/tts/models/zonos2/model.py` were preserved and not reverted. YouthNaturalLoRA commits do not stage those unrelated runtime edits.

## Ready For Later Preference Learning

The branch is ready to carry future generation records with prompt hashes, voice profile IDs, rights lanes, adapter strength, sampling parameters, seeds, transcript/alignment fields, metric payloads, and future preference eligibility. It does not invent chosen/rejected pairs or preference labels.

## Remaining GGUF Work

No GGUF, GGML, quantization export, or custom runtime work was implemented. Later GGUF work should start from a real exported adapter, perform MLX adapter-on versus merged-weight parity, then only proceed to runtime conversion after the floating-point lineage is proven.
