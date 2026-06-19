# YouthNaturalLoRA Subagent Execution Plan

Branch: `feature/zonos2-youth-natural-sft`

Lead workspace: `/Users/elishipley/VoiceForge/external/mlx-audio-zonos2`

The lead session owns orchestration, contracts, integration, final rights decisions,
final promotion decisions, and final evidence. Write-capable subtasks use
`agent/youth-natural/<task-id>-<slug>` branches or forked subagent workspaces with
disjoint file ownership.

## Wave 0: Read-Only Discovery

Run concurrently where possible:

- `yn-00-model-audit`: inspect pinned MLX ZONOS2 prompt layout, model topology,
  DAC/shear behavior, speaker conditioning, checkpoint loading, and sequence limits.
- `yn-00-rights-audit`: verify official dataset terms and create allowed/blocked
  lane decisions for the registry. No private data access.
- `yn-00-hardware-audit`: record Apple Silicon, macOS, MLX/Metal availability,
  memory/disk constraints, and safe concurrency recommendations.
- `yn-00-repo-conventions`: inspect packaging, tests, import paths, existing
  CLI conventions, and Git hygiene.
- `yn-00-test-gap-audit`: map requested tests to existing coverage and propose
  synthetic fixture strategy.

Wave 0 agents are read-only and produce handoffs under
`artifacts/youth_natural/orchestration/handoffs/`.

## Wave 1: Contracts And Ownership

The lead freezes:

- JSON schemas for handoffs, dataset items, adapter manifests, generation records;
- Python package entrypoint under `mlx_audio.research.zonos2_youth`;
- synthetic-fixture-only CI boundary;
- ownership map for research docs, orchestration artifacts, data pipeline,
  adapter/training core, evaluation harness, and reproducibility reports.

## Wave 2: Isolated Implementation

Write-capable tasks start only after ownership is clear:

- `yn-01-data-pipeline`: synthetic-first ingestion, rights-lane validation,
  split/duplicate/privacy guards.
- `yn-01-mlx-training`: LoRA mapping, teacher forcing, loss masking, checkpoint
  and manifest code. Owns adapter/training files only.
- `yn-01-eval-harness`: baseline/evaluation record schemas, anti-studio metrics,
  bandwidth audit fixtures. Does not modify training code.
- `yn-01-tests`: independent parity/adversarial tests against contracts.
- `yn-01-repro-report`: docs, one-command workflow verification, final report
  scaffolding.

## Wave 3: Resource-Aware Runs

Serialize large `mlx_metal_training` and full-model inference on this Mac unless
the memory probe explicitly clears a safe margin. CPU-light preprocessing, schema
tests, rights validation, and reports may continue in parallel.

## Wave 4: Independent Review

Required independent reviews:

- correctness/reproducibility review by a session that did not implement the MLX
  training core;
- rights/privacy/leakage review by a session that did not implement the dataset
  pipeline;
- metrics/promotion review before any automatic checkpoint promotion when
  measured runs exist.

## Wave 5: Integration And Evidence

The lead accepts candidate changes only after:

1. reading the durable handoff;
2. inspecting the diff;
3. checking ownership and privacy boundaries;
4. rerunning required tests in the integration workspace;
5. recording acceptance or rejection in `integration_log.jsonl`.

If subagent support is unavailable for a workstream, the same task contract is
preserved and the lead records the sequential fallback in `session_registry.jsonl`.
