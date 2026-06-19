# Training Plan

The first topology is `YouthNaturalLoRA`, a reversible adapter explicitly enabled
by callers. It is never chosen by inferred age.

## Topology

- Frozen Studio base model.
- Rank-8 LoRA on selected `attention.wq`.
- Rank-8 LoRA on only the value slice of combined `attention.wkv`.
- Rank-8 LoRA on selected `attention.wo`.
- No router, expert, embedding, norm, output head, DAC, or speaker-encoder
  updates in the first pass.

## Gates

Real training requires:

- rights-checked dataset snapshot;
- memory probe and disk headroom;
- synthetic tiny overfit pass;
- zero-strength Studio parity;
- save/reload and resume parity;
- independent correctness review.

This branch includes a synthetic fixture path and truthful `not_run` receipts for
real model stages when those gates are not satisfied.

## Implemented Fixture Contracts

- `build_teacher_forced_batch` constructs prompt-plus-target rows so the first
  target frame is predicted from the final prompt row.
- Targets are delayed/sheared codebook rows matching ZONOS2 generation behavior.
- Prompt, padding, and invalid cells are masked with `INVALID_TARGET_ID`.
- `masked_cross_entropy` reports total loss, accuracy, entropy, valid token
  count, and per-codebook loss/accuracy on synthetic MLX arrays.
- Adapter-only tensors can be saved and reloaded as Safetensors for fixture
  parity without saving base weights.
