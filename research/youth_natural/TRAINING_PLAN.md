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
