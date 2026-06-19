# Evaluation Plan

Evaluation is automatic and deterministic. It compares matched prompts,
references, seeds, and sampling parameters across Studio, youth-stage adapters,
and natural-continuation adapters.

Required metrics include WER/CER, speaker similarity, F0 distribution, energy
dynamics, pause distribution, speaking-rate drift, effective bandwidth,
codebook entropy/repetition/EOS behavior, MoE occupancy, peak memory, and
generation speed.

The anti-studio score only rewards movement toward held-out conversational
speech in pause, F0, energy, and rate-variability distances. Noise, clipping,
reverb, and bandwidth loss are penalties, not positive naturalness signals.
