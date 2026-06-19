# Future RL Handoff

This branch does not implement DPO, IPO, GRPO, PPO, reward-model training, or
online preference updates.

For future preference learning, generation records contain prompt hashes,
reference hashes, rights lane, base/adapter hashes, sampling parameters, seed,
checkpoint stage, automatic metrics, local audio path, and an explicit
`future_preference_eligible` flag. The export contains no invented chosen or
rejected pairs.

Restricted or private youth-speaker samples must remain local unless the
governing license and consent record allow publication.
