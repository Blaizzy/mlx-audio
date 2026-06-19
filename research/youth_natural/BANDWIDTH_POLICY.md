# Bandwidth Policy

Child and teen speech corpora may be narrowband. The pipeline estimates effective
bandwidth from the waveform instead of trusting container sample rate.

Until a pinned DAC codebook contribution study proves otherwise, narrowband
examples use normal CE on all codebooks plus stronger full-band anchors and KL
regularization. The branch does not hard-code a claim that late codebooks equal
treble.

Narrowband data may not dominate the natural-continuation stage, and decoded
full-band evaluation must show no material bandwidth cutoff before promotion.
