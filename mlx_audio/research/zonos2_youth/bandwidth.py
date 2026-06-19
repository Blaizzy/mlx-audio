from __future__ import annotations

import numpy as np


def estimate_effective_bandwidth(
    audio: np.ndarray,
    sample_rate: int,
    *,
    energy_floor_db: float = -45.0,
) -> float:
    """Estimate useful bandwidth from a mono waveform without trusting sample rate.

    This is a lightweight fixture-friendly estimator. Production runs should keep
    the full spectral report, but tests only need deterministic behavior.
    """

    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    if samples.size == 0 or sample_rate <= 0:
        return 0.0
    windowed = samples * np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    if not np.any(spectrum):
        return 0.0
    db = 20.0 * np.log10(spectrum / np.max(spectrum) + 1e-12)
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    active = freqs[db >= energy_floor_db]
    return float(active[-1]) if active.size else 0.0


def bandwidth_tier(bandwidth_hz: float) -> str:
    if bandwidth_hz <= 0:
        return "unknown"
    if bandwidth_hz < 4000:
        return "narrowband_low"
    if bandwidth_hz < 8000:
        return "narrowband_16k"
    if bandwidth_hz < 14000:
        return "wideband_mid"
    return "fullband"


def codebook_policy_for_tier(tier: str) -> dict[str, object]:
    if tier in {"narrowband_low", "narrowband_16k"}:
        return {
            "policy": "all_codebooks_with_anchor_kl",
            "ce_weight": [1.0] * 9,
            "kl_weight": [1.0] * 9,
            "notes": "Do not downweight codebooks without a pinned DAC contribution study.",
        }
    return {
        "policy": "normal_ce",
        "ce_weight": [1.0] * 9,
        "kl_weight": [0.0] * 9,
        "notes": "Full-band or unknown-safe default.",
    }

