"""Latent normalization + sampling helper for the dots.tts MLX runtime.

Imports ONLY mlx / numpy — never torch. Ports ``dots_tts.models.dots_tts.core.IOHelper``:

  * ``normalize(x)``    = ``(x - mean) / sqrt(var)``   (x is ``[B, T, 128]``)
  * ``denormalize(x)``  = ``x * sqrt(var) + mean``
  * ``sample_from_latent(latent, noise)`` : ``latent`` is ``[B, 256, T]`` (the
    mean/log_std pair stacked on the channel axis); returns
    ``z = mean + noise * exp(log_std)`` transposed to ``[B, T, 128]``.

``mean`` / ``var`` are per-channel ``[128]`` stats loaded from ``latent_stats.npz``
(keys ``mean`` / ``var``). They broadcast over the last (channel) dim of the
normalize/denormalize inputs.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


class IOHelper:
    """Per-channel latent normalize / denormalize / sample (mirrors torch IOHelper)."""

    def __init__(self, latent_stats_path: str | Path | None = None):
        if latent_stats_path is not None:
            stats = mx.load(str(latent_stats_path))
            if "mean" not in stats or "var" not in stats:
                raise ValueError(f"latent_stats missing mean/var keys: {sorted(stats)}")
            # fp32 throughout: parity oracle is fp32 and the stats are tiny.
            self.global_mean = stats["mean"].astype(mx.float32)
            self.global_var = stats["var"].astype(mx.float32)
        else:
            self.global_mean = None
            self.global_var = None

    def normalize(self, x: mx.array) -> mx.array:
        """``(x - mean) / sqrt(var)`` over the last dim; identity if stats absent."""
        if self.global_mean is not None and self.global_var is not None:
            x = (x - self.global_mean) / mx.sqrt(self.global_var)
        return x

    def denormalize(self, x: mx.array) -> mx.array:
        """``x * sqrt(var) + mean`` over the last dim; identity if stats absent."""
        if self.global_mean is not None and self.global_var is not None:
            x = x * mx.sqrt(self.global_var) + self.global_mean
        return x

    @staticmethod
    def sample_from_latent(latent: mx.array, noise: mx.array | None = None) -> mx.array:
        """Sample ``z`` from a ``[B, 256, T]`` latent and return it as ``[B, T, 128]``.

        ``mean, log_std = split(latent, 2, axis=1)`` (channel); ``z = mean +
        noise * exp(log_std)``; then transpose ``[B, 128, T] -> [B, T, 128]``.
        ``noise`` is injectable for parity (default = a fresh standard-normal draw
        the shape of ``mean``).
        """
        mean, log_std = mx.split(latent, 2, axis=1)
        if noise is None:
            noise = mx.random.normal(mean.shape)
        z = mean + noise * mx.exp(log_std)
        return z.transpose(0, 2, 1)
