# Original: ClearerVoice-Studio (github.com/modelscope/ClearerVoice-Studio)
# Copyright (c) Speech Lab, Alibaba Group
# Licensed under Apache License 2.0
# MLX port by Dmitry Starkov

"""
MLX implementation of compute_deltas function
Compatible with torchaudio.functional.compute_deltas
"""

import mlx.core as mx


def compute_deltas(
    specgram: mx.array, win_length: int = 5, mode: str = "edge"
) -> mx.array:
    """
    Compute delta coefficients of a tensor, usually a spectrogram.

    The formula is:
    d_t = sum_{n=1}^{N} n * (c_{t+n} - c_{t-n}) / (2 * sum_{n=1}^{N} n^2)

    where d_t is the deltas at time t, c_t is the spectrogram coefficients at time t,
    and N is (win_length-1)//2.

    Args:
        specgram: MLX array of dimension (..., freq, time)
        win_length: The window length used for computing delta (default: 5)
        mode: Padding mode - "edge" (replicate in PyTorch) or "constant" (default: "edge")

    Returns:
        MLX array of deltas of dimension (..., freq, time)
    """
    if win_length < 3:
        raise ValueError(
            f"Window length should be greater than or equal to 3. Found win_length {win_length}"
        )

    # Get dimensions
    original_shape = specgram.shape
    # Flatten all but last dimension
    specgram = specgram.reshape(-1, original_shape[-1])
    num_features = specgram.shape[0]

    n = (win_length - 1) // 2

    # Denominator: twice sum of integer squared
    # sum_{i=1}^{n} i^2 = n(n+1)(2n+1)/6
    # We need 2 * sum_{i=1}^{n} i^2 = n(n+1)(2n+1)/3
    denom = float(n * (n + 1) * (2 * n + 1)) / 3.0

    # Pad the specgram
    if mode == "edge":
        # Replicate padding - repeat edge values
        pad_left = mx.repeat(specgram[:, 0:1], n, axis=1)
        pad_right = mx.repeat(specgram[:, -1:], n, axis=1)
        padded = mx.concatenate([pad_left, specgram, pad_right], axis=1)
    else:
        # Constant padding with zeros
        padded = mx.pad(specgram, [(0, 0), (n, n)])

    # Create kernel weights: [-n, -n+1, ..., -1, 0, 1, ..., n-1, n]
    kernel_weights = mx.arange(-n, n + 1, dtype=padded.dtype)

    # Compute deltas using convolution-like operation
    # For each time step t, compute sum_{i=-n}^{n} i * padded[t+i]
    time_steps = padded.shape[1] - 2 * n
    output = mx.zeros((num_features, time_steps), dtype=padded.dtype)

    for i in range(time_steps):
        # Extract window for all features at once
        window = padded[:, i : i + win_length]
        # Compute weighted sum
        weighted = window * kernel_weights
        output[:, i] = mx.sum(weighted, axis=1) / denom

    # Reshape back to original shape
    output = output.reshape(original_shape)

    return output
