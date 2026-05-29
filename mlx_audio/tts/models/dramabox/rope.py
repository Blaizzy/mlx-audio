from __future__ import annotations

import math
from enum import Enum

import mlx.core as mx
import numpy as np


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"


def _as_rope_type(rope_type: LTXRopeType | str) -> LTXRopeType:
    return rope_type if isinstance(rope_type, LTXRopeType) else LTXRopeType(rope_type)


def apply_rotary_emb(
    input_tensor: mx.array,
    freqs_cis: tuple[mx.array, mx.array],
    rope_type: LTXRopeType | str = LTXRopeType.INTERLEAVED,
) -> mx.array:
    rope_type = _as_rope_type(rope_type)
    if rope_type is LTXRopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, *freqs_cis)
    if rope_type is LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, *freqs_cis)
    raise ValueError(f"Invalid rope type: {rope_type}")


def apply_interleaved_rotary_emb(
    input_tensor: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    original_shape = input_tensor.shape
    pairs = input_tensor.reshape(*original_shape[:-1], original_shape[-1] // 2, 2)
    first = pairs[..., 0]
    second = pairs[..., 1]
    rotated = mx.stack([-second, first], axis=-1).reshape(original_shape)
    return input_tensor * cos_freqs + rotated * sin_freqs


def apply_split_rotary_emb(
    input_tensor: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    needs_reshape = input_tensor.ndim != 4 and cos_freqs.ndim == 4
    if needs_reshape:
        batch, heads, seq_len, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(batch, seq_len, heads, -1).transpose(
            0, 2, 1, 3
        )

    split_input = input_tensor.reshape(*input_tensor.shape[:-1], 2, -1)
    first_half = split_input[..., :1, :]
    second_half = split_input[..., 1:, :]

    cos = cos_freqs[..., None, :]
    sin = sin_freqs[..., None, :]
    first_out = first_half * cos - second_half * sin
    second_out = second_half * cos + first_half * sin
    output = mx.concatenate([first_out, second_out], axis=-2).reshape(
        input_tensor.shape
    )

    if needs_reshape:
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    return output


def generate_freq_grid_np(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> mx.array:
    theta = positional_embedding_theta
    num_elements = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(1.0) / np.log(theta),
            np.log(theta) / np.log(theta),
            inner_dim // num_elements,
            dtype=np.float64,
        ),
    )
    return mx.array(pow_indices * math.pi / 2, dtype=mx.float32)


def generate_freq_grid(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> mx.array:
    theta = positional_embedding_theta
    num_elements = 2 * positional_embedding_max_pos_count
    indices = theta ** (
        mx.linspace(
            math.log(1.0, theta),
            math.log(theta, theta),
            inner_dim // num_elements,
            dtype=mx.float32,
        )
    )
    return indices.astype(mx.float32) * math.pi / 2


def get_fractional_positions(indices_grid: mx.array, max_pos: list[float]) -> mx.array:
    if indices_grid.shape[1] != len(max_pos):
        raise ValueError(
            f"Number of position dimensions ({indices_grid.shape[1]}) must "
            f"match max_pos length ({len(max_pos)})"
        )
    return mx.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(len(max_pos))],
        axis=-1,
    )


def generate_freqs(
    indices: mx.array,
    indices_grid: mx.array,
    max_pos: list[float],
    use_middle_indices_grid: bool,
) -> mx.array:
    if use_middle_indices_grid:
        if indices_grid.ndim != 4 or indices_grid.shape[-1] != 2:
            raise ValueError("Middle-index RoPE expects bounds shaped [B, D, T, 2]")
        indices_grid = (indices_grid[..., 0] + indices_grid[..., 1]) / 2.0
    elif indices_grid.ndim == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    freqs = indices * (fractional_positions[..., None] * 2 - 1)
    return freqs.transpose(0, 1, 3, 2).reshape(freqs.shape[0], freqs.shape[1], -1)


def split_freqs_cis(
    freqs: mx.array,
    pad_size: int,
    num_attention_heads: int,
) -> tuple[mx.array, mx.array]:
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)
    if pad_size != 0:
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    batch, seq_len, _ = cos_freq.shape
    cos_freq = cos_freq.reshape(batch, seq_len, num_attention_heads, -1).transpose(
        0, 2, 1, 3
    )
    sin_freq = sin_freq.reshape(batch, seq_len, num_attention_heads, -1).transpose(
        0, 2, 1, 3
    )
    return cos_freq, sin_freq


def interleaved_freqs_cis(
    freqs: mx.array,
    pad_size: int,
) -> tuple[mx.array, mx.array]:
    cos_freq = mx.repeat(mx.cos(freqs), repeats=2, axis=-1)
    sin_freq = mx.repeat(mx.sin(freqs), repeats=2, axis=-1)
    if pad_size != 0:
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)
    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: mx.array,
    dim: int,
    out_dtype=mx.float32,
    theta: float = 10000.0,
    max_pos: list[float] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType | str = LTXRopeType.INTERLEAVED,
    double_precision: bool = False,
) -> tuple[mx.array, mx.array]:
    if max_pos is None:
        max_pos = [20, 2048, 2048]
    rope_type = _as_rope_type(rope_type)
    generator = generate_freq_grid_np if double_precision else generate_freq_grid
    indices = generator(theta, indices_grid.shape[1], dim)
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    if rope_type is LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        pad_size = expected_freqs - freqs.shape[-1]
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        num_elements = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % num_elements)
    return cos_freq.astype(out_dtype), sin_freq.astype(out_dtype)
