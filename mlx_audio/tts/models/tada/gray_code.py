import math

import mlx.core as mx


def int_to_gray_code(n: int, num_bits: int) -> list:
    gray = n ^ (n >> 1)
    bits = []
    for i in range(num_bits - 1, -1, -1):
        bits.append(1.0 if (gray >> i) & 1 else -1.0)
    return bits


def gray_code_to_int(bits: list) -> int:
    gray = 0
    for b in bits:
        gray = (gray << 1) | (1 if b > 0 else 0)
    n = gray
    mask = n >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


def encode_time_with_gray_code(
    time_before: mx.array, time_after: mx.array, num_bits: int
) -> mx.array:
    batch_size = time_before.shape[0]
    result = mx.zeros((batch_size, 2 * num_bits))

    for b in range(batch_size):
        tb = int(time_before[b].item())
        ta = int(time_after[b].item())
        tb_bits = int_to_gray_code(tb, num_bits)
        ta_bits = int_to_gray_code(ta, num_bits)
        result[b, :num_bits] = mx.array(tb_bits)
        result[b, num_bits:] = mx.array(ta_bits)

    return result


def decode_gray_code_to_time(gray_code_bits: mx.array, num_bits: int) -> mx.array:
    if gray_code_bits.ndim == 1:
        bits = gray_code_bits.tolist()
        return mx.array(gray_code_to_int(bits))

    batch_size = gray_code_bits.shape[0]
    results = []
    for b in range(batch_size):
        bits = gray_code_bits[b].tolist()
        results.append(gray_code_to_int(bits))
    return mx.array(results)
