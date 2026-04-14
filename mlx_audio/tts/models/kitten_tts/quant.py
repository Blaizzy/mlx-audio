import mlx.core as mx


def fake_quant_dynamic_u8(x: mx.array) -> mx.array:
    """Dynamic per-tensor uint8 fake-quantization (matches ONNX DynamicQuantizeLinear)."""
    x_f = x.astype(mx.float32)
    x_min = mx.minimum(mx.min(x_f), 0.0)
    x_max = mx.maximum(mx.max(x_f), 0.0)
    scale = (x_max - x_min) / 255.0

    # Avoid division by zero when input is constant.
    scale_safe = mx.where(scale == 0, 1.0, scale)
    zero_point = mx.round(-x_min / scale_safe)
    zero_point = mx.clip(zero_point, 0.0, 255.0)

    q = mx.round(x_f / scale_safe + zero_point)
    q = mx.clip(q, 0.0, 255.0)
    deq = (q - zero_point) * scale_safe

    return mx.where(scale == 0, mx.zeros_like(deq), deq)


def maybe_fake_quant(x: mx.array, enabled: bool) -> mx.array:
    return fake_quant_dynamic_u8(x) if enabled else x
