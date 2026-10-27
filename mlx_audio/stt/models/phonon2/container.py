"""Read Fermion's five-value Parakeet containers without the Fermion runtime.

Container decoding and name mapping adapted from fermion-research 0.2.13,
Copyright 2026 Fermion Research, Apache-2.0.
"""

import json
import math
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

CONTAINER_FORMAT = "fermion-five-value-parakeet-v1"
_ATTENTION_NAMES = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_out",
    "relative_k_proj": "linear_pos",
    "bias_u": "pos_bias_u",
    "bias_v": "pos_bias_v",
}


def _mlx_name(name: str) -> str:
    name = name.replace("encoder.subsampling.layers.", "encoder.pre_encode.conv.")
    name = name.replace("encoder.subsampling.linear.", "encoder.pre_encode.out.")
    for source, target in _ATTENTION_NAMES.items():
        name = name.replace(f".self_attn.{source}", f".self_attn.{target}")
    name = name.replace(".conv.norm.", ".conv.batch_norm.")
    for source, target in (
        ("encoder_projector.", "joint.enc."),
        ("decoder.decoder_projector.", "joint.pred."),
        ("joint.head.", "joint.joint_net.2."),
        ("decoder.embedding.", "decoder.prediction.embed."),
    ):
        if name.startswith(source):
            return target + name[len(source) :]
    return name


def _five_value(blob: bytes, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    if len(shape) != 2 or shape[1] % 128:
        raise ValueError("Invalid Phonon five-value shape")
    outputs, inputs = shape
    trit_bytes = outputs * ((inputs + 4) // 5)
    if len(blob) < trit_bytes + 4 * outputs:
        raise ValueError("Truncated Phonon five-value record")
    packed = np.frombuffer(blob[:trit_bytes], dtype=np.uint8)
    if np.any(packed >= 243):
        raise ValueError("Invalid Phonon ternary codes")
    packed = packed.reshape(outputs, -1)
    codes = np.stack([(packed // 3**k) % 3 for k in range(5)], axis=-1)
    codes = codes.reshape(outputs, -1)[:, :inputs]
    nonzero = codes != 1
    count = int(nonzero.sum())
    mask_bytes = (count + 7) // 8
    offset = trit_bytes + mask_bytes
    if len(blob) != offset + 4 * outputs:
        raise ValueError("Invalid Phonon five-value record size")
    high = np.zeros(shape, dtype=bool)
    high[nonzero] = np.unpackbits(
        np.frombuffer(blob[trit_bytes:offset], dtype=np.uint8), bitorder="little"
    )[:count]

    def pack(plane):
        plane = plane.astype(np.uint32).reshape(outputs, inputs // 16, 16)
        return np.bitwise_or.reduce(
            plane << (2 * np.arange(16, dtype=np.uint32)), axis=-1
        )

    return {
        "base_q": pack(codes),
        "residual_q": pack(np.where(high, codes, 1)),
        "row_lo": np.frombuffer(blob[offset : offset + 2 * outputs], dtype=np.float16),
        "row_hi": np.frombuffer(blob[offset + 2 * outputs :], dtype=np.float16),
    }


def _intn(blob: bytes, shape: tuple[int, ...], bits: int) -> np.ndarray:
    outputs, total = shape[0], math.prod(shape)
    code_bytes = total if bits == 8 else ((total + 3) // 4) * 3
    if len(blob) != code_bytes + 2 * outputs:
        raise ValueError("Invalid Phonon integer table size")
    body = blob[:code_bytes]
    scales = np.frombuffer(blob[code_bytes:], dtype=np.float16).astype(np.float32)
    if bits == 8:
        codes = np.frombuffer(body, dtype=np.int8).astype(np.float32)
    elif bits == 6:
        triples = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        packed = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        codes = np.stack([(packed >> k) & 63 for k in (0, 6, 12, 18)], axis=-1)
        codes = codes.reshape(-1)[:total].astype(np.float32) - 32
    else:
        raise ValueError(f"Unsupported Phonon integer table: int{bits}")
    return (codes.reshape(outputs, -1) * scales[:, None]).reshape(shape)


def _dense_weights(tensors: dict[str, np.ndarray]) -> dict[str, mx.array]:
    weights = {}
    for name, value in tensors.items():
        if name.endswith("num_batches_tracked"):
            continue
        target = _mlx_name(name)
        lstm = re.fullmatch(r"decoder\.lstm\.(weight|bias)_(ih|hh)_l(\d+)", name)
        if lstm:
            kind, side, layer = lstm.groups()
            target = f"decoder.prediction.dec_rnn.lstm.{layer}."
            if kind == "weight":
                target += "Wx" if side == "ih" else "Wh"
            elif side == "hh":
                continue
            else:
                target += "bias"
                other = tensors[f"decoder.lstm.bias_hh_l{layer}"]
                value = (value.astype(np.float32) + other.astype(np.float32)).astype(
                    value.dtype
                )
        elif name.startswith("encoder.subsampling.layers.") and name.endswith(
            ".weight"
        ):
            value = value.transpose(0, 2, 3, 1)
        elif name.endswith(".conv.depthwise_conv.weight"):
            value = value.transpose(0, 2, 1)
        elif re.search(r"\.conv\.pointwise_conv[12]\.weight$", name):
            value = value[:, None, :] if value.ndim == 2 else value.transpose(0, 2, 1)
        if target in weights:
            raise ValueError(f"Duplicate Phonon tensor: {target}")
        # Match Fermion's inference dtype; palettes remain fp16 and codes uint32.
        weights[target] = mx.array(np.ascontiguousarray(value)).astype(mx.bfloat16)
    return weights


def read_container(path: Path) -> tuple[dict[str, mx.array], list[dict]]:
    """Stream records into MLX weights, retaining the exact five-value planes."""
    weights, tensors, modules, seen = {}, {}, [], set()
    with path.open("rb") as handle:
        header_bytes = int.from_bytes(handle.read(8), "little")
        if not 0 < header_bytes <= min(16 << 20, path.stat().st_size - 8):
            raise ValueError("Invalid Phonon container header size")
        header = json.loads(handle.read(header_bytes))
        if header.get("format") != CONTAINER_FORMAT:
            raise ValueError("Unsupported Phonon container format")
        for row in header["index"]:
            name, kind, shape, size = row["n"], row["k"], tuple(row["shape"]), row["b"]
            if (
                name in seen
                or (not shape and kind != "fp16")
                or any(type(d) is not int or d <= 0 for d in shape)
            ):
                raise ValueError("Duplicate Phonon record or invalid shape")
            seen.add(name)
            if (
                type(size) is not int
                or not 0 <= size <= path.stat().st_size - handle.tell()
            ):
                raise ValueError(f"Truncated Phonon container record: {name}")
            blob = handle.read(size)
            if kind == "five_value":
                target = _mlx_name(name)
                modules.append({"name": target, "shape": list(shape)})
                for suffix, value in _five_value(blob, shape).items():
                    weights[f"{target}.{suffix}"] = mx.array(value)
            elif kind in {"int6", "int8"}:
                tensors[name] = _intn(blob, shape, int(kind[3:]))
            elif kind == "fp16":
                if size != 2 * math.prod(shape):
                    raise ValueError(f"Invalid Phonon fp16 record size: {name}")
                tensors[name] = np.frombuffer(blob, dtype=np.float16).reshape(shape)
            else:
                raise ValueError(f"Unsupported Phonon record kind: {kind}")
        if handle.read(1):
            raise ValueError("Trailing bytes in Phonon container")
    weights.update(_dense_weights(tensors))
    return weights, modules
