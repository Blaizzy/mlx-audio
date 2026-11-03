"""Import NVIDIA ModelOpt FP8 and NVFP4 checkpoints without PyTorch.

FP8 tensor/channel scales are decoded into dense weights. NVFP4 weights stay
packed, with their FP8 block scales and FP32 global multiplier kept separately.
Activation and KV calibration scales are unused by the weight-only MLX runtime.
"""

import json
import logging
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)


def _e4m3_decode_table():
    values = []
    for byte in range(256):
        exponent, mantissa = (byte >> 3) & 15, byte & 7
        if exponent == 0:
            value = mantissa / 8 * 2**-6
        elif exponent == 15 and mantissa == 7:
            value = float("nan")
        else:
            value = (1 + mantissa / 8) * 2 ** (exponent - 7)
        values.append(-value if byte & 128 else value)
    return mx.array(values, dtype=mx.float32)


_E4M3_DECODE_LUT = _e4m3_decode_table()
_ALGORITHMS = {"FP8", "NVFP4", "NVFP4_AWQ", "W4A16_NVFP4", "MIXED_PRECISION"}
_METHODS = {"modelopt", "modelopt_mixed"}


def _dequantize_fp8_weight(weight, scale):
    if weight.dtype != mx.uint8 or weight.ndim not in (2, 3):
        raise ValueError("ModelOpt FP8 weights must be E4M3 byte matrices")
    if not mx.issubdtype(scale.dtype, mx.floating):
        raise ValueError("ModelOpt FP8 scales must be floating point")
    dtype = mx.bfloat16 if scale.dtype == mx.float32 else scale.dtype
    scale = scale.astype(mx.float32)
    if scale.size == 1:
        scale = scale.reshape(())
    elif scale.shape == weight.shape[:-1]:
        scale = scale[..., None]
    elif weight.ndim == 3 and scale.shape == weight.shape[:-2]:
        scale = scale[..., None, None]
    elif scale.shape != (*weight.shape[:-1], 1):
        raise ValueError(
            f"Invalid ModelOpt FP8 scale shape {scale.shape} for {weight.shape}"
        )
    return (_E4M3_DECODE_LUT[weight.astype(mx.uint32)] * scale).astype(dtype)


def transform_modelopt_weights(weights, quantization_config):
    """Consume source tensors and return MLX tensors plus native quantization.

    Conversions are evaluated in bounded batches and consumed FP8 bytes are
    released as we go, avoiding simultaneous residency of an entire FP8 source
    and its BF16 expansion. Native NVFP4 packing is a zero-copy reinterpretation.
    """
    if (
        not quantization_config
        or quantization_config.get("quant_method") not in _METHODS
    ):
        return weights, None
    algorithm = quantization_config.get("quant_algo")
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unsupported ModelOpt quantization algorithm: {algorithm}")
    nvfp4 = {
        k.removesuffix(".weight_scale_2")
        for k in weights
        if k.endswith(".weight_scale_2")
    }
    fp8 = {
        k.removesuffix(".weight_scale") for k in weights if k.endswith(".weight_scale")
    } - nvfp4
    if not nvfp4 and not fp8:
        raise ValueError("ModelOpt checkpoint has no weight scales")
    result, pending = {}, []
    pending_bytes = 0
    consumed = {f"{p}.weight" for p in nvfp4 | fp8} | {
        f"{p}.weight_scale" for p in nvfp4
    }
    for key in list(weights):
        if key not in weights:
            continue
        if key.endswith((".input_scale", ".input_scale_2", ".k_scale", ".v_scale")):
            weights.pop(key)
        elif key.endswith(".weight_scale_2"):
            prefix = key.removesuffix(".weight_scale_2")
            global_scale = weights.pop(key)
            weight = weights.pop(f"{prefix}.weight", None)
            scale = weights.pop(f"{prefix}.weight_scale", None)
            if weight is None or scale is None:
                raise ValueError(f"Missing ModelOpt NVFP4 tensors for {prefix}")
            if (
                weight.dtype != mx.uint8
                or scale.dtype != mx.uint8
                or weight.ndim not in (2, 3)
                or scale.ndim != weight.ndim
                or not mx.issubdtype(global_scale.dtype, mx.floating)
                or global_scale.size
                not in (1, weight.shape[0] if weight.ndim == 3 else 1)
            ):
                raise ValueError(f"Invalid ModelOpt NVFP4 tensors for {prefix}")
            if (
                weight.shape[:-1] != scale.shape[:-1]
                or weight.shape[-1] != 8 * scale.shape[-1]
            ):
                raise ValueError(f"Invalid ModelOpt NVFP4 scale shape for {prefix}")
            if algorithm == "NVFP4_AWQ":
                nan = (scale & 127) == 127
                count = mx.sum(nan).item()
                if count:
                    logger.warning(
                        "Zeroing %d NaN NVFP4 AWQ block scales for %s", count, prefix
                    )
                    scale = mx.where(nan, mx.array(0, mx.uint8), scale)
            result[f"{prefix}.weight"] = weight.view(mx.uint32)
            result[f"{prefix}.scales"] = scale
            result[key] = global_scale.astype(mx.float32).squeeze()
        elif key.endswith(".weight_scale") and key.removesuffix(".weight_scale") in fp8:
            prefix = key.removesuffix(".weight_scale")
            scale = weights.pop(key)
            weight = weights.pop(f"{prefix}.weight", None)
            if weight is None:
                raise ValueError(f"Missing ModelOpt FP8 weight for {prefix}")
            value = _dequantize_fp8_weight(weight, scale)
            result[f"{prefix}.weight"] = value
            pending.append(value)
            pending_bytes += value.nbytes
            if pending_bytes >= 64 * 1024**2:
                mx.eval(pending)
                pending.clear()
                pending_bytes = 0
        elif key not in consumed:
            result[key] = weights.pop(key)
    if pending:
        mx.eval(pending)
    native = {"group_size": 16, "bits": 4, "mode": "nvfp4"} if nvfp4 else None
    return result, native


def prepare_modelopt_weights(weights, config, model_path=None):
    """Normalize source config and weights before model-specific sanitization."""
    # A saved MLX checkpoint may still carry a source hf_quant_config sidecar.
    if "bits" in (config.get("quantization") or {}):
        return weights
    source = config.get("quantization_config")
    if source is None:
        for name in ("text_config", "llm_config"):
            source = (config.get(name) or {}).get("quantization_config")
            if source is not None:
                break
    if source is None and model_path is not None:
        path = Path(model_path) / "hf_quant_config.json"
        if path.is_file():
            sidecar = json.loads(path.read_text())
            if (sidecar.get("producer") or {}).get("name", "").lower() == "modelopt":
                if not any(k.endswith(".weight_scale") for k in weights):
                    return weights
                source = {"quant_method": "modelopt", **sidecar.get("quantization", {})}
    if not source or source.get("quant_method") not in _METHODS:
        return weights
    weights, native = transform_modelopt_weights(weights, source)
    config.pop("quantization_config", None)
    for name in ("text_config", "llm_config"):
        nested = config.get(name)
        if (
            isinstance(nested, dict)
            and (nested.get("quantization_config") or {}).get("quant_method")
            in _METHODS
        ):
            nested.pop("quantization_config")
    if native:
        config["quantization"] = native
        config["quantization_config"] = dict(native)
    else:
        config.pop("quantization", None)
    return weights
