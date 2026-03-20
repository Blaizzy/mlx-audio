"""
Weight loading utilities for DeepFilterNet models.

Handles conversion between PyTorch and MLX parameter naming conventions.
"""

import re
from typing import Dict, List, Set

import mlx.core as mx
from mlx.utils import tree_flatten


def _apply_gru_mapping(name: str) -> str:
    """Map PyTorch GRU parameter names to MLX GRU layer names."""
    gru_comp_map = {
        "weight_ih": "Wx",
        "weight_hh": "Wh",
        "bias_ih": "b",
        "bias_hh": "bhn",
    }

    m = re.search(r"\.gru\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)$", name)
    if not m:
        return name

    comp = gru_comp_map[m.group(1)]
    layer = m.group(2)
    return re.sub(
        r"\.gru\.(weight_ih|weight_hh|bias_ih|bias_hh)_l\d+$",
        f".gru_layers.{layer}.{comp}",
        name,
    )


def _apply_stride_conv_index_offset(name: str) -> str:
    """Map PyTorch Sequential indices to dict-based conv block indices."""
    stride_conv_patterns = [
        r"^enc\.erb_conv[123]\.",
        r"^enc\.df_conv1\.",
    ]

    if not any(re.match(pattern, name) for pattern in stride_conv_patterns):
        return name

    parts = name.split(".")
    if len(parts) >= 3 and parts[2].isdigit():
        parts[2] = str(int(parts[2]) + 1)
        return ".".join(parts)
    return name


def _sequential_layer_candidates(name: str) -> List[str]:
    """Return candidate names where PyTorch Sequential idx is represented as .layers.idx in MLX."""
    candidates = [name]
    replacements = {
        ".linear_in.0.": ".linear_in.layers.0.",
        ".linear_in.1.": ".linear_in.layers.1.",
        ".linear_out.0.": ".linear_out.layers.0.",
        ".linear_out.1.": ".linear_out.layers.1.",
        ".df_fc_emb.0.": ".df_fc_emb.layers.0.",
        ".df_fc_emb.1.": ".df_fc_emb.layers.1.",
        ".lsnr_fc.0.": ".lsnr_fc.layers.0.",
        ".lsnr_fc.1.": ".lsnr_fc.layers.1.",
        ".df_fc_a.0.": ".df_fc_a.layers.0.",
        ".df_fc_a.1.": ".df_fc_a.layers.1.",
        ".df_out.0.": ".df_out.layers.0.",
        ".df_out.1.": ".df_out.layers.1.",
        ".clc_fc_a.0.": ".clc_fc_a.layers.0.",
        ".clc_fc_a.1.": ".clc_fc_a.layers.1.",
        ".clc_fc_out.0.": ".clc_fc_out.layers.0.",
        ".clc_fc_out.1.": ".clc_fc_out.layers.1.",
        ".fc_emb.0.": ".fc_emb.layers.0.",
    }

    for src, dst in replacements.items():
        if src in name:
            candidates.append(name.replace(src, dst))
    return candidates


def get_weight_mapping(pt_names: Set[str], mlx_names: Set[str]) -> Dict[str, str]:
    """Build mapping from PyTorch weight names to MLX names.

    Returns:
        Dictionary mapping PyTorch names to MLX names
    """
    mapping = {}
    has_model_prefix = any(name.startswith("model.") for name in mlx_names)

    for pt_name in pt_names:
        if "num_batches_tracked" in pt_name or pt_name.endswith(".h0"):
            continue

        base = _apply_stride_conv_index_offset(_apply_gru_mapping(pt_name))
        # DF1 conv blocks expose pointwise conv as `pwconv`.
        base = base.replace(".1x1conv.", ".pwconv.")
        candidates: List[str] = []

        # Keep raw pt_name as a fallback because some names are already aligned.
        candidates.append(base)
        candidates.extend(_sequential_layer_candidates(base))
        if pt_name != base:
            candidates.append(pt_name)
            candidates.extend(_sequential_layer_candidates(pt_name))

        if has_model_prefix:
            candidates.extend([f"model.{c}" for c in list(candidates)])

        # Deduplicate while preserving first-match behavior.
        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            if cand in mlx_names:
                mapping[pt_name] = cand
                break

    return mapping


def set_weight(model, name: str, value: mx.array) -> bool:
    """Set a weight in the model by dot-separated path.

    Handles dict-based modules with string keys and integer indices for lists.

    Args:
        model: The model object
        name: Dot-separated path (e.g., 'enc.erb_conv0.1.weight')
        value: The weight value

    Returns:
        True if successful, False otherwise
    """
    parts = name.split(".")
    obj = model

    try:
        # Navigate to the parent object
        for part in parts[:-1]:
            # Try attribute first
            if hasattr(obj, part):
                obj = getattr(obj, part)
            # Try dict key
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            # Try integer index for lists/tuples
            elif part.isdigit():
                idx = int(part)
                if hasattr(obj, "__getitem__") and len(obj) > idx:
                    obj = obj[idx]
                else:
                    return False
            # Try string key access
            elif hasattr(obj, "__getitem__"):
                try:
                    obj = obj[part]
                except (KeyError, IndexError, TypeError):
                    return False
            else:
                return False

        # Set the final attribute
        final_part = parts[-1]

        # Special handling for GroupedLinearEinsum weight reshaping
        # PyTorch uses groups=8, MLX model uses groups=8
        # Both use weight shape (groups, ws_per_group, hs_per_group)
        # PyTorch: (8, 64, 32), MLX: (8, 64, 32)
        # No reshaping needed if groups match

        if hasattr(obj, final_part):
            # MLX nn.GRU uses bhn shape [H] (n-gate only), while PyTorch bias_hh is [3H].
            # Preserve PyTorch behavior by storing r/z bias_hh for later fold into b.
            if final_part == "bhn":
                current = getattr(obj, final_part)
                if (
                    hasattr(current, "shape")
                    and len(current.shape) == 1
                    and len(value.shape) == 1
                ):
                    if current.shape[0] * 3 == value.shape[0]:
                        h = current.shape[0]
                        rz = value[: 2 * h]
                        setattr(obj, "_pt_bhn_rz", rz)
                        setattr(obj, final_part, value[2 * h :])
                        setattr(obj, "_pt_bhn_folded", False)
                        return True
            if final_part == "b" and hasattr(obj, "_pt_bhn_rz"):
                rz = getattr(obj, "_pt_bhn_rz")
                h = value.shape[0] // 3
                rz_pad = mx.concatenate([rz, mx.zeros((h,), dtype=value.dtype)], axis=0)
                setattr(obj, final_part, value + rz_pad)
                setattr(obj, "_pt_bhn_folded", True)
                return True

            setattr(obj, final_part, value)
            return True
        elif isinstance(obj, dict) and final_part in obj:
            obj[final_part] = value
            return True
        elif final_part.isdigit():
            idx = int(final_part)
            if hasattr(obj, "__setitem__"):
                obj[idx] = value
                return True

    except (KeyError, AttributeError, TypeError, IndexError):
        pass

    return False


def load_weights(model, weights: Dict[str, mx.array]) -> int:
    """Load PyTorch weights into MLX model.

    Args:
        model: MLX DfNet model
        weights: Dictionary of weights from PyTorch checkpoint

    Returns:
        Number of weights loaded
    """
    # Get model parameter names
    param_tree = tree_flatten(model.parameters())
    mlx_names = {k for k, v in param_tree}
    pt_names = set(weights.keys())

    # Build mapping
    mapping = get_weight_mapping(pt_names, mlx_names)

    # Load weights
    loaded = 0
    for pt_name, mlx_name in mapping.items():
        value = weights[pt_name]
        if set_weight(model, mlx_name, value):
            loaded += 1

    # Finalize GRU bias folding for cases where bias_ih loaded before bias_hh.
    def _finalize_bias_fold(obj):
        if (
            hasattr(obj, "_pt_bhn_rz")
            and hasattr(obj, "b")
            and getattr(obj, "b") is not None
        ):
            if not getattr(obj, "_pt_bhn_folded", False):
                rz = getattr(obj, "_pt_bhn_rz")
                b = getattr(obj, "b")
                h = b.shape[0] // 3
                rz_pad = mx.concatenate([rz, mx.zeros((h,), dtype=b.dtype)], axis=0)
                setattr(obj, "b", b + rz_pad)
                setattr(obj, "_pt_bhn_folded", True)
        for v in getattr(obj, "__dict__", {}).values():
            if isinstance(v, list):
                for it in v:
                    _finalize_bias_fold(it)
            elif isinstance(v, dict):
                for it in v.values():
                    _finalize_bias_fold(it)
            elif hasattr(v, "__dict__"):
                _finalize_bias_fold(v)

    _finalize_bias_fold(model)

    # Load filterbanks - handle both wrapper and direct model cases
    if "erb_fb" in weights:
        if hasattr(model, "model") and hasattr(model.model, "erb_fb"):
            model.model.erb_fb = weights["erb_fb"]
        elif hasattr(model, "erb_fb"):
            model.erb_fb = weights["erb_fb"]

    if "mask.erb_inv_fb" in weights:
        if hasattr(model, "model"):
            # DeepFilterNet wrapper
            if hasattr(model.model, "erb_inv_fb"):
                model.model.erb_inv_fb = weights["mask.erb_inv_fb"]
            if hasattr(model.model, "mask"):
                model.model.mask.erb_inv_fb = weights["mask.erb_inv_fb"]
        elif hasattr(model, "mask"):
            # Direct DfNet
            model.mask.erb_inv_fb = weights["mask.erb_inv_fb"]

    return loaded
