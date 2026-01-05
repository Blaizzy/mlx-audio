"""
Weight conversion script for CosyVoice3.

Converts PyTorch weights to MLX format.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import numpy as np
import torch


def convert_weight_normalized_conv(
    weights: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, mx.array]:
    """
    Convert weight-normalized conv weights.

    PyTorch stores weight_g (scale) and weight_v (direction) separately.
    We compute: weight = weight_g * weight_v / ||weight_v||
    """
    result = {}

    if f"{prefix}.parametrizations.weight.original0" in weights:
        # Weight normalized
        weight_g = weights[f"{prefix}.parametrizations.weight.original0"]
        weight_v = weights[f"{prefix}.parametrizations.weight.original1"]

        # Compute normalized weight
        norm = torch.norm(weight_v, dim=(1, 2), keepdim=True)
        weight = weight_g * weight_v / (norm + 1e-8)

        # Convert to MLX (transpose for Conv1d: PyTorch is [out, in, k], MLX is [out, k, in])
        weight = weight.permute(0, 2, 1).numpy()
        result["weight"] = mx.array(weight)
    else:
        # Regular weight
        if f"{prefix}.weight" in weights:
            weight = weights[f"{prefix}.weight"]
            weight = weight.permute(0, 2, 1).numpy()
            result["weight"] = mx.array(weight)

    if f"{prefix}.bias" in weights:
        result["bias"] = mx.array(weights[f"{prefix}.bias"].numpy())

    return result


def convert_linear(
    weights: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, mx.array]:
    """Convert linear layer weights."""
    result = {}
    if f"{prefix}.weight" in weights:
        result["weight"] = mx.array(weights[f"{prefix}.weight"].T.numpy())
    if f"{prefix}.bias" in weights:
        result["bias"] = mx.array(weights[f"{prefix}.bias"].numpy())
    return result


def convert_embedding(
    weights: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, mx.array]:
    """Convert embedding weights."""
    result = {}
    if f"{prefix}.weight" in weights:
        result["weight"] = mx.array(weights[f"{prefix}.weight"].numpy())
    return result


def convert_flow_weights(flow_pt: Dict[str, torch.Tensor]) -> Dict[str, mx.array]:
    """Convert flow module weights."""
    mlx_weights = {}

    # Input embedding
    if "input_embedding.weight" in flow_pt:
        mlx_weights["flow.input_embedding.weight"] = mx.array(
            flow_pt["input_embedding.weight"].numpy()
        )

    # Speaker embedding affine layer
    for k in ["spk_embed_affine_layer.weight", "spk_embed_affine_layer.bias"]:
        if k in flow_pt:
            new_key = f"flow.{k}"
            if "weight" in k:
                mlx_weights[new_key] = mx.array(flow_pt[k].T.numpy())
            else:
                mlx_weights[new_key] = mx.array(flow_pt[k].numpy())

    # Pre-lookahead layer
    for k, v in flow_pt.items():
        if k.startswith("pre_lookahead_layer."):
            new_key = k.replace("pre_lookahead_layer.", "flow.pre_lookahead_layer.")
            if "conv" in k and "weight" in k:
                # Conv1d weight: [out, in, k] -> [out, k, in]
                mlx_weights[new_key] = mx.array(v.permute(0, 2, 1).numpy())
            elif "bias" in k:
                mlx_weights[new_key] = mx.array(v.numpy())

    # DiT estimator
    for k, v in flow_pt.items():
        if k.startswith("decoder.estimator."):
            new_key = k.replace("decoder.estimator.", "flow.decoder.estimator.")

            if "conv_pos_embed.conv" in k:
                # Grouped conv weights
                if "weight" in k:
                    mlx_weights[new_key] = mx.array(v.permute(0, 2, 1).numpy())
                else:
                    mlx_weights[new_key] = mx.array(v.numpy())
            elif (
                "proj" in k
                or "linear" in k
                or "to_q" in k
                or "to_k" in k
                or "to_v" in k
                or "to_out" in k
            ):
                # Linear layers
                if "weight" in k:
                    mlx_weights[new_key] = mx.array(v.T.numpy())
                else:
                    mlx_weights[new_key] = mx.array(v.numpy())
            elif "time_mlp" in k:
                # Time embedding MLP
                if "weight" in k:
                    mlx_weights[new_key] = mx.array(v.T.numpy())
                else:
                    mlx_weights[new_key] = mx.array(v.numpy())
            elif "ff.ff" in k:
                # Feed-forward
                if "weight" in k:
                    mlx_weights[new_key] = mx.array(v.T.numpy())
                else:
                    mlx_weights[new_key] = mx.array(v.numpy())
            elif "rotary_embed" in k:
                mlx_weights[new_key] = mx.array(v.numpy())
            else:
                mlx_weights[new_key] = mx.array(v.numpy())

    return mlx_weights


def convert_hift_weights(hift_pt: Dict[str, torch.Tensor]) -> Dict[str, mx.array]:
    """Convert HIFT vocoder weights."""
    mlx_weights = {}

    for k, v in hift_pt.items():
        new_key = f"hift.{k}"

        # Handle weight normalization
        if ".parametrizations.weight.original0" in k:
            # Weight norm scale
            base_key = k.replace(".parametrizations.weight.original0", "")
            weight_g = v
            weight_v_key = k.replace("original0", "original1")
            weight_v = hift_pt[weight_v_key]

            # Compute normalized weight
            norm = torch.norm(weight_v, dim=(1, 2), keepdim=True)
            weight = weight_g * weight_v / (norm + 1e-8)

            # Convert for Conv1d
            new_key = f"hift.{base_key}.weight"
            mlx_weights[new_key] = mx.array(weight.permute(0, 2, 1).numpy())
        elif ".parametrizations.weight.original1" in k:
            # Skip, handled above
            continue
        elif "alpha" in k:
            # Snake activation alpha
            mlx_weights[new_key] = mx.array(v.numpy())
        elif "classifier.weight" in k or "l_linear.weight" in k:
            # Linear layers
            mlx_weights[new_key] = mx.array(v.T.numpy())
        elif ".bias" in k:
            mlx_weights[new_key] = mx.array(v.numpy())
        elif ".weight" in k:
            # Regular conv weights
            if len(v.shape) == 3:
                mlx_weights[new_key] = mx.array(v.permute(0, 2, 1).numpy())
            else:
                mlx_weights[new_key] = mx.array(v.numpy())

    return mlx_weights


def convert_llm_weights(llm_pt: Dict[str, torch.Tensor]) -> Dict[str, mx.array]:
    """Convert LLM weights."""
    mlx_weights = {}

    for k, v in llm_pt.items():
        # Map to new structure
        if k.startswith("llm.model."):
            # Qwen2 model weights
            new_key = k.replace("llm.model.", "")

            if "weight" in k and "norm" not in k and "embed" not in k:
                # Linear layers need transposing
                mlx_weights[new_key] = mx.array(v.T.numpy())
            else:
                mlx_weights[new_key] = mx.array(v.numpy())
        elif k == "speech_embedding.weight":
            mlx_weights[k] = mx.array(v.numpy())
        elif k == "llm_decoder.weight":
            mlx_weights[k] = mx.array(v.T.numpy())

    return mlx_weights


def convert_all_weights(
    flow_path: str,
    hift_path: str,
    llm_path: str = None,
    output_dir: str = "converted_weights",
) -> Dict[str, mx.array]:
    """
    Convert all CosyVoice3 weights to MLX format.

    Args:
        flow_path: Path to flow.pt
        hift_path: Path to hift.pt
        llm_path: Path to llm.pt (optional)
        output_dir: Output directory for converted weights

    Returns:
        Dictionary of converted weights
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_weights = {}

    # Convert Flow weights
    print("Converting Flow weights...")
    flow_pt = torch.load(flow_path, map_location="cpu", weights_only=True)
    flow_weights = convert_flow_weights(flow_pt)
    all_weights.update(flow_weights)
    print(f"  Converted {len(flow_weights)} flow weight tensors")

    # Convert HIFT weights
    print("Converting HIFT weights...")
    hift_pt = torch.load(hift_path, map_location="cpu", weights_only=True)
    hift_weights = convert_hift_weights(hift_pt)
    all_weights.update(hift_weights)
    print(f"  Converted {len(hift_weights)} hift weight tensors")

    # Convert LLM weights if provided
    if llm_path:
        print("Converting LLM weights...")
        llm_pt = torch.load(llm_path, map_location="cpu", weights_only=True)
        llm_weights = convert_llm_weights(llm_pt)
        all_weights.update(llm_weights)
        print(f"  Converted {len(llm_weights)} llm weight tensors")

    # Save weights
    output_path = output_dir / "weights.safetensors"
    mx.save_safetensors(str(output_path), all_weights)
    print(f"Saved converted weights to {output_path}")

    return all_weights


def main():
    parser = argparse.ArgumentParser(description="Convert CosyVoice3 weights to MLX")
    parser.add_argument(
        "--flow-path",
        type=str,
        default="cosyvoice3_weights/flow.pt",
        help="Path to flow.pt",
    )
    parser.add_argument(
        "--hift-path",
        type=str,
        default="cosyvoice3_weights/hift.pt",
        help="Path to hift.pt",
    )
    parser.add_argument(
        "--llm-path",
        type=str,
        default=None,
        help="Path to llm.pt (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="converted_weights",
        help="Output directory",
    )
    args = parser.parse_args()

    convert_all_weights(
        args.flow_path,
        args.hift_path,
        args.llm_path,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
