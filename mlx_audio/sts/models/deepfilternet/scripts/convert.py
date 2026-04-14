#!/usr/bin/env python3
"""
Convert DeepFilterNet PyTorch weights to MLX format.

This script converts pretrained DeepFilterNet models from the original
PyTorch implementation to MLX-compatible format with proper weight mapping.
"""

import argparse
import configparser
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import numpy as np
import torch


def convert_weight(weight: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array."""
    return mx.array(weight.detach().cpu().numpy())


def parse_config(config_path: Path) -> Dict[str, Any]:
    """Parse DeepFilterNet config.ini file."""
    config = configparser.ConfigParser()
    config.read(config_path)

    linear_groups = config.getint("deepfilternet", "linear_groups", fallback=16)
    df_order = config.getint(
        "df",
        "df_order",
        fallback=config.getint("deepfilternet", "df_order", fallback=5),
    )
    df_lookahead = config.getint(
        "df",
        "df_lookahead",
        fallback=config.getint("deepfilternet", "df_lookahead", fallback=0),
    )

    result = {
        # [df] section
        "sample_rate": config.getint("df", "sr", fallback=48000),
        "fft_size": config.getint("df", "fft_size", fallback=960),
        "hop_size": config.getint("df", "hop_size", fallback=480),
        "nb_erb": config.getint("df", "nb_erb", fallback=32),
        "nb_df": config.getint("df", "nb_df", fallback=96),
        "df_order": df_order,
        "df_lookahead": df_lookahead,
        "lsnr_max": config.getint("df", "lsnr_max", fallback=35),
        "lsnr_min": config.getint("df", "lsnr_min", fallback=-15),
        # [deepfilternet] section
        "conv_ch": config.getint("deepfilternet", "conv_ch", fallback=64),
        "conv_k_enc": config.getint("deepfilternet", "conv_k_enc", fallback=1),
        "conv_k_dec": config.getint("deepfilternet", "conv_k_dec", fallback=1),
        "conv_width_factor": config.getint(
            "deepfilternet", "conv_width_factor", fallback=1
        ),
        "conv_dec_mode": config.get(
            "deepfilternet", "conv_dec_mode", fallback="transposed"
        ),
        "emb_hidden_dim": config.getint(
            "deepfilternet", "emb_hidden_dim", fallback=256
        ),
        "emb_num_layers": config.getint("deepfilternet", "emb_num_layers", fallback=3),
        "df_hidden_dim": config.getint("deepfilternet", "df_hidden_dim", fallback=256),
        "df_num_layers": config.getint("deepfilternet", "df_num_layers", fallback=2),
        "gru_groups": config.getint("deepfilternet", "gru_groups", fallback=8),
        "linear_groups": linear_groups,
        # DeepFilterNet2 configs do not expose enc_linear_groups separately; in that case it
        # should follow linear_groups to keep grouped-linear tensor shapes aligned.
        "enc_linear_groups": config.getint(
            "deepfilternet", "enc_linear_groups", fallback=linear_groups
        ),
        "group_shuffle": config.getboolean(
            "deepfilternet", "group_shuffle", fallback=False
        ),
        "mask_pf": config.getboolean("deepfilternet", "mask_pf", fallback=False),
        "conv_lookahead": config.getint("deepfilternet", "conv_lookahead", fallback=2),
        "conv_depthwise": config.getboolean(
            "deepfilternet", "conv_depthwise", fallback=True
        ),
        "convt_depthwise": config.getboolean(
            "deepfilternet", "convt_depthwise", fallback=False
        ),
        "enc_concat": config.getboolean("deepfilternet", "enc_concat", fallback=False),
        "emb_gru_skip_enc": config.get(
            "deepfilternet", "emb_gru_skip_enc", fallback="none"
        ),
        "emb_gru_skip": config.get("deepfilternet", "emb_gru_skip", fallback="none"),
        "df_gru_skip": config.get(
            "deepfilternet", "df_gru_skip", fallback="groupedlinear"
        ),
        "dfop_method": config.get(
            "deepfilternet", "dfop_method", fallback="real_unfold"
        ),
    }

    # Parse conv_kernel strings
    conv_kernel = config.get("deepfilternet", "conv_kernel", fallback="1,3")
    result["conv_kernel"] = [int(x) for x in conv_kernel.split(",")]

    convt_kernel = config.get("deepfilternet", "convt_kernel", fallback="1,3")
    result["convt_kernel"] = [int(x) for x in convt_kernel.split(",")]

    conv_kernel_inp = config.get("deepfilternet", "conv_kernel_inp", fallback="3,3")
    result["conv_kernel_inp"] = [int(x) for x in conv_kernel_inp.split(",")]

    return result


def convert_pytorch_to_mlx(
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    model_name: str = "DeepFilterNet3",
):
    """Convert PyTorch checkpoint to MLX format with proper weight mapping."""

    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get state dict
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    print(f"Found {len(state_dict)} parameters in checkpoint")

    # Parse config
    print(f"Parsing config from {config_path}")
    config_dict = parse_config(config_path)
    config_dict["model_version"] = model_name

    # Print weight shapes for debugging
    print("\nPyTorch weight shapes:")
    for key, value in list(state_dict.items())[:20]:
        print(f"  {key}: {tuple(value.shape)}")
    print("  ...")

    # Convert weights - direct mapping since we'll match the architecture
    print("\nConverting weights to MLX format...")
    mlx_weights = {}

    for key, value in state_dict.items():
        # Skip buffers that aren't needed for inference
        if "num_batches_tracked" in key:
            continue

        # Convert weight
        mlx_array = convert_weight(value)
        mlx_weights[key] = mlx_array

    print(f"Converted {len(mlx_weights)} weights")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = output_dir / "model.safetensors"
    print(f"Saving weights to {weights_path}")
    mx.save_safetensors(str(weights_path), mlx_weights)

    # Save config
    config_out_path = output_dir / "config.json"
    print(f"Saving config to {config_out_path}")
    with open(config_out_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nConversion complete! Output saved to {output_dir}")
    print(f"  - model.safetensors: {weights_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  - config.json")

    return mlx_weights, config_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepFilterNet PyTorch weights to MLX"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to DeepFilterNet model directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for MLX model"
    )
    parser.add_argument("--name", type=str, default="DeepFilterNet3", help="Model name")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Find checkpoint
    checkpoint_dir = input_dir / "checkpoints"
    if checkpoint_dir.exists():
        # Look for best checkpoint
        checkpoints = list(checkpoint_dir.glob("*.best"))
        if not checkpoints:
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if checkpoints:
            checkpoint_path = checkpoints[0]
        else:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    else:
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find config
    config_path = input_dir / "config.ini"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    convert_pytorch_to_mlx(checkpoint_path, config_path, output_dir, args.name)


if __name__ == "__main__":
    main()
