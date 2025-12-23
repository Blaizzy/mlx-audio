# Copyright (c) 2024 MLX Audio Contributors

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import numpy as np


def convert_pytorch_to_mlx(
    pytorch_weights: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, mx.array]:
    """
    Convert PyTorch weights to MLX format.

    Args:
        pytorch_weights: Dictionary of PyTorch weight tensors
        config: Model configuration dictionary

    Returns:
        Dictionary of MLX weight arrays
    """
    mlx_weights = {}

    # Weight name mapping patterns
    name_mappings = [
        # Transformer layers
        (
            r"transformer\.layers\.(\d+)\.attention\.",
            r"transformer.layers.\1.attention.",
        ),
        (
            r"transformer\.layers\.(\d+)\.feed_forward\.",
            r"transformer.layers.\1.feed_forward.",
        ),
        (
            r"transformer\.layers\.(\d+)\.attention_norm\.",
            r"transformer.layers.\1.attention_norm.",
        ),
        (r"transformer\.layers\.(\d+)\.ffn_norm\.", r"transformer.layers.\1.ffn_norm."),
        (
            r"transformer\.layers\.(\d+)\.cross_attention\.",
            r"transformer.layers.\1.cross_attention.",
        ),
        (
            r"transformer\.layers\.(\d+)\.scale_shift_table",
            r"transformer.layers.\1.scale_shift_table",
        ),
        # DiT embedders
        (r"transformer\.x_embedder\.", r"transformer.x_embedder."),
        (r"transformer\.y_embedder\.", r"transformer.y_embedder."),
        (r"transformer\.t_embedder\.", r"transformer.t_embedder."),
        (r"transformer\.t_block\.", r"transformer.t_block."),
        (r"transformer\.norm\.", r"transformer.norm."),
        (r"transformer\.output\.", r"transformer.output."),
        # Audio codec
        (r"audio_codec\.encoder\.", r"audio_codec.encoder."),
        (r"audio_codec\.decoder\.", r"audio_codec.decoder."),
        (r"audio_codec\.quantizer\.", r"audio_codec.quantizer_"),
        # Other components
        (r"proj\.", r"proj."),
        (r"memory_proj\.", r"memory_proj."),
        (r"embed_anchors\.", r"embed_anchors."),
        (r"timestep_emb\.", r"timestep_emb."),
    ]

    for name, tensor in pytorch_weights.items():
        # Skip text encoder and ranker weights (loaded separately)
        if any(
            name.startswith(prefix)
            for prefix in [
                "text_encoder.",
                "visual_ranker.",
                "text_ranker.",
                "span_predictor.",
            ]
        ):
            continue

        # Convert tensor to numpy then MLX
        if hasattr(tensor, "cpu"):
            np_array = tensor.cpu().numpy()
        else:
            np_array = np.array(tensor)

        # Apply name mappings
        mlx_name = name
        for pattern, replacement in name_mappings:
            mlx_name = re.sub(pattern, replacement, mlx_name)

        # Handle specific weight transformations
        if ".weight" in mlx_name and len(np_array.shape) == 2:
            # Linear layer weights may need transposition
            # MLX uses (out_features, in_features) like PyTorch, so usually no change needed
            pass

        # Conv1d weights: PyTorch (out, in, kernel) -> MLX (out, kernel, in)
        if "conv" in mlx_name.lower() and ".weight" in mlx_name:
            if len(np_array.shape) == 3:
                # PyTorch Conv1d: (out_channels, in_channels, kernel_size)
                # MLX Conv1d: (out_channels, kernel_size, in_channels)
                np_array = np.transpose(np_array, (0, 2, 1))

        # Weight normalization conversions
        if "weight_g" in mlx_name or "weight_v" in mlx_name:
            # Ensure proper shape for weight normalization
            pass

        mlx_weights[mlx_name] = mx.array(np_array)

    return mlx_weights


def convert_model(
    input_path: str,
    output_path: str,
    model_name: str = "sam-audio-large",
):
    """
    Convert a SAM-Audio model from PyTorch to MLX format.

    Args:
        input_path: Path to PyTorch model directory or checkpoint
        output_path: Path for output MLX model
        model_name: Model variant name
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = input_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Config not found at {config_path}")

    # Load PyTorch weights
    import torch

    weights_path = input_path / "checkpoint.pt"
    if not weights_path.exists():
        weights_path = input_path / "model.safetensors"
        if weights_path.exists():
            from safetensors.torch import load_file

            pytorch_weights = load_file(str(weights_path))
        else:
            raise FileNotFoundError(f"Weights not found at {input_path}")
    else:
        pytorch_weights = torch.load(str(weights_path), map_location="cpu")

    # Convert weights
    print(f"Converting {len(pytorch_weights)} weight tensors...")
    mlx_weights = convert_pytorch_to_mlx(pytorch_weights, config)
    print(f"Converted to {len(mlx_weights)} MLX tensors")

    # Save MLX weights
    output_weights_path = output_path / "model.safetensors"
    mx.save_safetensors(str(output_weights_path), mlx_weights)
    print(f"Saved weights to {output_weights_path}")

    # Copy config
    output_config_path = output_path / "config.json"
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {output_config_path}")

    print("Conversion complete!")


def download_and_convert(
    model_id: str = "facebook/sam-audio-large",
    output_path: str = None,
):
    """
    Download a model from HuggingFace and convert to MLX format.

    Args:
        model_id: HuggingFace model ID
        output_path: Output directory (default: ~/.cache/mlx-audio/sam-audio)
    """
    from huggingface_hub import snapshot_download

    if output_path is None:
        output_path = Path.home() / ".cache" / "mlx-audio" / model_id.replace("/", "_")

    # Download model
    print(f"Downloading {model_id}...")
    input_path = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.safetensors", "*.json", "*.bin", "*.pt"],
    )

    # Convert
    convert_model(input_path, output_path)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SAM-Audio to MLX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/sam-audio-large",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Local path to PyTorch model (alternative to --model-id)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output directory for MLX model",
    )

    args = parser.parse_args()

    if args.input_path:
        convert_model(args.input_path, args.output_path or "mlx_sam_audio")
    else:
        download_and_convert(args.model_id, args.output_path)
