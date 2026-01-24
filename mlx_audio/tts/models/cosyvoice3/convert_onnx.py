# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
"""Convert CosyVoice3 ONNX models (CAMPPlus, S3Tokenizer) to MLX safetensors."""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np


def extract_onnx_weights(onnx_path: str) -> dict:
    """Extract weights from an ONNX model file, resolving onnx:: key names.

    Handles Conv, MatMul, Add (bias), and Mul (LayerNorm weight) node types.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    init_map = {init.name: init for init in model.graph.initializer}

    # Build mapping from onnx:: keys to proper names using graph nodes
    onnx_to_name = {}
    matmul_weights = set()  # Track MatMul weights that need transposition
    for node in model.graph.node:
        onnx_inputs = [inp for inp in node.input if inp.startswith("onnx::")]
        if not onnx_inputs:
            continue

        # Extract PyTorch path from node name
        name_parts = node.name.strip("/").split("/")

        if node.op_type == "Conv":
            # Remove trailing "Conv" from name
            if name_parts[-1] == "Conv":
                name_parts = name_parts[:-1]
            pytorch_path = ".".join(name_parts)
            for idx, inp in enumerate(onnx_inputs):
                param_name = "weight" if idx == 0 else "bias"
                onnx_to_name[inp] = f"{pytorch_path}.{param_name}"

        elif node.op_type == "MatMul":
            # Remove trailing "MatMul" from name
            if name_parts[-1] == "MatMul":
                name_parts = name_parts[:-1]
            pytorch_path = ".".join(name_parts)
            for inp in onnx_inputs:
                onnx_to_name[inp] = f"{pytorch_path}.weight"
                matmul_weights.add(inp)

        elif node.op_type == "Add":
            # Add nodes: bias for linear layers or LayerNorm bias
            # Remove trailing "Add" or "Add_N" from name
            if name_parts[-1].startswith("Add"):
                name_parts = name_parts[:-1]
            pytorch_path = ".".join(name_parts)
            for inp in onnx_inputs:
                if inp in init_map:
                    arr = numpy_helper.to_array(init_map[inp])
                    if arr.ndim >= 1 and arr.size > 1:
                        onnx_to_name[inp] = f"{pytorch_path}.bias"

        elif node.op_type == "Mul":
            # Mul nodes: LayerNorm weight
            if name_parts[-1].startswith("Mul"):
                name_parts = name_parts[:-1]
            pytorch_path = ".".join(name_parts)
            for inp in onnx_inputs:
                if inp in init_map:
                    arr = numpy_helper.to_array(init_map[inp])
                    if arr.ndim >= 1 and arr.size > 1:
                        onnx_to_name[inp] = f"{pytorch_path}.weight"

    # Extract all weights with proper names
    weights = {}
    for init in model.graph.initializer:
        orig_name = init.name
        name = orig_name
        if name.startswith("onnx::"):
            if name in onnx_to_name:
                name = onnx_to_name[name]
            else:
                continue  # Skip unmapped onnx:: keys (reshape constants, etc.)
        arr = numpy_helper.to_array(init)
        # Transpose MatMul weights: ONNX stores (in, out), MLX Linear needs (out, in)
        if orig_name in matmul_weights and arr.ndim == 2:
            arr = arr.T
        weights[name] = mx.array(arr)

    return weights


def convert_campplus(onnx_path: str, output_path: str):
    """Convert CAMPPlus ONNX to safetensors."""
    from mlx_audio.tts.models.cosyvoice3.campplus import CAMPPlus

    print(f"Converting CAMPPlus: {onnx_path}")
    raw_weights = extract_onnx_weights(onnx_path)
    print(f"  Extracted {len(raw_weights)} weights (with resolved names)")

    # Create model
    model = CAMPPlus(feat_dim=80, embedding_size=192)

    # Sanitize: renames + transposes
    sanitized = model.sanitize(raw_weights)
    print(f"  Sanitized to {len(sanitized)} MLX weights")

    # Check for missing parameters
    from mlx.utils import tree_flatten

    expected = dict(tree_flatten(model.parameters()))
    missing = [k for k in expected if k not in sanitized]
    if missing:
        print(f"  Warning: {len(missing)} missing parameters:")
        for k in missing[:10]:
            print(f"    {k}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")

    # Save as safetensors
    mx.save_safetensors(output_path, sanitized)
    print(f"  Saved to {output_path}")

    # Verify by loading
    model.load_weights(list(sanitized.items()), strict=False)
    mx.eval(model.parameters())
    print("  Verification: model loads successfully")

    return model


def convert_speech_tokenizer(onnx_path: str, output_path: str):
    """Convert S3 speech tokenizer ONNX to safetensors."""
    from mlx_audio.codec.models.s3.model_v2 import S3TokenizerV2

    print(f"Converting Speech Tokenizer: {onnx_path}")
    raw_weights = extract_onnx_weights(onnx_path)
    print(f"  Extracted {len(raw_weights)} weights")

    # ONNX exports paths relative to the encoder module.
    # Fix naming: add encoder prefix and flatten nested mlp.mlp → mlp
    prefixed = {}
    for k, v in raw_weights.items():
        new_k = k
        # The ONNX export has mlp.mlp.N (nested Sequential attr) → flatten to mlp.N
        new_k = new_k.replace(".mlp.mlp.", ".mlp.")
        # Add encoder prefix for encoder-level params
        if new_k.startswith("blocks.") or new_k.startswith("conv1.") or new_k.startswith("conv2."):
            new_k = f"encoder.{new_k}"
        prefixed[new_k] = v

    # Create model with v3 name
    model = S3TokenizerV2("speech_tokenizer_v3")

    # Use existing sanitize method
    sanitized = model.sanitize(prefixed)
    print(f"  Sanitized to {len(sanitized)} MLX weights")

    # Check for missing parameters
    from mlx.utils import tree_flatten

    expected = dict(tree_flatten(model.parameters()))
    missing = [k for k in expected if k not in sanitized]
    if missing:
        print(f"  Warning: {len(missing)} missing parameters:")
        for k in missing[:10]:
            print(f"    {k}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")

    # Save as safetensors
    mx.save_safetensors(output_path, sanitized)
    print(f"  Saved to {output_path}")

    # Verify by loading
    model.load_weights(list(sanitized.items()), strict=False)
    mx.eval(model.parameters())
    print("  Verification: model loads successfully")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Convert CosyVoice3 ONNX models to MLX safetensors"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to CosyVoice3 pretrained model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to model-dir)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir

    # Convert CAMPPlus
    campplus_onnx = model_dir / "campplus.onnx"
    if campplus_onnx.exists():
        convert_campplus(
            str(campplus_onnx),
            str(output_dir / "campplus.safetensors"),
        )
    else:
        print(f"Warning: {campplus_onnx} not found, skipping CAMPPlus")

    # Convert Speech Tokenizer
    tokenizer_onnx = model_dir / "speech_tokenizer_v3.onnx"
    if tokenizer_onnx.exists():
        convert_speech_tokenizer(
            str(tokenizer_onnx),
            str(output_dir / "speech_tokenizer_v3.safetensors"),
        )
    else:
        print(f"Warning: {tokenizer_onnx} not found, skipping speech tokenizer")


if __name__ == "__main__":
    main()
