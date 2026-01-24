# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
"""
Convert CosyVoice3 weights to MLX safetensors format.

Converts:
- PyTorch .pt files (flow.pt, hift.pt, llm.pt) -> model.safetensors
- ONNX files (campplus.onnx, speech_tokenizer_v3.onnx) -> .safetensors

Usage:
    python -m mlx_audio.tts.models.cosyvoice3.convert --model-dir /path/to/model
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np


# --- ONNX conversion helpers ---


def extract_onnx_weights(onnx_path: str) -> dict:
    """Extract weights from an ONNX model file, resolving onnx:: key names."""
    try:
        import onnx
        from onnx import numpy_helper

    except ImportError:
        raise ImportError("onnx not installed, please install it with `pip install onnx`")


    model = onnx.load(onnx_path)
    init_map = {init.name: init for init in model.graph.initializer}

    # Build mapping from onnx:: keys to proper names using graph nodes
    onnx_to_name = {}
    matmul_weights = set()
    for node in model.graph.node:
        onnx_inputs = [inp for inp in node.input if inp.startswith("onnx::")]
        if not onnx_inputs:
            continue

        name_parts = node.name.strip("/").split("/")

        if node.op_type == "Conv":
            if name_parts[-1] == "Conv":
                name_parts = name_parts[:-1]
            pytorch_path = ".".join(name_parts)
            for idx, inp in enumerate(onnx_inputs):
                param_name = "weight" if idx == 0 else "bias"
                onnx_to_name[inp] = f"{pytorch_path}.{param_name}"

        elif node.op_type == "MatMul":
            if name_parts[-1] == "MatMul":
                name_parts = name_parts[:-1]
            pytorch_path = ".".join(name_parts)
            for inp in onnx_inputs:
                onnx_to_name[inp] = f"{pytorch_path}.weight"
                matmul_weights.add(inp)

        elif node.op_type == "Add":
            if name_parts[-1].startswith("Add"):
                name_parts = name_parts[:-1]
            pytorch_path = ".".join(name_parts)
            for inp in onnx_inputs:
                if inp in init_map:
                    arr = numpy_helper.to_array(init_map[inp])
                    if arr.ndim >= 1 and arr.size > 1:
                        onnx_to_name[inp] = f"{pytorch_path}.bias"

        elif node.op_type == "Mul":
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
                continue
        arr = numpy_helper.to_array(init)
        if orig_name in matmul_weights and arr.ndim == 2:
            arr = arr.T
        weights[name] = mx.array(arr)

    return weights


def convert_campplus(onnx_path: str, output_path: str):
    """Convert CAMPPlus ONNX to safetensors."""
    from mlx.utils import tree_flatten

    from mlx_audio.tts.models.cosyvoice3.campplus import CAMPPlus

    print(f"  Converting CAMPPlus: {onnx_path}")
    raw_weights = extract_onnx_weights(onnx_path)
    print(f"    Extracted {len(raw_weights)} weights")

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    sanitized = model.sanitize(raw_weights)
    print(f"    Sanitized to {len(sanitized)} MLX weights")

    expected = dict(tree_flatten(model.parameters()))
    missing = [k for k in expected if k not in sanitized]
    if missing:
        print(f"    Warning: {len(missing)} missing parameters")

    mx.save_safetensors(output_path, sanitized)
    model.load_weights(list(sanitized.items()), strict=False)
    mx.eval(model.parameters())
    print(f"    Saved to {output_path}")


def convert_speech_tokenizer(onnx_path: str, output_path: str):
    """Convert S3 speech tokenizer ONNX to safetensors."""
    from mlx.utils import tree_flatten

    from mlx_audio.codec.models.s3.model_v2 import S3TokenizerV2

    print(f"  Converting Speech Tokenizer: {onnx_path}")
    raw_weights = extract_onnx_weights(onnx_path)
    print(f"    Extracted {len(raw_weights)} weights")

    # Fix naming: add encoder prefix and flatten nested mlp.mlp -> mlp
    prefixed = {}
    for k, v in raw_weights.items():
        new_k = k.replace(".mlp.mlp.", ".mlp.")
        if new_k.startswith("blocks.") or new_k.startswith("conv1.") or new_k.startswith("conv2."):
            new_k = f"encoder.{new_k}"
        prefixed[new_k] = v

    model = S3TokenizerV2("speech_tokenizer_v3")
    sanitized = model.sanitize(prefixed)
    print(f"    Sanitized to {len(sanitized)} MLX weights")

    expected = dict(tree_flatten(model.parameters()))
    missing = [k for k in expected if k not in sanitized]
    if missing:
        print(f"    Warning: {len(missing)} missing parameters")

    mx.save_safetensors(output_path, sanitized)
    model.load_weights(list(sanitized.items()), strict=False)
    mx.eval(model.parameters())
    print(f"    Saved to {output_path}")


# --- PyTorch .pt conversion ---


def convert_model_weights(model_dir: Path, output_path: str):
    """Convert flow.pt, hift.pt, llm.pt to a single model.safetensors."""
    try:
        import torch
    except ImportError:
        raise ImportError("torch not installed, please install it with `pip install torch`")

    from .cosyvoice3 import Model, ModelConfig

    config = ModelConfig(model_path=model_dir)
    model = Model(config, load_llm=True)

    all_weights = {}

    # Flow weights
    flow_path = model_dir / "flow.pt"
    if flow_path.exists():
        flow_pt = torch.load(str(flow_path), map_location="cpu", weights_only=True)
        for k, v in flow_pt.items():
            new_key = f"flow.{k}"
            if "weight" in k and v.ndim == 3:
                v = v.permute(0, 2, 1)
            all_weights[new_key] = mx.array(v.numpy())
        print(f"    Loaded flow.pt: {len(flow_pt)} keys")
    else:
        print(f"    WARNING: flow.pt not found")

    # HIFT weights
    hift_path = model_dir / "hift.pt"
    if hift_path.exists():
        hift_pt = torch.load(str(hift_path), map_location="cpu", weights_only=True)
        for k, v in hift_pt.items():
            new_key = f"hift.{k}"
            if "weight" in k and v.ndim == 3 and "parametrizations" not in k:
                v = v.permute(0, 2, 1)
            all_weights[new_key] = mx.array(v.numpy())
        print(f"    Loaded hift.pt: {len(hift_pt)} keys")
    else:
        print(f"    WARNING: hift.pt not found")

    # LLM weights
    llm_path = model_dir / "llm.pt"
    if llm_path.exists():
        llm_pt = torch.load(str(llm_path), map_location="cpu", weights_only=True)
        for k, v in llm_pt.items():
            new_key = f"llm.{k}"
            all_weights[new_key] = mx.array(v.numpy())
        print(f"    Loaded llm.pt: {len(llm_pt)} keys")
    else:
        print(f"    WARNING: llm.pt not found")

    # Sanitize and save
    sanitized = model.sanitize(all_weights)
    print(f"    Sanitized: {len(all_weights)} -> {len(sanitized)} keys")

    mx.save_safetensors(output_path, sanitized)
    print(f"    Saved to {output_path}")


# --- Main ---


def convert(model_dir: str, output_dir: str = None):
    """
    Convert all CosyVoice3 weights to safetensors format.

    Converts PyTorch .pt files and ONNX models in one pass.

    Args:
        model_dir: Directory containing the original model files
        output_dir: Output directory (defaults to model_dir)
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir) if output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting CosyVoice3 from {model_dir}")

    # 1. Convert PyTorch model weights -> model.safetensors
    has_pt = (model_dir / "flow.pt").exists() or (model_dir / "hift.pt").exists()
    if has_pt:
        print("\n[1/3] Converting PyTorch weights (flow.pt, hift.pt, llm.pt)...")
        convert_model_weights(model_dir, str(output_dir / "model.safetensors"))
    else:
        print("\n[1/3] No .pt files found, skipping model weight conversion")

    # 2. Convert CAMPPlus ONNX -> campplus.safetensors
    campplus_onnx = model_dir / "campplus.onnx"
    if campplus_onnx.exists():
        print("\n[2/3] Converting CAMPPlus ONNX...")
        convert_campplus(str(campplus_onnx), str(output_dir / "campplus.safetensors"))
    else:
        print("\n[2/3] campplus.onnx not found, skipping")

    # 3. Convert Speech Tokenizer ONNX -> speech_tokenizer_v3.safetensors
    tokenizer_onnx = model_dir / "speech_tokenizer_v3.onnx"
    if tokenizer_onnx.exists():
        print("\n[3/3] Converting Speech Tokenizer ONNX...")
        convert_speech_tokenizer(
            str(tokenizer_onnx), str(output_dir / "speech_tokenizer_v3.safetensors")
        )
    else:
        print("\n[3/3] speech_tokenizer_v3.onnx not found, skipping")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CosyVoice3 weights (PyTorch + ONNX) to MLX safetensors"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing flow.pt, hift.pt, llm.pt, campplus.onnx, speech_tokenizer_v3.onnx",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to model-dir)",
    )
    args = parser.parse_args()

    convert(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
