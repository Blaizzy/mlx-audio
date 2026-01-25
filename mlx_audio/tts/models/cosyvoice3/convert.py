# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
"""
Convert CosyVoice3 weights to MLX safetensors format.

Converts:
- PyTorch .pt files (flow.pt, hift.pt, llm.pt) + campplus.onnx -> model.safetensors
- ONNX file (speech_tokenizer_v3.onnx) -> speech_tokenizer_v3.safetensors

Usage:
    # Convert to float32 (default)
    python -m mlx_audio.tts.models.cosyvoice3.convert --model-dir /path/to/model

    # Convert to float16 (HiFT stays in fp32 for numerical stability)
    python -m mlx_audio.tts.models.cosyvoice3.convert --model-dir /path/to/model --dtype float16
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


def cast_weights(weights: dict, dtype: str) -> dict:
    """
    Cast model weights to the specified dtype.

    HiFT weights are kept in fp32 for numerical stability (vocoder needs precision
    for upsampling/ResBlocks which accumulate large values that overflow in fp16).

    Args:
        weights: Dictionary of model weights
        dtype: Target dtype ("float16" or "bfloat16")

    Returns:
        Dictionary of casted weights
    """
    dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
    target_dtype = dtype_map[dtype]

    casted = {}
    casted_count = 0
    kept_count = 0

    for name, weight in weights.items():
        # Keep HiFT in fp32 for numerical stability
        # The vocoder's upsampling and ResBlocks accumulate values that overflow in fp16
        if name.startswith("hift."):
            casted[name] = weight
            kept_count += 1
        elif weight.dtype in (mx.float32, mx.float16, mx.bfloat16):
            casted[name] = weight.astype(target_dtype)
            casted_count += 1
        else:
            casted[name] = weight
            kept_count += 1

    print(f"    Casted {casted_count} weights to {dtype}, kept {kept_count} unchanged")
    return casted


def convert_model_weights(model_dir: Path, output_path: str, dtype: str = "float32"):
    """Convert flow.pt, hift.pt, llm.pt, and campplus.onnx to a single model.safetensors."""
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

    # CAMPPlus weights (from ONNX)
    campplus_onnx = model_dir / "campplus.onnx"
    if campplus_onnx.exists():
        raw_weights = extract_onnx_weights(str(campplus_onnx))
        for k, v in raw_weights.items():
            all_weights[f"campplus.{k}"] = v
        print(f"    Loaded campplus.onnx: {len(raw_weights)} keys")
    else:
        print(f"    WARNING: campplus.onnx not found")

    # Sanitize
    sanitized = model.sanitize(all_weights)
    print(f"    Sanitized: {len(all_weights)} -> {len(sanitized)} keys")

    # Cast dtype if needed
    if dtype != "float32":
        sanitized = cast_weights(sanitized, dtype)

    mx.save_safetensors(output_path, sanitized)
    print(f"    Saved to {output_path}")


# --- Main ---


def convert(model_dir: str, output_dir: str = None, dtype: str = "float32"):
    """
    Convert all CosyVoice3 weights to safetensors format.

    Converts PyTorch .pt files and ONNX models in one pass.

    Args:
        model_dir: Directory containing the original model files, or HuggingFace model ID
        output_dir: Output directory (defaults to model_dir)
        dtype: Target dtype ("float32", "float16", or "bfloat16")
    """
    # Handle HuggingFace model IDs
    model_path = Path(model_dir)

    # Check if this looks like a HuggingFace model ID (contains /) and doesn't have model files
    is_hf_id = "/" in model_dir and not (
        (model_path / "model.safetensors").exists()
        or (model_path / "flow.pt").exists()
    )

    if is_hf_id or not model_path.exists():
        # Try to download from HuggingFace
        try:
            from huggingface_hub import snapshot_download

            print(f"Downloading {model_dir} from HuggingFace...")
            model_dir = snapshot_download(model_dir)
            model_path = Path(model_dir)
            print(f"Downloaded to {model_dir}")
        except Exception as e:
            raise FileNotFoundError(f"Model directory not found: {model_dir}. Error: {e}")

    model_dir = model_path
    output_dir = Path(output_dir) if output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting CosyVoice3 from {model_dir}")
    if dtype != "float32":
        print(f"Target dtype: {dtype} (HiFT stays in fp32 for numerical stability)")

    # 1. Convert model weights (flow.pt, hift.pt, llm.pt, campplus.onnx) -> model.safetensors
    has_pt = (model_dir / "flow.pt").exists() or (model_dir / "hift.pt").exists()

    if has_pt:
        print("\n[1/4] Converting model weights (flow.pt, hift.pt, llm.pt, campplus.onnx)...")
        convert_model_weights(model_dir, str(output_dir / "model.safetensors"), dtype)
    else:
        print("\n[1/4] No .pt files found, skipping model weight conversion")

    # 2. Convert Speech Tokenizer ONNX -> speech_tokenizer/speech_tokenizer_v3.safetensors
    tokenizer_onnx = model_dir / "speech_tokenizer_v3.onnx"
    tokenizer_safetensors = model_dir / "speech_tokenizer_v3.safetensors"
    speech_tokenizer_dir = output_dir / "speech_tokenizer"

    if tokenizer_safetensors.exists():
        # Copy existing safetensors
        print("\n[2/4] Found existing speech_tokenizer_v3.safetensors, copying...")
        speech_tokenizer_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(tokenizer_safetensors, speech_tokenizer_dir / "speech_tokenizer_v3.safetensors")
        print(f"    Copied to {speech_tokenizer_dir / 'speech_tokenizer_v3.safetensors'}")
    elif tokenizer_onnx.exists():
        print("\n[2/4] Converting Speech Tokenizer ONNX...")
        speech_tokenizer_dir.mkdir(parents=True, exist_ok=True)
        convert_speech_tokenizer(
            str(tokenizer_onnx), str(speech_tokenizer_dir / "speech_tokenizer_v3.safetensors")
        )
    else:
        print("\n[2/4] speech_tokenizer_v3.onnx not found, skipping")

    # 3. Copy tokenizer and config files
    print("\n[3/4] Copying tokenizer and config files...")
    import shutil

    files_to_copy = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "cosyvoice3.yaml",
    ]

    # Check for tokenizer files in subdirectories (e.g., CosyVoice-BlankEN/)
    tokenizer_dirs = [".", "CosyVoice-BlankEN"]
    for fname in files_to_copy:
        for subdir in tokenizer_dirs:
            src = model_dir / subdir / fname
            if src.exists():
                shutil.copy(src, output_dir / fname)
                print(f"    Copied {fname}" + (f" (from {subdir}/)" if subdir != "." else ""))
                break

    # Create config.json if it doesn't exist
    config_path = output_dir / "config.json"
    if not config_path.exists():
        import json

        config = {
            "model_type": "cosyvoice3",
            "sample_rate": 24000,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("    Created config.json")

    # 4. Generate random noise matching PyTorch exactly
    print("\n[4/4] Generating random noise (matching PyTorch seeds)...")
    generate_random_noise(output_dir)

    print("\nDone!")


def generate_random_noise(output_dir: Path):
    """Generate random noise matching PyTorch's RNG for reproducibility."""
    try:
        import torch
    except ImportError:
        print("    WARNING: torch not installed, skipping random noise generation")
        print("    Install torch to generate noise matching PyTorch exactly")
        return

    random_noise_dir = output_dir / "random_noise"
    random_noise_dir.mkdir(parents=True, exist_ok=True)

    noise_tensors = {}

    # Flow CFM noise: torch.manual_seed(0); torch.randn([1, 80, 50*300])
    torch.manual_seed(0)
    flow_rand_noise = torch.randn([1, 80, 50 * 300])
    noise_tensors["flow_rand_noise"] = mx.array(flow_rand_noise.numpy())
    print(f"    Generated flow_rand_noise: {flow_rand_noise.shape}")

    # HiFT sine_waves_noise: torch.manual_seed(2); torch.rand([1, 300*24000, 9])
    torch.manual_seed(2)
    sine_waves_noise = torch.rand([1, 300 * 24000, 9])
    noise_tensors["sine_waves_noise"] = mx.array(sine_waves_noise.numpy())
    print(f"    Generated sine_waves_noise: {sine_waves_noise.shape}")

    # HiFT fixed_noise: torch.manual_seed(3); torch.rand([1, 300*24000, 1])
    torch.manual_seed(3)
    fixed_noise = torch.rand([1, 300 * 24000, 1])
    noise_tensors["fixed_noise"] = mx.array(fixed_noise.numpy())
    print(f"    Generated fixed_noise: {fixed_noise.shape}")

    # Save as safetensors
    output_path = random_noise_dir / "noise.safetensors"
    mx.save_safetensors(str(output_path), noise_tensors)
    print(f"    Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CosyVoice3 weights (PyTorch + ONNX) to MLX safetensors"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing model files (PyTorch .pt files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to model-dir)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Target dtype for weights (default: float32). HiFT stays in fp32 for stability.",
    )
    args = parser.parse_args()

    convert(args.model_dir, args.output_dir, args.dtype)


if __name__ == "__main__":
    main()
