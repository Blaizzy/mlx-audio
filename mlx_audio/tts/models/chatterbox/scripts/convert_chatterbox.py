#!/usr/bin/env python3
"""
Convert Chatterbox weights from PyTorch/ONNX to MLX format.

This script downloads the original weights and converts them to MLX-compatible
safetensors format. It uses the model's own sanitize() methods to ensure
consistency between conversion and runtime loading.

The S3Tokenizer is converted separately and uploaded to its own repo, as it's
shared between multiple TTS models (Chatterbox, CosyVoice2, etc.).

Usage:
    # Convert Chatterbox (without S3Tokenizer) to fp16
    python scripts/convert_chatterbox.py

    # Convert to 4-bit quantized
    python scripts/convert_chatterbox.py --quantize

    # Convert S3Tokenizer only (shared component)
    python scripts/convert_chatterbox.py --s3-tokenizer-only

    # Upload Chatterbox to Hugging Face
    python scripts/convert_chatterbox.py --quantize --upload-repo

    # Upload S3Tokenizer to Hugging Face
    python scripts/convert_chatterbox.py --s3-tokenizer-only --upload-repo

    # Custom repos
    python scripts/convert_chatterbox.py --upload-repo my-org/my-chatterbox
    python scripts/convert_chatterbox.py --s3-tokenizer-only --upload-repo my-org/my-s3tokenizer

Requirements (for conversion only):
    pip install torch safetensors huggingface_hub onnx s3tokenizer

After conversion, the model only needs:
    pip install mlx mlx-lm
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


def download_chatterbox_weights(cache_dir: Path) -> Path:
    """Download Chatterbox weights from Hugging Face."""
    from huggingface_hub import snapshot_download

    print("Downloading Chatterbox weights from Hugging Face...")
    ckpt_dir = Path(
        snapshot_download(
            repo_id="ResembleAI/chatterbox",
            allow_patterns=[
                "Cangjie5_TC.json",
                "ve.safetensors",
                "t3_mtl23ls_v2.safetensors",
                "s3gen.safetensors",
                "grapheme_mtl_merged_expanded_v1.json",
            ],
            cache_dir=cache_dir,
        )
    )
    print(f"Downloaded to: {ckpt_dir}")
    return ckpt_dir


def download_s3tokenizer_onnx(cache_dir: Path) -> Path:
    """Download S3Tokenizer ONNX weights from Hugging Face."""
    from huggingface_hub import hf_hub_download

    print("Downloading S3Tokenizer ONNX from Hugging Face...")
    onnx_path = hf_hub_download(
        repo_id="FunAudioLLM/CosyVoice2-0.5B",
        filename="speech_tokenizer_v2.onnx",
        cache_dir=cache_dir,
    )
    print(f"Downloaded to: {onnx_path}")
    return Path(onnx_path)


def load_pytorch_safetensors(path: Path) -> Dict[str, np.ndarray]:
    """Load PyTorch safetensors and convert to numpy."""
    import torch
    from safetensors.torch import load_file

    state_dict = load_file(path)
    return {k: v.cpu().numpy() for k, v in state_dict.items()}


def load_onnx_weights(path: Path) -> Dict[str, np.ndarray]:
    """Load ONNX weights as numpy arrays using s3tokenizer's onnx2torch."""
    try:
        # Use s3tokenizer's conversion utility for proper key naming
        import torch
        from s3tokenizer.utils import onnx2torch

        pytorch_weights = onnx2torch(str(path), None, False)

        # Convert PyTorch tensors to numpy arrays
        weights = {}
        for key, value in pytorch_weights.items():
            if isinstance(value, torch.Tensor):
                weights[key] = value.cpu().numpy()
            else:
                weights[key] = np.array(value)

        return weights

    except ImportError:
        # Fallback: direct ONNX parsing (gives onnx:: internal names)
        print("WARNING: s3tokenizer not installed, using raw ONNX parsing")
        print("         This may result in incorrect weight names")
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(path))
        weights = {}

        for initializer in model.graph.initializer:
            weights[initializer.name] = numpy_helper.to_array(initializer)

        return weights


def numpy_to_mlx(weights: Dict[str, np.ndarray]) -> Dict:
    """Convert numpy arrays to MLX arrays for sanitization."""
    import mlx.core as mx

    return {k: mx.array(v) for k, v in weights.items()}


def mlx_to_numpy(weights: Dict) -> Dict[str, np.ndarray]:
    """Convert MLX arrays back to numpy for saving."""
    import numpy as np

    return {k: np.array(v) for k, v in weights.items()}


def save_mlx_safetensors(weights: Dict[str, np.ndarray], path: Path):
    """Save weights as MLX-compatible safetensors (for non-quantized weights)."""
    from safetensors.numpy import save_file

    # Ensure all values are numpy arrays with correct dtype
    clean_weights = {}
    for k, v in weights.items():
        if isinstance(v, np.ndarray):
            # Keep original dtype but ensure it's a supported type
            if v.dtype == np.float64:
                v = v.astype(np.float32)
            clean_weights[k] = v
        else:
            clean_weights[k] = np.array(v)

    save_file(clean_weights, path)
    print(f"Saved: {path} ({len(clean_weights)} tensors)")


def save_mlx_quantized(weights: Dict, path: Path):
    """Save quantized MLX weights directly using mx.save_safetensors.

    This preserves the packed uint32 format for quantized weights.
    """
    import mlx.core as mx

    mx.save_safetensors(str(path), weights, metadata={"format": "mlx"})
    print(f"Saved: {path} ({len(weights)} tensors)")


def quantize_t3_backbone(model, bits: int = 4, group_size: int = 64):
    """
    Selectively quantize the T3 LLaMA backbone.

    Only quantizes tfmr.model.layers.* (MLP and attention layers).
    Other components are kept in full precision as they are sensitive to quantization.

    Args:
        model: T3 model instance (or full Chatterbox Model)
        bits: Quantization bits (default: 4)
        group_size: Quantization group size (default: 64)

    Returns:
        Number of layers quantized
    """
    import mlx.nn as nn

    quantized_count = [0]

    def should_quantize(path, module):
        """Only quantize T3 transformer layers."""
        if isinstance(module, nn.Linear):
            # Handle both "t3.tfmr.model.layers" (full model) and "tfmr.model.layers" (T3 only)
            if "tfmr.model.layers" in path:
                quantized_count[0] += 1
                return True
        return False

    nn.quantize(
        model, bits=bits, group_size=group_size, class_predicate=should_quantize
    )
    return quantized_count[0]


def generate_readme(path: Path, upload_repo: str):
    """Generate README.md model card for Chatterbox on Hugging Face."""
    from mlx_audio.version import __version__

    card_text = f"""---
library_name: mlx-audio
base_model:
- ResembleAI/chatterbox
tags:
- mlx
pipeline_tag: text-to-speech
---

# {upload_repo}

This model was converted to MLX format from [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) using [mlx-audio](https://github.com/DePasqualeOrg/mlx-audio) version **{__version__}**.

**Note:** This model requires the S3Tokenizer weights from [mlx-community/S3TokenizerV2](https://huggingface.co/mlx-community/S3TokenizerV2), which will be downloaded automatically.

## Use with mlx-audio

```bash
pip install -U git+https://github.com/litmudoc/mlx-audio.git@main mlx-audio
```

### Command line

```bash
mlx_audio.tts --model {upload_repo} --text "Hello, this is Chatterbox on MLX!" --ref_audio reference.wav --ref_text "."
```

### Python

```python
from mlx_audio.tts.generate import generate_audio

generate_audio(
    text="Hello, this is Chatterbox on MLX!",
    model="{upload_repo}",
    ref_audio="reference.wav",
    ref_text=".",
    file_prefix="output",
)
```
"""
    card_path = path / "README.md"
    with open(card_path, "w") as f:
        f.write(card_text)
    print(f"Created: {card_path}")


def generate_s3_tokenizer_readme(path: Path, upload_repo: str):
    """Generate README.md model card for S3Tokenizer on Hugging Face."""
    card_text = f"""---
library_name: mlx-audio
base_model:
- FunAudioLLM/CosyVoice2-0.5B
tags:
- mlx
- speech-tokenizer
---

# {upload_repo}

S3TokenizerV2 (Supervised Semantic Speech Tokenizer) converted to MLX format from [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B).

This tokenizer is automatically downloaded when using Chatterbox or CosyVoice2 with [mlx-audio](https://github.com/DePasqualeOrg/mlx-audio).
"""
    card_path = path / "README.md"
    with open(card_path, "w") as f:
        f.write(card_text)
    print(f"Created: {card_path}")


def upload_to_hub(path: Path, upload_repo: str):
    """Upload converted model to Hugging Face Hub."""
    from huggingface_hub import HfApi

    print(f"\nUploading to {upload_repo}...")
    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=str(path),
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful! Visit https://huggingface.co/{upload_repo}")


def convert_s3_tokenizer(
    output_dir: Path,
    cache_dir: Path = None,
    upload_repo: str = None,
    dry_run: bool = False,
):
    """
    Convert S3Tokenizer weights to MLX format (standalone).

    This creates a separate repo for the S3Tokenizer, which is shared between
    multiple TTS models (Chatterbox, CosyVoice2, etc.).

    Args:
        output_dir: Directory to save converted weights
        cache_dir: Directory to cache downloaded weights
        upload_repo: Optional Hugging Face repo to upload to
        dry_run: If True, generate all files including README but skip upload
    """
    import json

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "chatterbox-convert"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Import S3Tokenizer for sanitize method
    from mlx_audio.tts.models.chatterbox.s3tokenizer import S3TokenizerV2

    # Download and convert S3Tokenizer from ONNX
    print("Converting S3Tokenizer...")
    onnx_path = download_s3tokenizer_onnx(cache_dir)
    s3tok_weights = load_onnx_weights(onnx_path)
    s3tok_weights_mx = numpy_to_mlx(s3tok_weights)
    s3tok = S3TokenizerV2("speech_tokenizer_v2_25hz")
    s3tok_weights_mx = s3tok.sanitize(s3tok_weights_mx)
    s3tok_weights = mlx_to_numpy(s3tok_weights_mx)
    print(f"  Converted {len(s3tok_weights)} S3Tokenizer weights")

    # Save weights (no quantization for tokenizer)
    print("\nSaving model.safetensors...")
    save_mlx_safetensors(s3tok_weights, output_dir / "model.safetensors")

    # Create config.json
    print("Creating config.json...")
    config = {
        "model_type": "s3_tokenizer_v2",
        "version": "2.0",
        "sample_rate": 16000,
        "token_rate": 25,
        "codebook_size": 6561,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate README if upload_repo is specified
    if upload_repo:
        print("Generating README.md...")
        generate_s3_tokenizer_readme(output_dir, upload_repo)

    print(f"\n✅ S3Tokenizer conversion complete! Output directory: {output_dir}")
    print(f"\nTotal weights: {len(s3tok_weights)}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

    # Upload to Hugging Face if requested (and not dry run)
    if upload_repo and not dry_run:
        upload_to_hub(output_dir, upload_repo)
    elif upload_repo:
        print(f"\n📁 Dry run - to upload to {upload_repo}, run without --dry-run")


def convert_all(
    output_dir: Path,
    cache_dir: Path = None,
    upload_repo: str = None,
    quantize: bool = False,
    bits: int = 4,
    group_size: int = 64,
    dry_run: bool = False,
):
    """
    Convert Chatterbox weights to MLX format (without S3Tokenizer).

    Saves weights to a single model.safetensors file with component prefixes
    (ve.*, t3.*, s3gen.*). S3Tokenizer is handled separately as it's shared
    between multiple TTS models.

    Uses the model's own sanitize() methods to ensure consistency
    between conversion and runtime loading.

    Args:
        output_dir: Directory to save converted weights
        cache_dir: Directory to cache downloaded weights
        upload_repo: Optional Hugging Face repo to upload to (e.g., "mlx-community/Chatterbox-TTS-fp16")
        quantize: Whether to apply selective quantization to T3 backbone
        bits: Quantization bits (default: 4)
        group_size: Quantization group size (default: 64)
        dry_run: If True, generate all files including README but skip upload
    """
    import json
    import shutil

    import mlx.core as mx
    from mlx.utils import tree_flatten

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "chatterbox-convert"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download Chatterbox weights
    ckpt_dir = download_chatterbox_weights(cache_dir)

    # Import model components for their sanitize methods
    from mlx_audio.tts.models.chatterbox.s3gen import S3Token2Wav
    from mlx_audio.tts.models.chatterbox.t3 import T3
    from mlx_audio.tts.models.chatterbox.voice_encoder import VoiceEncoder

    # Combined weights dict with prefixes
    all_weights = {}

    # Convert VoiceEncoder
    print("\nConverting VoiceEncoder...")
    ve_weights = load_pytorch_safetensors(ckpt_dir / "ve.safetensors")
    ve_weights_mx = numpy_to_mlx(ve_weights)
    ve = VoiceEncoder()
    ve_weights_mx = ve.sanitize(ve_weights_mx)
    ve_weights = mlx_to_numpy(ve_weights_mx)
    # Add with prefix
    for k, v in ve_weights.items():
        all_weights[f"ve.{k}"] = v
    print(f"  Added {len(ve_weights)} VoiceEncoder weights")

    # Convert T3
    print("\nConverting T3...")
    t3_weights = load_pytorch_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
    t3_weights_mx = numpy_to_mlx(t3_weights)
    t3 = T3()
    t3_weights_mx = t3.sanitize(t3_weights_mx)
    t3_weights = mlx_to_numpy(t3_weights_mx)
    # Add with prefix
    for k, v in t3_weights.items():
        all_weights[f"t3.{k}"] = v
    print(f"  Added {len(t3_weights)} T3 weights")

    # Convert S3Gen (excluding tokenizer.* keys which come from ONNX)
    print("\nConverting S3Gen...")
    s3gen_weights = load_pytorch_safetensors(ckpt_dir / "s3gen.safetensors")
    # Filter out tokenizer.* keys - S3Tokenizer is in a separate repo
    s3gen_weights = {
        k: v for k, v in s3gen_weights.items() if not k.startswith("tokenizer.")
    }
    s3gen_weights_mx = numpy_to_mlx(s3gen_weights)
    s3gen = S3Token2Wav()
    s3gen_weights_mx = s3gen.sanitize(s3gen_weights_mx)
    s3gen_weights = mlx_to_numpy(s3gen_weights_mx)
    # Add with prefix
    for k, v in s3gen_weights.items():
        all_weights[f"s3gen.{k}"] = v
    print(f"  Added {len(s3gen_weights)} S3Gen weights")

    # Note: S3Tokenizer is NOT included - it's in a separate repo (mlx-community/S3TokenizerV2)
    print(
        "\nNote: S3Tokenizer weights are loaded separately from mlx-community/S3TokenizerV2"
    )

    # Apply quantization if requested
    if quantize:
        print(f"\nApplying {bits}-bit quantization to T3 backbone...")

        # Create fresh model instances for quantization
        # (NOT using Model.load_weights which downloads S3Tokenizer)
        ve_model = VoiceEncoder()
        t3_model = T3()
        s3gen_model = S3Token2Wav()

        all_weights_mx = numpy_to_mlx(all_weights)

        # Split and load weights by component
        ve_w = {k[3:]: v for k, v in all_weights_mx.items() if k.startswith("ve.")}
        t3_w = {k[3:]: v for k, v in all_weights_mx.items() if k.startswith("t3.")}
        s3gen_w = {
            k[6:]: v for k, v in all_weights_mx.items() if k.startswith("s3gen.")
        }

        ve_model.load_weights(list(ve_w.items()), strict=False)
        t3_model.load_weights(list(t3_w.items()), strict=False)
        s3gen_model.load_weights(list(s3gen_w.items()), strict=False)

        mx.eval(ve_model.parameters())
        mx.eval(t3_model.parameters())
        mx.eval(s3gen_model.parameters())

        # Get original size
        orig_size = sum(v.nbytes for v in all_weights_mx.values())
        print(f"  Original size: {orig_size / 1e9:.2f} GB")

        # Apply selective quantization to T3 backbone only
        num_quantized = quantize_t3_backbone(t3_model, bits=bits, group_size=group_size)
        mx.eval(t3_model.parameters())
        print(f"  Quantized {num_quantized} Linear layers")

        # Collect weights with prefixes (excluding S3Tokenizer)
        new_weights = {}
        for k, v in dict(tree_flatten(ve_model.parameters())).items():
            new_weights[f"ve.{k}"] = v
        for k, v in dict(tree_flatten(t3_model.parameters())).items():
            new_weights[f"t3.{k}"] = v
        for k, v in dict(tree_flatten(s3gen_model.parameters())).items():
            new_weights[f"s3gen.{k}"] = v

        new_size = sum(v.nbytes for v in new_weights.values())
        print(f"  New size: {new_size / 1e9:.2f} GB")
        print(f"  Reduction: {(1 - new_size / orig_size) * 100:.1f}%")

        # Save quantized weights directly using MLX (preserves uint32 packed format)
        print("\nSaving combined model.safetensors (quantized)...")
        save_mlx_quantized(new_weights, output_dir / "model.safetensors")
    else:
        # Save non-quantized weights using numpy format
        print("\nSaving combined model.safetensors...")
        save_mlx_safetensors(all_weights, output_dir / "model.safetensors")

    # Copy tokenizer.json
    print("\nCopying tokenizer.json...")
    shutil.copy(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json", output_dir / "tokenizer.json")
    # Copy Cangjie5_TC.json
    print("\nCopying Cangjie5_TC.json...")
    shutil.copy(ckpt_dir / "Cangjie5_TC.json", output_dir / "Cangjie5_TC.json")

    # Create config.json
    print("\nCreating config.json...")
    config = {
        "model_type": "chatterbox",
        "version": "1.0",
        "multilingual": True,
        "vocab_size": 2454,
    }
    if quantize:
        config["quantization"] = {
            "bits": bits,
            "group_size": group_size,
            "quantized_components": ["t3.tfmr.model.layers"],
        }
    else:
        config["precision"] = "fp16"
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate README if upload_repo is specified
    if upload_repo:
        print("\nGenerating README.md...")
        generate_readme(output_dir, upload_repo)

    print(
        f"\n{'🔢' if quantize else '✅'} Conversion complete! Output directory: {output_dir}"
    )
    total_weights = len(new_weights) if quantize else len(all_weights)
    print(f"\nTotal weights: {total_weights}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

    # Upload to Hugging Face if requested (and not dry run)
    if not dry_run:
        upload_to_hub(output_dir, upload_repo)
    else:
        print(f"\n📁 To upload to {upload_repo}, run with --upload-repo")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Chatterbox weights to MLX format"
    )
    parser.add_argument(
        "--mlx-path",
        type=Path,
        default=None,
        help="Output directory for MLX weights (default: ./Chatterbox-TTS-{fp16|Nbit} or ./S3TokenizerV2)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Cache directory for downloads"
    )
    parser.add_argument(
        "--upload-repo",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Upload to Hugging Face. Optionally specify repo (default: mlx-community/Chatterbox-TTS-{fp16|Nbit} or mlx-community/S3TokenizerV2)",
    )
    parser.add_argument(
        "--quantize",
        "-q",
        action="store_true",
        help="Apply 4-bit quantization to T3 backbone (reduces size by ~53%%)",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Quantization bits (default: 4)",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--s3-tokenizer-only",
        action="store_true",
        help="Only convert S3Tokenizer (shared component for multiple TTS models)",
    )
    args = parser.parse_args()

    # Determine if we should upload (--upload-repo was provided, even without value)
    should_upload = args.upload_repo is not None

    if args.s3_tokenizer_only:
        # S3Tokenizer conversion mode
        mlx_path = args.mlx_path or Path("./S3TokenizerV2")
        upload_repo = (
            (args.upload_repo or "mlx-community/S3TokenizerV2")
            if args.upload_repo is not None
            else "mlx-community/S3TokenizerV2"
        )

        convert_s3_tokenizer(
            output_dir=mlx_path,
            cache_dir=args.cache_dir,
            upload_repo=upload_repo,
            dry_run=not should_upload,
        )
    else:
        # Chatterbox conversion mode
        precision_suffix = f"{args.q_bits}bit" if args.quantize else "fp16"
        mlx_path = args.mlx_path or Path(f"./Chatterbox-TTS-{precision_suffix}")
        upload_repo = (
            (args.upload_repo or f"mlx-community/Chatterbox-TTS-{precision_suffix}")
            if args.upload_repo is not None
            else f"mlx-community/Chatterbox-TTS-{precision_suffix}"
        )

        convert_all(
            output_dir=mlx_path,
            cache_dir=args.cache_dir,
            upload_repo=upload_repo,
            quantize=args.quantize,
            bits=args.q_bits,
            group_size=args.q_group_size,
            dry_run=not should_upload,
        )


if __name__ == "__main__":
    main()
