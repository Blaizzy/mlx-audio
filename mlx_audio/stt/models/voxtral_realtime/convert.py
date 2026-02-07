"""Convert Voxtral Realtime from Mistral format to mlx-audio format.

The HuggingFace repo (mistralai/Voxtral-Mini-4B-Realtime-2602) uses:
- params.json (Mistral config format)
- consolidated.safetensors (single weight file)
- tekken.json (tokenizer)

This script converts params.json to config.json with model_type: voxtral_realtime
and optionally converts weights to MLX format with quantization.

Usage:
    python -m mlx_audio.stt.models.voxtral_realtime.convert \
        --model mistralai/Voxtral-Mini-4B-Realtime-2602 \
        --output ./voxtral-realtime-mlx
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

from mlx_audio.utils import get_model_path


def params_to_config(params: dict) -> dict:
    """Convert Mistral params.json to mlx-audio config.json format."""
    config = {"model_type": "voxtral_realtime"}

    # Copy top-level decoder params
    decoder_keys = [
        "dim", "n_layers", "head_dim", "hidden_dim", "n_heads",
        "n_kv_heads", "vocab_size", "norm_eps", "rope_theta",
        "sliding_window", "tied_embeddings",
    ]
    decoder = {}
    for k in decoder_keys:
        if k in params:
            decoder[k] = params[k]
    config["decoder"] = decoder

    # Copy encoder args (nested under multimodal.whisper_model_args)
    multimodal = params.get("multimodal", {})
    whisper_args = multimodal.get("whisper_model_args", {})
    encoder_args = whisper_args.get("encoder_args", {})
    if encoder_args:
        # Pull downsample_factor from downsample_args
        ds_args = whisper_args.get("downsample_args", {})
        if "downsample_factor" in ds_args:
            encoder_args["downsample_factor"] = ds_args["downsample_factor"]
        config["encoder_args"] = encoder_args
    elif "encoder_args" in params:
        config["encoder_args"] = params["encoder_args"]

    # Ada norm
    if "ada_rms_norm_t_cond" in params:
        config["decoder"]["ada_rms_norm_t_cond"] = params["ada_rms_norm_t_cond"]
    if "ada_rms_norm_t_cond_dim" in params:
        config["decoder"]["ada_rms_norm_t_cond_dim"] = params["ada_rms_norm_t_cond_dim"]

    # Transcription delay
    if "transcription_delay_ms" in params:
        config["transcription_delay_ms"] = params["transcription_delay_ms"]

    return config


def convert(
    model_path: str,
    output_path: str,
    dtype: str = "float16",
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
):
    """Convert Voxtral Realtime to mlx-audio format."""
    # Download / resolve model path
    src = get_model_path(model_path)
    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)

    # Convert params.json -> config.json
    params_file = src / "params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"params.json not found at {src}")

    with open(params_file) as f:
        params = json.load(f)

    config = params_to_config(params)

    if quantize:
        config["quantization"] = {
            "group_size": q_group_size,
            "bits": q_bits,
        }

    with open(dst / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Created config.json at {dst}")

    # Copy tekken.json
    tekken_src = src / "tekken.json"
    if tekken_src.exists():
        shutil.copy2(tekken_src, dst / "tekken.json")
        print(f"Copied tekken.json")

    # Load and optionally convert weights
    weight_file = src / "consolidated.safetensors"
    if not weight_file.exists():
        # Try model.safetensors or weight shard files
        import glob
        weight_files = glob.glob(str(src / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors files found at {src}")
        print(f"Loading {len(weight_files)} weight file(s)...")
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
    else:
        print(f"Loading weights from consolidated.safetensors...")
        weights = mx.load(str(weight_file))

    # Convert dtype
    dtype_map = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    target_dtype = dtype_map.get(dtype, mx.float16)

    converted = {}
    for k, v in weights.items():
        # Keep norms and biases in float32
        if "norm" in k or "bias" in k:
            converted[k] = v.astype(mx.float32)
        else:
            converted[k] = v.astype(target_dtype)

    # Save weights
    mx.save_safetensors(str(dst / "model.safetensors"), converted)
    print(f"Saved weights ({len(converted)} tensors) to {dst / 'model.safetensors'}")

    print(f"\nConversion complete! Model saved to {dst}")
    print(f"Load with: mlx_audio.stt.load('{dst}')")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Voxtral Realtime to mlx-audio format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Voxtral-Mini-4B-Realtime-2602",
        help="HuggingFace repo ID or local path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Weight dtype",
    )
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--q-bits", type=int, default=4, help="Quantization bits")
    parser.add_argument(
        "--q-group-size", type=int, default=64, help="Quantization group size"
    )
    args = parser.parse_args()

    convert(
        model_path=args.model,
        output_path=args.output,
        dtype=args.dtype,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )


if __name__ == "__main__":
    main()
