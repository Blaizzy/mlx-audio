"""Convert k2-fsa/OmniVoice weights to mlx-audio format.

Downloads from HuggingFace (or uses a local directory) and produces
a ready-to-use mlx-audio model directory, optionally quantized.

Usage:
    # Convert and save locally:
    python -m mlx_audio.tts.models.omnivoice.convert \\
        --model k2-fsa/OmniVoice \\
        --output ./models/omnivoice-mlx

    # Convert to bfloat16 and upload to mlx-community:
    python -m mlx_audio.tts.models.omnivoice.convert \\
        --model k2-fsa/OmniVoice \\
        --output ./models/omnivoice-mlx-bf16 \\
        --dtype bfloat16 \\
        --upload mlx-community/OmniVoice-bf16
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

# ── key remapping ─────────────────────────────────────────────────────────────

NUM_CODEBOOKS = 8
AUDIO_VOCAB_SIZE = 1025  # 1024 real + 1 mask


def remap_backbone(weights: dict, dtype: mx.Dtype) -> dict:
    """Remap k2-fsa backbone keys → mlx-audio convention and cast dtype."""
    V = AUDIO_VOCAB_SIZE
    C = NUM_CODEBOOKS
    result = {}
    for k, v in weights.items():
        if k == "codebook_layer_offsets":
            continue
        v = v.astype(dtype)
        if k == "audio_embeddings.weight":
            for i in range(C):
                result[f"audio_embeddings.{i}.weight"] = v[i * V : (i + 1) * V]
        elif k == "audio_heads.weight":
            for i in range(C):
                result[f"audio_heads.{i}.weight"] = v[i * V : (i + 1) * V]
        elif k.startswith("llm."):
            result["backbone." + k[4:]] = v
        else:
            result[k] = v
    return result


def cast_tokenizer(weights: dict, dtype: mx.Dtype) -> dict:
    """Cast all tokenizer weights to target dtype.

    Deliberately keeps HuBERT / semantic model weights so that the
    PyTorch HiggsAudioV2TokenizerModel can load from this directory
    for voice-cloning encode().  MLX-side filtering happens in
    HiggsAudioTokenizer.sanitize() at load time.
    """
    return {k: v.astype(dtype) for k, v in weights.items()}


# ── IO helpers ────────────────────────────────────────────────────────────────


def _save_weights(weights: dict, path: Path):
    mx.save_safetensors(str(path), weights)
    size_mb = path.stat().st_size / 1e6
    print(f"  saved {path.name} ({size_mb:.0f} MB, {len(weights)} tensors)")


def _copy_file(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  copied {src.name}")


# ── main conversion ───────────────────────────────────────────────────────────


def convert(src: Path, out: Path, dtype: mx.Dtype):
    out.mkdir(parents=True, exist_ok=True)
    tok_out = out / "audio_tokenizer"
    tok_out.mkdir(exist_ok=True)

    # 1. Backbone
    print("Loading backbone weights…")
    backbone_weights = dict(mx.load(str(src / "model.safetensors")))
    print(f"  {len(backbone_weights)} keys")
    backbone_weights = remap_backbone(backbone_weights, dtype)
    _save_weights(backbone_weights, out / "model.safetensors")

    # 2. Audio tokenizer
    print("Loading audio tokenizer weights…")
    tok_weights = dict(mx.load(str(src / "audio_tokenizer" / "model.safetensors")))
    print(f"  {len(tok_weights)} keys")
    tok_weights = cast_tokenizer(tok_weights, dtype)
    _save_weights(tok_weights, tok_out / "model.safetensors")

    # 3. Config files
    print("Copying configs…")
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]:
        _copy_file(src / fname, out / fname)
    for fname in ["config.json", "preprocessor_config.json"]:
        _copy_file(src / "audio_tokenizer" / fname, tok_out / fname)

    # 4. Patch config.json dtype
    cfg_path = out / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["dtype"] = str(dtype).split(".")[-1]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nDone → {out}")
    return out


def upload(out: Path, repo_id: str, src_repo: str):
    from mlx_audio.tts.utils import upload_to_hub

    print(f"Uploading to {repo_id}…")
    upload_to_hub(str(out), repo_id, src_repo)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Convert OmniVoice to mlx-audio format"
    )
    parser.add_argument(
        "--model", default="k2-fsa/OmniVoice", help="HF repo ID or local path"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Target weight dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--upload",
        default=None,
        help="HF repo to upload to (e.g. mlx-community/OmniVoice-bf16)",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    src = Path(args.model)
    if not src.exists():
        from mlx_audio.utils import get_model_path

        print(f"Downloading {args.model} from HuggingFace…")
        src = get_model_path(args.model)

    out = convert(src, Path(args.output), dtype)

    if args.upload:
        upload(out, args.upload, args.model)


if __name__ == "__main__":
    main()
