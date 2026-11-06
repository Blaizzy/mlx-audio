import argparse
import json
from pathlib import Path


def _official_2_5_weights(checkpoint: Path):
    import torch

    obj = torch.load(str(checkpoint), map_location="cpu")
    return obj.get("model", obj) if isinstance(obj, dict) else obj


def _load_safetensors(path: Path):
    import mlx.core as mx
    from safetensors import safe_open

    weights = {}
    with safe_open(str(path), framework="numpy") as handle:
        for key in handle.keys():
            weights[key] = mx.array(handle.get_tensor(key))
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Convert the IndexTTS2 / IndexTTS-2.5 EnhancedCodec to MLX"
    )
    parser.add_argument(
        "--version",
        choices=["2.0", "2.5"],
        default="2.5",
        help="2.5 reads codec.pth; 2.0 reads the MaskGCT safetensors checkpoint",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="IndexTeam/IndexTTS-2.5",
        help="HuggingFace repo id containing the codec checkpoint",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Override checkpoint filename (defaults to codec.pth or the MaskGCT safetensors)",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory containing config.json + semantic_codec.safetensors",
    )
    args = parser.parse_args()

    import mlx.core as mx
    from huggingface_hub import hf_hub_download

    from mlx_audio.tts.models.indextts2.semantic_codec import (
        EnhancedCodec,
        EnhancedCodecConfig,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = args.filename
    if filename is None:
        filename = (
            "codec.pth" if args.version == "2.5" else "semantic_codec/model.safetensors"
        )
    checkpoint = Path(
        hf_hub_download(
            repo_id=args.hf_repo,
            filename=filename,
            revision=args.revision,
        )
    )

    if args.version == "2.5":
        weights = {
            k: mx.array(v.detach().cpu().numpy())
            for k, v in _official_2_5_weights(checkpoint).items()
        }
    else:
        weights = _load_safetensors(checkpoint)

    cfg = EnhancedCodecConfig(downsample_scale=2 if args.version == "2.5" else 1)
    model = EnhancedCodec(cfg)
    weights = model.sanitize(weights)
    weights = {f"semantic_codec.{key}": value for key, value in weights.items()}
    mx.save_safetensors(str(out_dir / "semantic_codec.safetensors"), weights)

    cfg_path = out_dir / "config.json"
    root = (
        json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg_path.exists()
        else {"model_type": "indextts2", "sample_rate": 22050}
    )
    root["version"] = args.version
    root["model_type"] = "indextts2_5" if args.version == "2.5" else "indextts2"
    root["semantic_codec"] = {
        "codebook_size": cfg.codebook_size,
        "hidden_size": cfg.hidden_size,
        "codebook_dim": cfg.codebook_dim,
        "vocos_dim": cfg.vocos_dim,
        "vocos_intermediate_dim": cfg.vocos_intermediate_dim,
        "vocos_num_layers": cfg.vocos_num_layers,
        "num_quantizers": cfg.num_quantizers,
        "downsample_scale": cfg.downsample_scale,
    }
    cfg_path.write_text(json.dumps(root, indent=2), encoding="utf-8")

    print(f"Saved MLX weights: {out_dir / 'semantic_codec.safetensors'}")
    print(f"Saved config: {cfg_path}")


if __name__ == "__main__":
    main()
