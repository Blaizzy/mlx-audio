import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert MaskGCT semantic codec (RepCodec) weights to MLX safetensors"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="amphion/MaskGCT",
        help="HuggingFace repo id containing semantic_codec/model.safetensors",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="semantic_codec/model.safetensors",
        help="Path to the safetensors file within the HF repo",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision/commit",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory containing config.json + semantic_codec.safetensors",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    import mlx.core as mx
    from safetensors import safe_open

    from mlx_audio.tts.models.indextts2.semantic_codec import RepCodec, RepCodecConfig

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    st_path = Path(
        hf_hub_download(
            repo_id=args.hf_repo,
            filename=args.filename,
            revision=args.revision,
        )
    )

    weights = {}
    with safe_open(str(st_path), framework="numpy") as f:
        for k in f.keys():
            weights[k] = mx.array(f.get_tensor(k))

    # Default config matches IndexTTS2 official config.yaml
    cfg = RepCodecConfig()
    model = RepCodec(cfg)

    weights = model.sanitize(weights)
    weights = {f"semantic_codec.{k}": v for k, v in weights.items()}

    mx.save_safetensors(str(out_dir / "semantic_codec.safetensors"), weights)

    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg_json = {"model_type": "indextts2", "sample_rate": 22050}

    cfg_json["semantic_codec"] = {
        "codebook_size": cfg.codebook_size,
        "hidden_size": cfg.hidden_size,
        "codebook_dim": cfg.codebook_dim,
        "vocos_dim": cfg.vocos_dim,
        "vocos_intermediate_dim": cfg.vocos_intermediate_dim,
        "vocos_num_layers": cfg.vocos_num_layers,
        "num_quantizers": cfg.num_quantizers,
        "downsample_scale": cfg.downsample_scale,
    }
    cfg_path.write_text(json.dumps(cfg_json, indent=2), encoding="utf-8")

    print(f"Saved MLX weights: {out_dir / 'semantic_codec.safetensors'}")
    print(f"Saved config: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
