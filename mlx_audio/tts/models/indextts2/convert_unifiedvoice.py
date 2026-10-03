import argparse
import json
from pathlib import Path


def _transpose_conv1d(w):
    # torch (O, I, K) -> mlx (O, K, I)
    return w.transpose(0, 2, 1)


def _transpose_conv2d(w):
    # torch (O, I, KH, KW) -> mlx (O, KH, KW, I)
    return w.transpose(0, 2, 3, 1)


def main():
    parser = argparse.ArgumentParser(description="Convert IndexTTS2 gpt.pth (UnifiedVoice) to MLX safetensors")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="IndexTeam/IndexTTS-2",
        help="HuggingFace repo id containing gpt.pth and config.yaml",
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
        help="Output directory containing config.json + unifiedvoice.safetensors + bpe.model",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    import torch
    import mlx.core as mx
    import yaml

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(hf_hub_download(args.hf_repo, "config.yaml", revision=args.revision))
    ckpt_path = Path(hf_hub_download(args.hf_repo, "gpt.pth", revision=args.revision))
    bpe_path = Path(hf_hub_download(args.hf_repo, "bpe.model", revision=args.revision))

    cfg_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    gpt_cfg = cfg_yaml["gpt"]

    # Copy bpe.model alongside config for local loading
    (out_dir / "bpe.model").write_bytes(bpe_path.read_bytes())

    sd = torch.load(str(ckpt_path), map_location="cpu")

    weights = {}
    for k, v in sd.items():
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else v

        # Conv weights
        if arr.ndim == 3 and ("conv" in k or "depthwise_conv" in k or "pointwise_conv" in k):
            arr = _transpose_conv1d(arr)
        if arr.ndim == 4 and "conv" in k:
            arr = _transpose_conv2d(arr)

        # GPT2 (mlx-lm) expects transposed Linear weights in several places.
        if ".attn.c_attn.weight" in k or ".attn.c_proj.weight" in k or ".mlp.c_fc.weight" in k or ".mlp.c_proj.weight" in k:
            if arr.ndim == 2:
                arr = arr.transpose(1, 0)

        weights[f"unifiedvoice.{k}"] = mx.array(arr)

    mx.save_safetensors(str(out_dir / "unifiedvoice.safetensors"), weights)

    # Merge config.json
    cfg_out = out_dir / "config.json"
    if cfg_out.exists():
        root = json.loads(cfg_out.read_text(encoding="utf-8"))
    else:
        root = {"model_type": "indextts2", "sample_rate": 22050}

    root["unifiedvoice"] = gpt_cfg
    root["unifiedvoice"]["bpe_model"] = "bpe.model"
    cfg_out.write_text(json.dumps(root, indent=2), encoding="utf-8")

    print(f"Saved MLX weights: {out_dir / 'unifiedvoice.safetensors'}")
    print(f"Saved BPE: {out_dir / 'bpe.model'}")
    print(f"Saved config: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
