import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_state_dict(path: Path) -> Dict[str, Any]:
    import torch

    sd = torch.load(str(path), map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(sd)}")
    return sd


def main():
    parser = argparse.ArgumentParser(
        description="Convert funasr/campplus CAMPPlus style encoder to MLX safetensors (IndexTTS2)"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="funasr/campplus",
        help="HuggingFace repo id containing campplus_cn_common.bin",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="campplus_cn_common.bin",
        help="Checkpoint filename in the HF repo",
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
        help="Output directory containing config.json + campplus.safetensors",
    )
    parser.add_argument(
        "--feat-dim",
        type=int,
        default=80,
        help="Input feature dimension (IndexTTS2 uses 80)",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=192,
        help="Output embedding size (IndexTTS2 uses 192)",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    import mlx.core as mx

    from mlx_audio.tts.models.chatterbox.s3gen.xvector import CAMPPlus

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(
        hf_hub_download(
            repo_id=args.hf_repo,
            filename=args.filename,
            revision=args.revision,
        )
    )

    state = _load_state_dict(ckpt_path)

    # Convert tensors to mx arrays.
    mx_weights = {}
    for k, v in state.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        mx_weights[k] = mx.array(v)

    camp = CAMPPlus(feat_dim=args.feat_dim, embedding_size=args.embedding_size)
    mx_weights = camp.sanitize(mx_weights)
    mx_weights = {f"campplus.{k}": v for k, v in mx_weights.items()}

    mx.save_safetensors(str(out_dir / "campplus.safetensors"), mx_weights)

    # Update or create config.json
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = {"model_type": "indextts2", "sample_rate": 22050}

    cfg["campplus"] = {"feat_dim": args.feat_dim, "embedding_size": args.embedding_size}
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"Saved MLX weights: {out_dir / 'campplus.safetensors'}")
    print(f"Saved config: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
