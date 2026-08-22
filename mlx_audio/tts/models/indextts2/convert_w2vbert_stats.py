import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert wav2vec2-bert stats (mean/var) to MLX safetensors (IndexTTS2)"
    )
    parser.add_argument(
        "--stats-pt",
        type=str,
        required=True,
        help="Path to wav2vec2bert_stats.pt (torch file with mean/var)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory containing config.json + w2vbert_stats.safetensors",
    )
    args = parser.parse_args()

    import torch
    import mlx.core as mx

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(args.stats_pt, map_location="cpu")
    if not isinstance(stats, dict) or "mean" not in stats or "var" not in stats:
        raise ValueError("Expected a dict with keys 'mean' and 'var'")

    mean = stats["mean"].detach().cpu().numpy()
    var = stats["var"].detach().cpu().numpy()

    weights = {
        "w2vbert_stats.mean": mx.array(mean),
        "w2vbert_stats.std": mx.sqrt(mx.array(var)),
    }

    mx.save_safetensors(str(out_dir / "w2vbert_stats.safetensors"), weights)

    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = {"model_type": "indextts2", "sample_rate": 22050}

    cfg["w2vbert_stats"] = {
        "mean": "w2vbert_stats.safetensors::w2vbert_stats.mean",
        "std": "w2vbert_stats.safetensors::w2vbert_stats.std",
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"Saved stats: {out_dir / 'w2vbert_stats.safetensors'}")
    print(f"Saved config: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
