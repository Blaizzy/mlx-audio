import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert IndexTTS feat1.pt/feat2.pt emotion matrices to safetensors"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing feat1.pt and feat2.pt",
    )
    parser.add_argument(
        "--spk-matrix",
        type=str,
        default=None,
        help="Optional override for feat1.pt",
    )
    parser.add_argument(
        "--emo-matrix",
        type=str,
        default=None,
        help="Optional override for feat2.pt",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    import mlx.core as mx
    import torch

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir) if args.output_dir else model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    spk_path = Path(args.spk_matrix) if args.spk_matrix else model_dir / "feat1.pt"
    emo_path = Path(args.emo_matrix) if args.emo_matrix else model_dir / "feat2.pt"
    spk = torch.load(str(spk_path), map_location="cpu").detach().float().numpy()
    emo = torch.load(str(emo_path), map_location="cpu").detach().float().numpy()
    if spk.shape[-1] != 192 or emo.shape[-1] != 1280 or spk.shape[0] != emo.shape[0]:
        raise ValueError(
            f"unexpected emotion matrix shapes: spk={spk.shape}, emo={emo.shape}"
        )

    output = out_dir / "emotion_matrices.safetensors"
    mx.save_safetensors(
        str(output),
        {
            "spk_matrix": mx.array(spk),
            "emo_matrix": mx.array(emo),
        },
    )

    cfg_path = out_dir / "config.json"
    cfg = (
        json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg_path.exists()
        else {"model_type": "indextts2", "sample_rate": 22050}
    )
    cfg["emotion_matrices"] = output.name
    cfg.setdefault("emo_num", [3, 17, 2, 8, 4, 5, 10, 24])
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Saved emotion matrices: {output}")
    print(f"Updated config: {cfg_path}")


if __name__ == "__main__":
    main()
