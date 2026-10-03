import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_torch_checkpoint(path: Path) -> Dict[str, Any]:
    import torch

    ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
    if "generator" in ckpt and isinstance(ckpt["generator"], dict):
        return ckpt["generator"]
    # Some checkpoints may be raw state_dict
    return ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA BigVGAN v2 generator to MLX safetensors (IndexTTS2 vocoder)"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="nvidia/bigvgan_v2_22khz_80band_256x",
        help="HuggingFace repo id of the BigVGAN model",
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
        help="Output directory containing config.json + model.safetensors",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    import mlx.core as mx

    from mlx_audio.codec.models.bigvgan.bigvgan import BigVGAN, BigVGANConfig

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(
        hf_hub_download(
            repo_id=args.hf_repo,
            filename="config.json",
            revision=args.revision,
        )
    )
    gen_path = Path(
        hf_hub_download(
            repo_id=args.hf_repo,
            filename="bigvgan_generator.pt",
            revision=args.revision,
        )
    )

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    vocoder_cfg = BigVGANConfig(
        num_mels=int(cfg["num_mels"]),
        upsample_rates=list(map(int, cfg["upsample_rates"])),
        upsample_kernel_sizes=list(map(int, cfg["upsample_kernel_sizes"])),
        upsample_initial_channel=int(cfg["upsample_initial_channel"]),
        resblock=str(cfg["resblock"]),
        resblock_kernel_sizes=list(map(int, cfg["resblock_kernel_sizes"])),
        resblock_dilation_sizes=cfg["resblock_dilation_sizes"],
        activation=str(cfg["activation"]),
        snake_logscale=bool(cfg["snake_logscale"]),
        use_bias_at_final=bool(cfg.get("use_bias_at_final", True)),
        use_tanh_at_final=bool(cfg.get("use_tanh_at_final", True)),
    )

    state = _load_torch_checkpoint(gen_path)

    # Convert tensors to mx arrays.
    mx_weights = {}
    for k, v in state.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        mx_weights[k] = mx.array(v)

    # Sanitize (transpose conv kernels for MLX layout where needed).
    model = BigVGAN(vocoder_cfg)
    mx_weights = model.sanitize(mx_weights)

    # Prefix for embedding under IndexTTS2 Model(bigvgan=...)
    mx_weights = {f"bigvgan.{k}": v for k, v in mx_weights.items()}

    # Save weights
    mx.save_safetensors(str(out_dir / "bigvgan.safetensors"), mx_weights)

    # Merge into (or create) an mlx-audio config.json
    cfg_out_path = out_dir / "config.json"
    if cfg_out_path.exists():
        out_cfg = json.loads(cfg_out_path.read_text(encoding="utf-8"))
    else:
        out_cfg = {}

    out_cfg["model_type"] = "indextts2"
    out_cfg.setdefault("sample_rate", int(cfg.get("sampling_rate", 22050)))
    out_cfg["vocoder"] = {
        "num_mels": vocoder_cfg.num_mels,
        "upsample_rates": vocoder_cfg.upsample_rates,
        "upsample_kernel_sizes": vocoder_cfg.upsample_kernel_sizes,
        "upsample_initial_channel": vocoder_cfg.upsample_initial_channel,
        "resblock": vocoder_cfg.resblock,
        "resblock_kernel_sizes": vocoder_cfg.resblock_kernel_sizes,
        "resblock_dilation_sizes": vocoder_cfg.resblock_dilation_sizes,
        "activation": vocoder_cfg.activation,
        "snake_logscale": vocoder_cfg.snake_logscale,
        "use_bias_at_final": vocoder_cfg.use_bias_at_final,
        "use_tanh_at_final": vocoder_cfg.use_tanh_at_final,
    }
    cfg_out_path.write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")

    print(f"Saved MLX weights: {out_dir / 'bigvgan.safetensors'}")
    print(f"Saved config: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
