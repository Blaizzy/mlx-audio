import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def _linear_strip_weightnorm(weight_g, weight_v, eps: float = 1e-8):
    # w = g * v / ||v||
    import numpy as np

    v = weight_v
    norm = np.sqrt(np.sum(v * v, axis=1, keepdims=True) + eps)
    g = weight_g.reshape(-1, 1)
    return v * (g / norm)


def _conv1d_strip_weightnorm(weight_g, weight_v, eps: float = 1e-8):
    # torch conv1d weight_v: (O, I, K)
    import numpy as np

    v = weight_v
    norm = np.sqrt(np.sum(v * v, axis=(1, 2), keepdims=True) + eps)
    g = weight_g.reshape(-1, 1, 1)
    return v * (g / norm)


def _transpose_conv1d_torch_to_mlx(w):
    # (O, I, K) -> (O, K, I)
    return w.transpose(0, 2, 1)


def _transpose_conv2d_torch_to_mlx(w):
    # (O, I, KH, KW) -> (O, KH, KW, I)
    return w.transpose(0, 2, 3, 1)


def main():
    parser = argparse.ArgumentParser(description="Convert IndexTTS2 s2mel.pth to MLX safetensors")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="IndexTeam/IndexTTS-2",
        help="HuggingFace repo id containing s2mel.pth and config.yaml",
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
        help="Output directory containing config.json + s2mel.safetensors",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    import torch
    import mlx.core as mx
    import yaml

    from mlx_audio.tts.models.indextts2.s2mel import S2MelConfig, S2MelModel

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(hf_hub_download(args.hf_repo, "config.yaml", revision=args.revision))
    ckpt_path = Path(hf_hub_download(args.hf_repo, "s2mel.pth", revision=args.revision))

    cfg_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    s2mel_cfg = S2MelConfig.from_dict(cfg_yaml["s2mel"])
    model = S2MelModel(s2mel_cfg)

    obj = torch.load(str(ckpt_path), map_location="cpu")
    sd = obj["net"]

    # Flatten into one dict matching MLX parameter names.
    # MLX expects weights for:
    # - s2mel.cfm.estimator.*
    # - s2mel.length_regulator.*
    # - s2mel.gpt_layer.*
    # The torch checkpoint stores: cfm.*, length_regulator.*, gpt_layer.*

    out: Dict[str, mx.array] = {}

    def add_module(prefix_out: str, module_sd: Dict[str, torch.Tensor]):
        # Normalize key names to match our MLX modules (strip SConv1d wrappers)
        norm_sd: Dict[str, torch.Tensor] = {}
        for k, v in module_sd.items():
            nk = k.replace(".conv.conv.", ".")
            norm_sd[nk] = v
        module_sd = norm_sd

        # Handle weightnorm pairs
        used = set()
        for k, v in module_sd.items():
            if k.endswith(".weight_g"):
                base = k[: -len(".weight_g")]
                v_key = base + ".weight_v"
                if v_key not in module_sd:
                    continue
                w_g = module_sd[k].detach().cpu().numpy()
                w_v = module_sd[v_key].detach().cpu().numpy()
                if w_v.ndim == 2:
                    w = _linear_strip_weightnorm(w_g, w_v)
                elif w_v.ndim == 3:
                    w = _conv1d_strip_weightnorm(w_g, w_v)
                elif w_v.ndim == 4:
                    # rare
                    w = w_v
                else:
                    w = w_v
                if w.ndim == 3:
                    w = _transpose_conv1d_torch_to_mlx(w)
                elif w.ndim == 4:
                    w = _transpose_conv2d_torch_to_mlx(w)
                out[prefix_out + base + ".weight"] = mx.array(w)
                used.add(k)
                used.add(v_key)

        for k, v in module_sd.items():
            if k in used:
                continue
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else v
            if getattr(arr, "ndim", 0) == 3:
                arr = _transpose_conv1d_torch_to_mlx(arr)
            elif getattr(arr, "ndim", 0) == 4:
                arr = _transpose_conv2d_torch_to_mlx(arr)
            out[prefix_out + k] = mx.array(arr)

    add_module("s2mel.cfm.", sd["cfm"])
    add_module("s2mel.length_regulator.", sd["length_regulator"])
    add_module("s2mel.gpt_layer.", sd["gpt_layer"])

    # Add derived / MLX-only parameters that are not stored in torch ckpt
    # 1) freqs_cis buffer for GPTFastTransformer
    head_dim = int(cfg_yaml["s2mel"]["DiT"].get("hidden_dim", 512)) // int(
        cfg_yaml["s2mel"]["DiT"].get("num_heads", 8)
    )
    seq_len = 16384
    base = 10000
    import numpy as np
    freqs = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(seq_len, dtype=np.float32)
    outer = np.outer(t, freqs)
    freqs_cis = np.stack([np.cos(outer), np.sin(outer)], axis=-1)
    out["s2mel.cfm.estimator.transformer.freqs_cis"] = mx.array(freqs_cis)

    # 2) FinalLayer LayerNorm affine params (MLX LayerNorm always has them)
    out["s2mel.cfm.estimator.final_layer.norm_final.weight"] = mx.ones((512,), dtype=mx.float32)
    out["s2mel.cfm.estimator.final_layer.norm_final.bias"] = mx.zeros((512,), dtype=mx.float32)

    mx.save_safetensors(str(out_dir / "s2mel.safetensors"), out)

    # Merge config.json
    cfg_out = out_dir / "config.json"
    if cfg_out.exists():
        root = json.loads(cfg_out.read_text(encoding="utf-8"))
    else:
        root = {"model_type": "indextts2", "sample_rate": 22050}

    root["s2mel"] = cfg_yaml["s2mel"]
    cfg_out.write_text(json.dumps(root, indent=2), encoding="utf-8")

    print(f"Saved MLX weights: {out_dir / 's2mel.safetensors'}")
    print(f"Saved config: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
