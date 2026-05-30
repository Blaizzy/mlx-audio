#!/usr/bin/env python3
"""Dump the Mega-ASR audio-quality router state dict (keys -> shapes).

The released checkpoint metadata is unreliable for architecture; the tensor
names/shapes here are the source of truth for the MLX port (Phase 2).

Usage:
    python dump_router_keys.py [path/to/best_acc_model.safetensors]

If no path is given, falls back to the HF cache copy of
`zhifeixie/Mega-ASR/audio_quality_router/best_acc_model.safetensors`.
Writes `router_state_dict_keys.json` next to this script and prints a summary
(d_model, num transformer layers, conv layout, classifier head dims).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def find_default_checkpoint() -> str | None:
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            "zhifeixie/Mega-ASR",
            "audio_quality_router/best_acc_model.safetensors",
        )
    except Exception:
        return None


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else find_default_checkpoint()
    if not path:
        print("No checkpoint path given and HF download unavailable.", file=sys.stderr)
        return 1

    from safetensors import safe_open

    shapes: dict[str, list[int]] = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            shapes[key] = list(f.get_slice(key).get_shape())

    out = Path(__file__).with_name("router_state_dict_keys.json")
    out.write_text(json.dumps(dict(sorted(shapes.items())), indent=2))

    n_layers = len(
        {k.split(".")[2] for k in shapes if k.startswith("transformer.layers.")}
    )
    d_model = shapes.get("transformer.norm.weight", [None])[0]
    print(f"checkpoint: {path}")
    print(f"tensors: {len(shapes)}")
    print(f"d_model (transformer.norm): {d_model}")
    print(f"transformer layers: {n_layers}")
    print(
        f"conv.0: {shapes.get('frontend.conv.0.weight')}  conv.4: {shapes.get('frontend.conv.4.weight')}"
    )
    print(
        f"classifier.0: {shapes.get('classifier.0.weight')}  classifier.3: {shapes.get('classifier.3.weight')}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
