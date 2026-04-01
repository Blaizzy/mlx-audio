"""Convert MeloTTS-English PyTorch weights to MLX format."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def merge_weight_norm(weight_g, weight_v):
    """Merge weight_norm decomposition: weight = g * v / ||v||."""
    norm = np.sqrt(
        (weight_v**2).sum(axis=tuple(range(1, weight_v.ndim)), keepdims=True)
    )
    return weight_g * weight_v / (norm + 1e-8)


def is_conv_transpose_key(key):
    """Check if a weight key belongs to a ConvTranspose1d layer."""
    return bool(re.search(r"dec\.ups\.\d+\.weight", key))


def convert_melotts(output_dir: str, model_repo: str = "myshell-ai/MeloTTS-English"):
    import torch
    from huggingface_hub import hf_hub_download

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_repo} checkpoint...")
    ckpt_path = hf_hub_download(model_repo, "checkpoint.pth")
    config_path = hf_hub_download(model_repo, "config.json")

    with open(config_path) as f:
        config = json.load(f)

    mlx_config = {
        "model_type": "melotts",
        **config["data"],
        **config["model"],
        "num_tones": config["num_tones"],
        "num_languages": config["num_languages"],
        "n_vocab": len(config["symbols"]),
        "symbols": config["symbols"],
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(mlx_config, f, indent=2)
    print(f"Saved config.json ({len(config['symbols'])} symbols)")

    print("Loading PyTorch checkpoint...")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    weights = state["model"]

    sanitized = {}
    skip_prefixes = ("net_dur_disc", "net_d")

    for key in sorted(weights.keys()):
        if any(key.startswith(p) for p in skip_prefixes):
            continue

        value = weights[key].numpy()
        new_key = key

        new_key = new_key.replace("flow.flows.", "flow_layers.")

        if new_key.endswith(".weight_g"):
            continue

        if new_key.endswith(".weight_v"):
            base_key = key[: -len(".weight_v")]
            g_key = base_key + ".weight_g"
            if g_key in weights:
                wg = weights[g_key].numpy()
                wv = value
                merged = merge_weight_norm(wg, wv)
                new_base = new_key[: -len(".weight_v")]
                sanitized[new_base + ".weight"] = merged
            else:
                sanitized[new_key] = value
            continue

        if new_key.endswith(".gamma"):
            new_key = new_key[:-6] + ".weight"
        elif new_key.endswith(".beta"):
            new_key = new_key[:-5] + ".bias"

        sanitized[new_key] = value

    remapped = {}
    for key, value in sanitized.items():
        if value.ndim == 3 and key.endswith(".weight"):
            if is_conv_transpose_key(key):
                new_val = np.transpose(value, (1, 2, 0))
                new_key = key.replace(".weight", ".conv_t.weight")
                remapped[new_key] = new_val
            else:
                new_val = np.transpose(value, (0, 2, 1))
                new_key = key.replace(".weight", ".conv.weight")
                remapped[new_key] = new_val
        elif value.ndim == 1 and key.endswith(".bias"):
            weight_key = key[:-5] + ".weight"
            if weight_key in sanitized and sanitized[weight_key].ndim == 3:
                if is_conv_transpose_key(weight_key):
                    new_key = key.replace(".bias", ".conv_t.bias")
                else:
                    new_key = key.replace(".bias", ".conv.bias")
                remapped[new_key] = value
            else:
                remapped[key] = value
        else:
            remapped[key] = value
    sanitized = remapped

    from safetensors.numpy import save_file as save_safetensors

    save_safetensors(sanitized, str(output_path / "model.safetensors"))
    print(f"Saved {len(sanitized)} weight tensors to model.safetensors")

    print("\nConverting BERT (bert-base-uncased)...")
    convert_bert(output_path)

    print(f"\nDone! Model saved to {output_path}")
    print(f"To use: mlx_audio.tts.utils.load('{output_path}')")


def convert_bert(output_path: Path):
    """Convert bert-base-uncased to MLX format."""
    from transformers import AutoModel

    print("Loading bert-base-uncased...")
    bert = AutoModel.from_pretrained("bert-base-uncased")
    state = bert.state_dict()

    def remap_key(k):
        k = k.replace(".layer.", ".layers.")
        k = k.replace(".self.key.", ".key_proj.")
        k = k.replace(".self.query.", ".query_proj.")
        k = k.replace(".self.value.", ".value_proj.")
        k = k.replace(".attention.output.dense.", ".attention.out_proj.")
        k = k.replace(".attention.output.LayerNorm.", ".ln1.")
        k = k.replace(".output.LayerNorm.", ".ln2.")
        k = k.replace(".intermediate.dense.", ".linear1.")
        k = k.replace(".output.dense.", ".linear2.")
        k = k.replace("embeddings.LayerNorm.", "embeddings.norm.")
        k = k.replace("pooler.dense.", "pooler.")
        return k

    bert_weights = {}
    for key, value in state.items():
        if "position_ids" in key:
            continue
        new_key = remap_key(key)
        bert_weights[new_key] = value.numpy()

    np.savez(str(output_path / "bert_weights.npz"), **bert_weights)
    print(f"Saved {len(bert_weights)} BERT weight tensors to bert_weights.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MeloTTS to MLX")
    parser.add_argument(
        "--model",
        type=str,
        default="myshell-ai/MeloTTS-English",
        help="HuggingFace repo ID (e.g. myshell-ai/MeloTTS-English-v3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mlx-melotts-english",
        help="Output directory for MLX weights",
    )
    args = parser.parse_args()
    convert_melotts(args.output_dir, model_repo=args.model)
