import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch


def convert_ace_step(model_repo: str, output_dir: str, local_files_only: bool = False):
    import mlx.core as mx
    import safetensors.torch
    from diffusers.models import AutoencoderOobleck
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_repo}...")
    local_dir = snapshot_download(model_repo, local_files_only=local_files_only)
    turbo_dir = Path(local_dir) / "acestep-v15-turbo"
    vae_dir = Path(local_dir) / "vae"
    text_dir = Path(local_dir) / "Qwen3-Embedding-0.6B"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw state dict to bypass PyTorch bugs...")
    state_dict = safetensors.torch.load_file(str(turbo_dir / "model.safetensors"))

    with open(turbo_dir / "config.json") as f:
        config_dict = json.load(f)

    weights = {}
    for key, value in state_dict.items():
        np_val = value.cpu().float().numpy()
        new_key = key

        # Skip rotary embedding caches (recomputed at runtime)
        if "rotary_emb" in key:
            continue

        # Handle decoder Conv1d proj_in: PyTorch [out, in, K] -> MLX [out, K, in]
        # Use underscore naming (proj_in_weight) to match bare params in DiTModel
        if "decoder.proj_in.1." in new_key:
            new_key = new_key.replace("proj_in.1.", "proj_in_")
            if new_key.endswith("_weight"):
                np_val = np_val.swapaxes(1, 2)

        # Handle decoder ConvTranspose1d proj_out: PyTorch [in, out, K] -> MLX [out, K, in]
        elif "decoder.proj_out.1." in new_key:
            new_key = new_key.replace("proj_out.1.", "proj_out_")
            if new_key.endswith("_weight"):
                np_val = np_val.transpose(1, 2, 0)

        weights[new_key] = mx.array(np_val)

    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)

    print("Loading VAE...")
    pt_vae = AutoencoderOobleck.from_pretrained(str(vae_dir))
    vae_weights = {}

    for key, value in pt_vae.state_dict().items():
        np_val = value.detach().cpu().float().numpy()
        vae_weights[key] = mx.array(np_val)

    vae_out_dir = out_dir / "vae"
    vae_out_dir.mkdir(exist_ok=True)
    mx.save_safetensors(
        str(vae_out_dir / "diffusion_pytorch_model.safetensors"), vae_weights
    )
    shutil.copy2(vae_dir / "config.json", vae_out_dir / "config.json")

    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    silence_path = turbo_dir / "silence_latent.pt"
    if silence_path.exists():
        pt_silence = torch.load(silence_path, map_location="cpu", weights_only=True)
        np.save(out_dir / "silence_latent.npy", pt_silence.numpy())

    print("Copying Qwen3 text encoder...")
    if text_dir.exists():
        text_out_dir = out_dir / "Qwen3-Embedding-0.6B"
        text_out_dir.mkdir(exist_ok=True)

        text_state = safetensors.torch.load_file(str(text_dir / "model.safetensors"))
        text_weights = {}
        for key, value in text_state.items():
            clean_key = key[6:] if key.startswith("model.") else key
            text_weights[clean_key] = mx.array(value.cpu().float().numpy())

        mx.save_safetensors(str(text_out_dir / "model.safetensors"), text_weights)

        for fname in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ]:
            src = text_dir / fname
            if src.exists():
                shutil.copy2(src, text_out_dir / fname)

    print(f"Success! Wrote converted files to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_repo", nargs="?", default="ACE-Step/Ace-Step1.5")
    parser.add_argument("output_dir", nargs="?", default="/tmp/ace_step-mlx-converted")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    convert_ace_step(args.model_repo, args.output_dir, args.local_files_only)


if __name__ == "__main__":
    main()
