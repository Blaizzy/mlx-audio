import argparse
import logging
from pathlib import Path
from typing import Dict
import numpy as np

import mlx.core as mx
from huggingface_hub import snapshot_download

from .dit import MLXDiTDecoder
from .vae import MLXAutoEncoderOobleck
from .conditioner import MLXAceStepConditionEncoder
from .config import AceStepConfig

logger = logging.getLogger(__name__)

def convert_acestep(model_repo: str, output_dir: str):
    """
    Download the PyTorch weights from HuggingFace and convert them to pure MLX format.
    ACE-Step provides MLX definitions but distributes weights in PT format.
    We run their `convert_and_load` internally and save out the clean `.safetensors`.
    """
    logger.info(f"Downloading {model_repo}...")
    local_dir = snapshot_download(model_repo)
    
    # We must load the PT model to extract its state dict properly 
    # since ACE-Step uses some complex nn.Sequential unpacking.
    import torch
    from transformers import AutoModel
    from diffusers.models import AutoencoderOobleck
    
    logger.info("Loading PyTorch AutoModel for DiT...")
    pt_model = AutoModel.from_pretrained(local_dir, trust_remote_code=True)
    
    logger.info("Initializing MLX Models...")
    config = AceStepConfig.from_dict(pt_model.config.to_dict())
    mlx_dit = MLXDiTDecoder.from_config(pt_model.config)
    mlx_encoder = MLXAceStepConditionEncoder(pt_model.config)
    
    logger.info("Converting DiT weights to MLX...")
    weights = []
    
    # Extract Decoder
    for key, value in pt_model.decoder.state_dict().items():
        np_val = value.detach().cpu().float().numpy()
        new_key = "dit." + key
        if key.startswith("proj_in.1."):
            new_key = "dit." + key.replace("proj_in.1.", "proj_in.")
            if new_key.endswith(".weight"):
                np_val = np_val.swapaxes(1, 2)
        elif key.startswith("proj_out.1."):
            new_key = "dit." + key.replace("proj_out.1.", "proj_out.")
            if new_key.endswith(".weight"):
                np_val = np_val.transpose(1, 2, 0)
        elif "rotary_emb" in key:
            continue
        weights.append((new_key, mx.array(np_val)))
        
    # Extract ConditionEncoder
    for key, value in pt_model.encoder.state_dict().items():
        np_val = value.detach().cpu().float().numpy()
        new_key = "encoder." + key
        if "rotary_emb" in key:
            continue
        weights.append((new_key, mx.array(np_val)))
        
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_path / "model.safetensors"), dict(weights))
    
    # Now convert VAE
    logger.info("Loading PyTorch VAE...")
    pt_vae = AutoencoderOobleck.from_pretrained(f"{local_dir}/vae")
    
    logger.info("Converting VAE weights to MLX...")
    mlx_vae = MLXAutoEncoderOobleck.from_pytorch_config(pt_vae)
    vae_weights = []
    
    def _fuse_weight_norm(weight, g, dim):
        norm = np.sqrt(np.sum(weight**2, axis=dim, keepdims=True))
        return weight * (g / norm)
        
    state_dict = pt_vae.state_dict()
    for key, value in state_dict.items():
        if key.endswith(".weight_g"):
            continue
        if key.endswith(".weight_v"):
            prefix = key[:-9]
            weight_v = value.detach().cpu().numpy()
            weight_g = state_dict[f"{prefix}.weight_g"].detach().cpu().numpy()
            dim = 0 if "conv" in prefix else 1
            fused = _fuse_weight_norm(weight_v, weight_g, dim)
            if "conv" in prefix:
                fused = fused.transpose(0, 2, 1)
            vae_weights.append((f"{prefix}.weight", mx.array(fused)))
        else:
            np_val = value.detach().cpu().numpy()
            if "conv" in key and key.endswith(".weight"):
                np_val = np_val.transpose(0, 2, 1)
            vae_weights.append((key, mx.array(np_val)))
            
    mlx_vae.load_weights(vae_weights)
    mx.eval(mlx_vae.parameters())
    
    # Save MLX VAE
    mx.save_safetensors(str(out_path / "vae.safetensors"), dict(vae_weights))
    
    # Save Config
    import json
    with open(out_path / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=4)
        
    # We should also copy over silence_latent.pt as numpy so it doesn't need PyTorch
    silence_path = Path(local_dir) / "silence_latent.pt"
    if silence_path.exists():
        import torch
        pt_silence = torch.load(silence_path, map_location="cpu", weights_only=True)
        np.save(out_path / "silence_latent.npy", pt_silence.numpy())
        
    logger.info(f"Conversion complete! MLX model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ACE-Step/acestep-v15-turbo")
    parser.add_argument("--output", default="acestep-mlx")
    args = parser.parse_args()
    convert_acestep(args.model, args.output)
