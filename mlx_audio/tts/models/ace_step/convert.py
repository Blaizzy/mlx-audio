import sys
import os
import torch
import transformers
import dataclasses

def convert_ace_step(model_repo, output_dir):
    import mlx.core as mx
    import safetensors.torch
    import json
    from mlx_audio.tts.models.ace_step.config import ModelConfig
    from diffusers.models import AutoencoderOobleck
    
    print(f"Downloading {model_repo}...")
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(model_repo)
    # The models are nested in a subfolder for ACE-Step
    turbo_dir = os.path.join(local_dir, "acestep-v15-turbo")
    
    print("Loading raw state dict to bypass PyTorch bugs...")
    state_dict = safetensors.torch.load_file(os.path.join(turbo_dir, "model.safetensors"))
    
    with open(os.path.join(turbo_dir, "config.json")) as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)
    
    weights = []
    # Extract Decoder manually
    for key, value in state_dict.items():
        if not key.startswith("decoder.") and not key.startswith("encoder."): continue
        np_val = value.cpu().float().numpy()
        
        if key.startswith("decoder."):
            new_key = key.replace("decoder.", "dit.")
            if "proj_in.1." in new_key:
                new_key = new_key.replace("proj_in.1.", "proj_in.")
                if new_key.endswith(".weight"):
                    np_val = np_val.swapaxes(1, 2)
            elif "proj_out.1." in new_key:
                new_key = new_key.replace("proj_out.1.", "proj_out.")
                if new_key.endswith(".weight"):
                    np_val = np_val.transpose(1, 2, 0)
            elif "rotary_emb" in new_key:
                continue
            weights.append((new_key, mx.array(np_val)))
            
        elif key.startswith("encoder."):
            new_key = key
            if "rotary_emb" in new_key:
                continue
            weights.append((new_key, mx.array(np_val)))
            
    out_path = output_dir
    mx.save_safetensors(os.path.join(out_path, "model.safetensors"), dict(weights))
    
    print("Loading VAE...")
    pt_vae = AutoencoderOobleck.from_pretrained(f"{local_dir}/vae")
    vae_weights = []
    
    import numpy as np
    def _fuse_weight_norm(weight_g, weight_v, eps=1e-9):
        v_flat = weight_v.reshape(weight_v.shape[0], -1)
        norm = np.linalg.norm(v_flat, axis=1).reshape(weight_g.shape)
        return weight_v * (weight_g / np.maximum(norm, eps))
        
    vae_state = pt_vae.state_dict()
    processed = set()
    all_keys = sorted(vae_state.keys())
    for key in all_keys:
        if key in processed: continue
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            v_key = base + ".weight_v"
            g = vae_state[key].detach().cpu().float().numpy()
            v = vae_state[v_key].detach().cpu().float().numpy()
            w = _fuse_weight_norm(g, v)
            if "conv_t1" in base: w = w.transpose(1, 2, 0)
            else: w = w.swapaxes(1, 2)
            vae_weights.append((base + ".weight", mx.array(w)))
            processed.add(key)
            processed.add(v_key)
            continue
        if key.endswith(".weight_v"): continue
        val = vae_state[key].detach().cpu().float().numpy()
        if key.endswith(".alpha") or key.endswith(".beta"):
            val = val.squeeze()
        if "conv" in key and key.endswith(".weight"):
            if "conv_t1" in key:
                val = val.transpose(1, 2, 0)
            else:
                val = val.swapaxes(1, 2)
        vae_weights.append((key, mx.array(val)))
        processed.add(key)
            
    mx.save_safetensors(os.path.join(out_path, "vae.safetensors"), dict(vae_weights))
    
    with open(os.path.join(out_path, "config.json"), "w") as f:
        # Save raw dict since we are bypassing to_dict
        json.dump(config_dict, f, indent=4)
        
    silence_path = os.path.join(turbo_dir, "silence_latent.pt")
    if os.path.exists(silence_path):
        pt_silence = torch.load(silence_path, map_location="cpu", weights_only=True)
        np.save(os.path.join(out_path, "silence_latent.npy"), pt_silence.numpy())

    print("Success!")

print("Starting conversion...")
convert_ace_step("ACE-Step/Ace-Step1.5", "/tmp/ace_step-mlx-converted")
