import json
from pathlib import Path
from typing import Optional, List, Union, Generator, Tuple

import mlx.core as mx
import mlx.nn as nn
import time
from transformers import AutoTokenizer

from ..base import GenerationResult
from .config import ModelArgs, LMConfig, AudioVAEConfig
from .minicpm import MiniCPMModel
from .encoder import VoxCPMLocEnc
from .dit import UnifiedCFM, VoxCPMLocDiT
from .audio_vae import AudioVAE

class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, latent_dim: int = 64, scale: int = 9):
        super().__init__()
        self.scale = scale
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)

    def __call__(self, x):
        x = self.in_proj(x)
        x = mx.tanh(x)
        # Rounding
        x = mx.round(x * self.scale) / self.scale
        return self.out_proj(x)

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.patch_size = args.patch_size
        self.feat_dim = args.feat_dim
        
        # LM Backbone
        self.base_lm = MiniCPMModel(args.lm_config)
        
        # Residual LM (vocab_size=0)
        res_config = LMConfig(**vars(args.lm_config))
        res_config.num_hidden_layers = args.residual_lm_num_layers
        res_config.vocab_size = 0
        self.residual_lm = MiniCPMModel(res_config)
        
        # Encoder
        enc_config = LMConfig(**vars(args.lm_config))
        enc_config.hidden_size = args.encoder_config.hidden_dim
        enc_config.intermediate_size = args.encoder_config.ffn_dim
        enc_config.num_attention_heads = args.encoder_config.num_heads
        enc_config.num_hidden_layers = args.encoder_config.num_layers
        enc_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(enc_config, input_dim=args.feat_dim)
        
        # DiT / CFM
        dit_config = LMConfig(**vars(args.lm_config)) # base on LM config but override
        dit_config.hidden_size = args.dit_config.hidden_dim
        dit_config.intermediate_size = args.dit_config.ffn_dim
        dit_config.num_attention_heads = args.dit_config.num_heads
        dit_config.num_hidden_layers = args.dit_config.num_layers
        dit_config.vocab_size = 0
        
        estimator = VoxCPMLocDiT(dit_config, in_channels=args.feat_dim)
        self.feat_decoder = UnifiedCFM(
            in_channels=args.feat_dim,
            cfm_params=args.dit_config.cfm_config,
            estimator=estimator
        )
        
        # Projections
        self.fsq_layer = ScalarQuantizationLayer(
            args.lm_config.hidden_size,
            args.lm_config.hidden_size,
            args.scalar_quantization_latent_dim,
            args.scalar_quantization_scale
        )
        
        self.enc_to_lm_proj = nn.Linear(args.encoder_config.hidden_dim, args.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(args.lm_config.hidden_size, args.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(args.lm_config.hidden_size, args.dit_config.hidden_dim)
        
        # Stop Predictor
        self.stop_proj = nn.Linear(args.lm_config.hidden_size, args.lm_config.hidden_size)
        self.stop_head = nn.Linear(args.lm_config.hidden_size, 2, bias=False)
        
        # Audio VAE
        self.audio_vae = AudioVAE(args.audio_vae_config)
        
        # Placeholder for tokenizer
        self.tokenizer = None

    def sanitize(self, weights: dict):
        from mlx.utils import tree_flatten
        
        # Track whether VAE weights were already sanitized to prevent double processing
        vae_already_sanitized = False
        
        # 0. Check if audio_vae weights are present. If not, try to load from pth
        has_vae = any(k.startswith("audio_vae.") for k in weights.keys())
        if not has_vae and self.args.model_path:
            p = Path(self.args.model_path) / "audiovae.pth"
            print(f"[DEBUG] Looking for audiovae at: {p}")
            if p.exists():
                try:
                    import torch
                    state = torch.load(p, map_location="cpu")
                    if "state_dict" in state:
                        state = state["state_dict"]
                    
                    print(f"[DEBUG] Loaded {len(state)} keys from audiovae.pth")
                    
                    # Convert to numpy/mlx and prefix
                    vae_weights_pth = {}
                    for k, v in state.items():
                        if k.startswith("module."):
                            k = k[7:]
                        # v is tensor
                        arr = mx.array(v.numpy())
                        vae_weights_pth[k] = arr
                        
                    # Sanitize these VAE weights
                    sanitized_vae = self.audio_vae.sanitize(vae_weights_pth)
                    
                    # Add to main weights
                    for k, v in sanitized_vae.items():
                         weights[f"audio_vae.{k}"] = v
                    
                    # Mark that VAE weights have been sanitized
                    vae_already_sanitized = True
                         
                except ImportError:
                    print(f"Warning: torch not installed, skipping loading {p}")
                except Exception as e:
                    print(f"Error loading {p}: {e}")

        # Delegate AudioVAE sanitize (if keys exist now and not already sanitized)
        # Extract audio_vae weights
        vae_weights = {k: v for k, v in weights.items() if k.startswith("audio_vae.")}
        # Strip prefix
        vae_weights_stripped = {k[len("audio_vae."):]: v for k, v in vae_weights.items()}
        
        if vae_weights_stripped and not vae_already_sanitized:
             # Sanitize VAE
             sanitized_vae = self.audio_vae.sanitize(vae_weights_stripped)
             # Put back
             for k in list(vae_weights.keys()):
                 del weights[k]
             for k, v in sanitized_vae.items():
                 weights[f"audio_vae.{k}"] = v
                 
        # General sanitization for other layers (transpose Linear if needed?)
        # MLX Linear: (out, in). PyTorch: (out, in). Matched.
        # But wait, MLX nn.Linear weight is (in, out) if using library?
        # Checking MLX docs: `nn.Linear(input_dims, output_dims)`. Weight shape `(input_dims, output_dims)`.
        # PyTorch `nn.Linear(in_features, out_features)`. Weight shape `(out_features, in_features)`.
        # So we NEED to transpose all Linear weights!
        
        new_weights = {}
        curr_shapes = {k: v.shape for k, v in tree_flatten(self.parameters())}
        
        for k, v in weights.items():
            if k not in curr_shapes:
                # might be skipped params or structure mismatch
                # keep it for now
                new_weights[k] = v
                continue
                
            target_shape = curr_shapes[k]
            if v.shape == target_shape:
                new_weights[k] = v
            else:
                # Try transpose
                if len(v.shape) == 2 and v.transpose().shape == target_shape:
                     new_weights[k] = v.transpose()
                else:
                    # Shape mismatch that transpose can't fix
                    # Check for 1D weights (bias, RMSNorm)
                    if len(v.shape) == 1 and len(target_shape) == 1 and v.shape != target_shape:
                         # e.g. embedding size mismatch
                         print(f"Shape mismatch for {k}: {v.shape} vs {target_shape}")
                    new_weights[k] = v
        
        # 3. Add computed buffers (RoPE) if missing
        # Since we are strict loading, we need to provide all params.
        # RoPE params are computed in __init__, so we can grab them from self.
        
        model_params = dict(tree_flatten(self.parameters()))
        for k, v in model_params.items():
            if k not in new_weights and ("rope" in k):
                 # It's a missing RoPE parameter that we have in memory.
                 # Add it.
                 new_weights[k] = v
                    
        return new_weights

    @classmethod
    def from_pretrained(cls, model_path: str):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        import numpy as np
        
        model_path = Path(model_path)
        if not model_path.exists():
            model_path = Path(snapshot_download(str(model_path)))
            
        with open(model_path / "config.json") as f:
            config = json.load(f)
            
        args = ModelArgs.from_dict(config)
        model = cls(args)
        
        # Load main weights
        weights = {}
        if (model_path / "model.safetensors").exists():
            with safe_open(model_path / "model.safetensors", framework="numpy") as f:
                for k in f.keys():
                    weights[k] = mx.array(f.get_tensor(k))
        
        # Load AudioVAE weights
        # PyTorch checkpoint for VAE
        if (model_path / "audiovae.pth").exists():
            import torch
            vae_pt = torch.load(model_path / "audiovae.pth", map_location="cpu")
            if "state_dict" in vae_pt:
                vae_pt = vae_pt["state_dict"]
            
            for k, v in vae_pt.items():
                weights[f"audio_vae.{k}"] = mx.array(v.numpy())
                
        # Sanitize
        weights = model.sanitize(weights)
        
        # Load weights
        model.load_weights(list(weights.items()), strict=False)
        
        # Tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model

    def generate(self, text: str, prompt_wav_path: str = None, inference_timesteps: int = 10, cfg_value: float = 2.0, max_len: int = 4096, **kwargs):
        # Tokenize
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
            
        start_time = time.perf_counter()
            
        input_ids = self.tokenizer.encode(text)
        input_ids = mx.array(input_ids)
        token_count = len(input_ids)
        
        # Handle prompt
        if prompt_wav_path:
             # TODO: implement audio loading and VAE encoding
             # For now assume no prompt or prompt feature cache handling needs pydub/librosa
             raise NotImplementedError("Prompt audio not fully implemented yet")
        else:
             # dummy prompt? The code handles it, looks like empty prompt features.
             pass
             
        # Just text generation for now
        # We need to construct combined embedding.
        
        # Text embedding
        # scale_emb
        scale_emb = self.args.lm_config.scale_emb if not self.args.lm_config.use_mup else 1.0 
        
        text_embed = self.base_lm.embed_tokens(input_ids[None, :]) # (1, L, D)
        
        # Initial run of base_lm
        
        start_token = mx.array([101])
        input_ids = mx.concatenate([input_ids, start_token])
        text_embed = self.base_lm.embed_tokens(input_ids[None, :])
        
        # combined_embed
        
        enc_outputs, lm_cache = self.base_lm(text_embed)
        lm_hidden = enc_outputs[:, -1, :]
        
        # fsq
        lm_hidden = self.fsq_layer(lm_hidden)
        
        # residual_lm initial run
        
        residual_outputs, res_cache = self.residual_lm(enc_outputs)
        residual_hidden = residual_outputs[:, -1, :]
        
        # Generation Loop
        
        pred_feat_seq = []
        prefix_feat_cond = mx.zeros((1, self.patch_size, self.feat_dim)) 
        
        for i in range(max_len):
            # DiT
            dit_h1 = self.lm_to_dit_proj(lm_hidden)
            dit_h2 = self.res_to_dit_proj(residual_hidden)
            dit_h = dit_h1 + dit_h2 # (1, H)
            
            cond_in = prefix_feat_cond.transpose(0, 2, 1) # (B, D, P)
            
            pred_feat = self.feat_decoder.sample(
                mu=dit_h,
                n_timesteps=inference_timesteps,
                patch_size=self.patch_size,
                cond=cond_in,
                cfg_value=cfg_value
            )
            
            pred_feat = pred_feat.transpose(0, 2, 1)
            pred_feat_seq.append(pred_feat)
            
            curr_embed = self.feat_encoder(pred_feat[:, None, :, :]) # (B, 1, H)
            curr_embed = self.enc_to_lm_proj(curr_embed)
            
            stop_logits = self.stop_head(nn.silu(self.stop_proj(lm_hidden)))
            stop_flag = mx.argmax(stop_logits, axis=-1).item()
            if i > 5 and stop_flag == 1: 
                break
                
            curr_embed_step = curr_embed # (B, 1, H)
            
            new_lm_out, lm_cache = self.base_lm(inputs_embeds=curr_embed_step, cache=lm_cache)
            
            lm_hidden_next = new_lm_out[:, -1, :]
            lm_hidden_next = self.fsq_layer(lm_hidden_next)
            
            res_in = lm_hidden_next[:, None, :] + curr_embed_step
            new_res_out, res_cache = self.residual_lm(inputs_embeds=res_in, cache=res_cache)
            residual_hidden = new_res_out[:, -1, :]
            
            lm_hidden = lm_hidden_next
            prefix_feat_cond = pred_feat
            
        all_feats = mx.concatenate(pred_feat_seq, axis=1) # (B, Total_P, D)
        
        audio = self.audio_vae.decode(all_feats)
        audio = audio.flatten()
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        samples = audio.shape[0]
        sample_rate = 24000 # Config?
        audio_duration_seconds = samples / sample_rate
        
        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0
        
        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{int(audio_duration_seconds // 3600):02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        # Yield single result
        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": round(token_count / elapsed_time, 2) if elapsed_time > 0 else 0
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )
