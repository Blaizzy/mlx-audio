import math
from typing import Optional, List

import mlx.core as mx
import mlx.nn as nn

from .minicpm import MiniCPMModel
from .config import DiTConfig, CFMConfig, LMConfig

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0

    def __call__(self, x, scale=1000):
        # x: (N,) or scalar
        if x.ndim < 1:
            x = x.reshape(1)
        
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim) * -emb) 
        # emb: (half_dim,)
        
        # x: (N,)
        emb = scale * x[:, None] * emb[None, :] # (N, half_dim)
        
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1) # (N, dim)
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: int = None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, out_dim or time_embed_dim)

    def __call__(self, x):
         x = self.linear_1(x)
         x = nn.silu(x)
         x = self.linear_2(x)
         return x

class VoxCPMLocDiT(nn.Module):
    def __init__(self, config: LMConfig, in_channels: int = 64):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        
        self.in_proj = nn.Linear(in_channels, config.hidden_size)
        self.cond_proj = nn.Linear(in_channels, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, in_channels)
        
        self.time_embeddings = SinusoidalPosEmb(config.hidden_size)
        self.time_mlp = TimestepEmbedding(config.hidden_size, config.hidden_size)
        self.delta_time_mlp = TimestepEmbedding(config.hidden_size, config.hidden_size)
        
        self.decoder = MiniCPMModel(config) # vocab_size=0

    def __call__(self, x, mu, t, cond, dt):
        # x: (N, C, T) -> Transpose to (N, T, C)
        # MLX Linear expects (..., C).
        # We assume input x is (N, C, T) to match PyTorch interface derived from existing code?
        # My implementation of inference loop will probably work with (N, T, C) easier, 
        # but let's stick to (N, C, T) if that's what `voxcpm.py` generates.
        # Actually `UnifiedCFM` operates on (N, C, T). `z` initialized as (batch, in_channels, t).
        
        x = x.transpose(0, 2, 1) # (N, T, C)
        x_proj = self.in_proj(x)
        
        cond = cond.transpose(0, 2, 1) # (N, T', C)
        cond_proj = self.cond_proj(cond)
        
        t_emb = self.time_embeddings(t)
        t_emb = self.time_mlp(t_emb)
        
        dt_emb = self.time_embeddings(dt)
        dt_emb = self.delta_time_mlp(dt_emb)
        
        t_comb = t_emb + dt_emb # (N, H)
        
        # mu: (N, H)? No, mu is (b, h_dit) from `voxcpm.py`.
        # So mu + t_comb -> (N, H)
        start_token = (mu + t_comb)[:, None, :] # (N, 1, H)
        
        # concat: start, cond, x
        hidden = mx.concatenate([start_token, cond_proj, x_proj], axis=1) # (N, 1 + T' + T, H)
        
        # pass to decoder
        # is_causal=False implicit if no mask passed to my MiniCPM?
        # My MiniCPM applies RoPE.
        # RoPE usually assumes causal positions 0..L.
        # Here we concatenate.
        # `voxcpm.py` says `is_causal=False` for DiT.
        # In this case, attention is full.
        # My `MiniCPMModel` does full attention if cache is None and mask is None.
        
        # Pass is_causal=False for full bidirectional attention (DiT uses non-causal)
        hidden, _ = self.decoder(inputs_embeds=hidden, is_causal=False)
        
        # slice output
        prefix = cond.shape[1]
        # hidden[:, prefix+1:, :]
        hidden = hidden[:, prefix + 1 :, :]
        
        hidden = self.out_proj(hidden) # (N, T, C)
        
        return hidden.transpose(0, 2, 1) # (N, C, T)

class UnifiedCFM(nn.Module):
    def __init__(self, in_channels: int, cfm_params: CFMConfig, estimator: VoxCPMLocDiT, mean_mode: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.estimator = estimator
        self.cfm_params = cfm_params
        
    def solve_euler(self, x, t_span, mu, cond, cfg_value=1.0, use_cfg_zero_star=True):
        t = t_span[0]
        dt = t_span[0] - t_span[1]  # Initial dt
        # t_span is linspace 1 -> 0
        
        # For MLX, we can't modify list in place well, but we can iterate.
        current_x = x
        
        zero_init_steps = max(1, int(len(t_span) * 0.04))
        
        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                 # dphi_dt = 0
                 dphi_dt = mx.zeros_like(current_x)
            else:
                # Classifier-Free Guidance inference introduced in VoiceBox
                b = current_x.shape[0]
                
                # CRITICAL: For CFG, the unconditional branch needs zeros for mu!
                # First batch: conditional (with mu)
                # Second batch: unconditional (with zeros for mu)
                x_in = mx.concatenate([current_x, current_x], axis=0)
                mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)  # FIXED: zeros for uncond branch
                
                # t and dt
                t_val = mx.full((x_in.shape[0],), t)
                
                # Mean mode uses dt, otherwise zeros (PyTorch: if not self.mean_mode: dt_in = zeros)
                # Our config has mean_mode=False by default
                dt_val_in = mx.zeros((x_in.shape[0],))  # FIXED: zeros for dt when mean_mode=False
                
                # cond - both branches get the same conditioning
                cond_in = mx.concatenate([cond, cond], axis=0)
                
                # Estimator call
                out = self.estimator(x_in, mu_in, t_val, cond_in, dt_val_in)
                
                # split: dphi_dt (conditional), cfg_dphi_dt (unconditional)
                chunk_size = current_x.shape[0]
                dphi_dt = out[:chunk_size]       # conditional result
                cfg_dphi_dt = out[chunk_size:]   # unconditional result
                
                if use_cfg_zero_star:
                    # Optimized scale
                    # flat views
                    positive_flat = dphi_dt.reshape(chunk_size, -1)
                    negative_flat = cfg_dphi_dt.reshape(chunk_size, -1)
                    
                    dot_prod = mx.sum(positive_flat * negative_flat, axis=1, keepdims=True)
                    sq_norm = mx.sum(negative_flat**2, axis=1, keepdims=True) + 1e-8
                    st_star = dot_prod / sq_norm
                    # reshape st_star to (B, 1, 1)
                    st_star = st_star.reshape(chunk_size, 1, 1)
                else:
                    st_star = 1.0
                    
                # CFG formula: cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)
                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)
            
            current_x = current_x - dt * dphi_dt
            t = t - dt
            
            # Update dt for next step (variable step sizes due to sway sampling)
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]
            
        return current_x

    def sample(self, mu, n_timesteps, patch_size, cond, temperature=1.0, cfg_value=1.0):
        # mu: (B, H) (hidden_size/dit dim)
        B = mu.shape[0]
        T = patch_size # Wait, patch_size arg in `voxcpm.py` seems to be passed as `feat_pred_seq` length logic?
        # In `voxcpm.py`:
        # pred_feat = self.feat_decoder(..., patch_size=self.patch_size, ...)
        # self.patch_size = config.patch_size (e.g. 2 or 4)
        
        # `UnifiedCFM.forward` (mapped to `sample` here conceptually):
        # t = patch_size.
        # z = randn(b, in_channels, t)
        
        z = mx.random.normal((B, self.in_channels, T)) * temperature
        
        # t_span linspace 1 -> 0
        t_span = mx.linspace(1, 0, n_timesteps + 1)
        # sway sampling
        sway_coef = 1.0 
        t_span = t_span + sway_coef * (mx.cos(math.pi / 2 * t_span) - 1 + t_span)
        
        return self.solve_euler(z, t_span, mu, cond, cfg_value=cfg_value)
