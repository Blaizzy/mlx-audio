import math

import mlx.core as mx
import mlx.nn as nn

from .config import CFMConfig, LMConfig
from .minicpm import MiniCPMModel


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0

    def __call__(self, x, scale=1000):
        if x.ndim < 1:
            x = x.reshape(1)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim) * -emb)

        emb = scale * x[:, None] * emb[None, :]

        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
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


class VoxCPMLocDiTV2(nn.Module):
    """VoxCPM2 Diffusion Transformer with multi-token mu support."""

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

        self.decoder = MiniCPMModel(config)  # vocab_size=0

    def __call__(self, x, mu, t, cond, dt):
        # x: (N, C, T) -> (N, T, C)
        x = x.transpose(0, 2, 1)
        x_proj = self.in_proj(x)

        # cond: (N, C, T') -> (N, T', C)
        cond = cond.transpose(0, 2, 1)
        cond_proj = self.cond_proj(cond)
        prefix = cond.shape[1]

        t_emb = self.time_embeddings(t)
        t_emb = self.time_mlp(t_emb)

        dt_emb = self.time_embeddings(dt)
        dt_emb = self.delta_time_mlp(dt_emb)

        t_comb = t_emb + dt_emb  # (N, H)

        # V2: mu is (N, 2*H) -> reshape to (N, 2, H) for multi-token start
        H = x_proj.shape[-1]  # hidden_size after projection
        mu_tokens = mu.reshape(x.shape[0], -1, H)  # (N, num_mu_tokens, H)
        num_mu_tokens = mu_tokens.shape[1]

        hidden = mx.concatenate(
            [mu_tokens, t_comb[:, None, :], cond_proj, x_proj], axis=1
        )

        hidden, _ = self.decoder(inputs_embeds=hidden, is_causal=False)

        # Slice output: skip mu_tokens + t_token + cond
        hidden = hidden[:, num_mu_tokens + 1 + prefix :, :]

        hidden = self.out_proj(hidden)

        return hidden.transpose(0, 2, 1)  # (N, C, T)


class UnifiedCFM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cfm_params: CFMConfig,
        estimator: VoxCPMLocDiTV2,
        mean_mode: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.estimator = estimator
        self.cfm_params = cfm_params
        self.mean_mode = mean_mode

    def solve_euler(self, x, t_span, mu, cond, cfg_value=1.0, use_cfg_zero_star=True):
        t = t_span[0]
        dt = t_span[0] - t_span[1]

        current_x = x

        zero_init_steps = max(1, int(len(t_span) * 0.04))

        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = mx.zeros_like(current_x)
            else:
                b = current_x.shape[0]

                x_in = mx.concatenate([current_x, current_x], axis=0)
                mu_in = mx.concatenate(
                    [mu, mx.zeros_like(mu)], axis=0
                )

                t_val = mx.full((x_in.shape[0],), t)

                # mean_mode controls whether dt is used
                if self.mean_mode:
                    dt_val_in = mx.full((x_in.shape[0],), dt)
                else:
                    dt_val_in = mx.zeros((x_in.shape[0],))

                cond_in = mx.concatenate([cond, cond], axis=0)

                out = self.estimator(x_in, mu_in, t_val, cond_in, dt_val_in)

                chunk_size = current_x.shape[0]
                dphi_dt = out[:chunk_size]
                cfg_dphi_dt = out[chunk_size:]

                if use_cfg_zero_star:
                    positive_flat = dphi_dt.reshape(chunk_size, -1)
                    negative_flat = cfg_dphi_dt.reshape(chunk_size, -1)

                    dot_prod = mx.sum(
                        positive_flat * negative_flat, axis=1, keepdims=True
                    )
                    sq_norm = mx.sum(negative_flat**2, axis=1, keepdims=True) + 1e-8
                    st_star = dot_prod / sq_norm
                    st_star = st_star.reshape(chunk_size, 1, 1)
                else:
                    st_star = 1.0

                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (
                    dphi_dt - cfg_dphi_dt * st_star
                )

            current_x = current_x - dt * dphi_dt
            t = t - dt

            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]

        return current_x

    def sample(self, mu, n_timesteps, patch_size, cond, temperature=1.0, cfg_value=1.0):
        B = mu.shape[0]
        T = patch_size

        z = mx.random.normal((B, self.in_channels, T)) * temperature

        t_span = mx.linspace(1, 0, n_timesteps + 1)
        sway_coef = 1.0
        t_span = t_span + sway_coef * (mx.cos(math.pi / 2 * t_span) - 1 + t_span)

        return self.solve_euler(z, t_span, mu, cond, cfg_value=cfg_value)
