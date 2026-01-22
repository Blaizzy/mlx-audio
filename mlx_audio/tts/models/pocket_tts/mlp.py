import math

import mlx.core as mx
import mlx.nn as nn

from .config import FlowLMConfig


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + scale) + shift


def _rms_norm(x: mx.array, alpha: mx.array, eps: float) -> mx.array:
    x_dtype = x.dtype
    x = x.astype(mx.float32)
    var = eps + mx.var(x, axis=-1, keepdims=True, ddof=1)
    y = x * (alpha.astype(var.dtype) * mx.rsqrt(var))
    return y.astype(x_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x, self.alpha, self.eps)


class LayerNorm(nn.Module):
    """LayerNorm implementation that matches the PyTorch reference."""

    def __init__(
        self, channels: int, eps: float = 1e-6, elementwise_affine: bool = True
    ):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = mx.ones((channels,))
            self.bias = mx.zeros((channels,))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.layer_norm(x, self.weight, self.bias, eps=self.eps)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        self.freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            RMSNorm(hidden_size),
        ]

    def __call__(self, t: mx.array) -> mx.array:
        if t.ndim == 1:
            t = t[:, None]
        args = t.astype(mx.float32) * self.freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        out = embedding
        for layer in self.mlp:
            out = layer(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = LayerNorm(channels, eps=1e-6)
        self.mlp = [
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        ]
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        ]

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        modulation = y
        for layer in self.adaLN_modulation:
            modulation = layer(modulation)
        shift_mlp, scale_mlp, gate_mlp = mx.split(modulation, 3, axis=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        for layer in self.mlp:
            h = layer(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        ]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        modulation = c
        for layer in self.adaLN_modulation:
            modulation = layer(modulation)
        shift, scale = mx.split(modulation, 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        cond_channels: int,
        num_res_blocks: int,
        num_time_conds: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_time_conds = num_time_conds

        if num_time_conds == 1:
            raise ValueError("num_time_conds must be != 1 for AdaLN conditioning.")

        self.time_embed = [
            TimestepEmbedder(model_channels) for _ in range(num_time_conds)
        ]
        self.cond_embed = nn.Linear(cond_channels, model_channels, bias=True)
        self.input_proj = nn.Linear(in_channels, model_channels, bias=True)
        self.res_blocks = [ResBlock(model_channels) for _ in range(num_res_blocks)]
        self.final_layer = FinalLayer(model_channels, out_channels)

    @classmethod
    def from_pydantic_config(
        cls, cfg: FlowLMConfig, latent_dim: int, cond_dim: int
    ) -> "SimpleMLPAdaLN":
        flow_dim = cfg.flow.dim
        flow_depth = cfg.flow.depth
        num_time_conds = 2
        return cls(
            latent_dim,
            flow_dim,
            latent_dim,
            cond_dim,
            flow_depth,
            num_time_conds=num_time_conds,
        )

    def __call__(self, c: mx.array, s: mx.array, t: mx.array, x: mx.array) -> mx.array:
        ts = [s, t]
        if len(ts) != self.num_time_conds:
            raise ValueError(
                f"Expected {self.num_time_conds} time conditions, got {len(ts)}"
            )
        x = self.input_proj(x)
        t_combined = mx.zeros((x.shape[0], self.model_channels), dtype=x.dtype)
        for idx, time_embed in enumerate(self.time_embed):
            t_combined = t_combined + time_embed(ts[idx])
        t_combined = t_combined / self.num_time_conds
        c = self.cond_embed(c)
        y = t_combined + c
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)
