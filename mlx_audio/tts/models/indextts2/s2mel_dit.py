from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .s2mel_gpt_fast import GPTFastArgs, GPTFastTransformer
from .s2mel_utils import sequence_mask
from .s2mel_wavenet import WN


def _mish(x: mx.array) -> mx.array:
    return x * mx.tanh(mx.log1p(mx.exp(x)))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        ]
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = 10000
        self.scale = 1000

        half = frequency_embedding_size // 2
        freqs = mx.exp(
            -mx.log(mx.array(self.max_period, dtype=mx.float32))
            * (mx.arange(half, dtype=mx.float32) / half)
        )
        self.freqs = freqs

    def timestep_embedding(self, t: mx.array) -> mx.array:
        args = self.scale * t[:, None].astype(mx.float32) * self.freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if self.frequency_embedding_size % 2:
            emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        x = self.timestep_embedding(t)
        for layer in self.mlp:
            x = layer(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        ]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        h = c
        for layer in self.adaLN_modulation:
            h = layer(h)
        shift, scale = mx.split(h, 2, axis=-1)
        x = self.norm_final(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        return self.linear(x)


@dataclass
class DiTConfig:
    hidden_dim: int = 512
    num_heads: int = 8
    depth: int = 13
    in_channels: int = 80
    content_dim: int = 512
    style_dim: int = 192
    is_causal: bool = False
    long_skip_connection: bool = True
    uvit_skip_connection: bool = True
    final_layer_type: str = "wavenet"  # wavenet or mlp
    wavenet_hidden_dim: int = 512
    wavenet_num_layers: int = 8
    wavenet_kernel_size: int = 5
    wavenet_dilation_rate: int = 1


class DiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg

        gpt_cfg = GPTFastArgs(
            block_size=16384,
            n_layer=cfg.depth,
            n_head=cfg.num_heads,
            dim=cfg.hidden_dim,
            head_dim=cfg.hidden_dim // cfg.num_heads,
            n_local_heads=cfg.num_heads,
            intermediate_size=int(2 * (4 * cfg.hidden_dim) / 3),
            uvit_skip_connection=cfg.uvit_skip_connection,
        )
        gpt_cfg.intermediate_size = 1536  # match checkpoint
        self.transformer = GPTFastTransformer(gpt_cfg)

        self.in_channels = cfg.in_channels

        self.x_embedder = nn.Linear(cfg.in_channels, cfg.hidden_dim, bias=True)

        # Present in torch checkpoints but unused for continuous conditioning in IndexTTS2.
        self.cond_embedder = nn.Embedding(1024, cfg.hidden_dim)
        self.content_mask_embedder = nn.Embedding(1, cfg.hidden_dim)

        self.cond_projection = nn.Linear(cfg.content_dim, cfg.hidden_dim, bias=True)
        self.t_embedder = TimestepEmbedder(cfg.hidden_dim)

        # (x + prompt_x + cond) + style
        self.cond_x_merge_linear = nn.Linear(
            cfg.hidden_dim + cfg.in_channels * 2 + cfg.style_dim,
            cfg.hidden_dim,
            bias=True,
        )

        self.long_skip_connection = cfg.long_skip_connection
        if self.long_skip_connection:
            self.skip_linear = nn.Linear(cfg.hidden_dim + cfg.in_channels, cfg.hidden_dim, bias=True)

        self.final_layer_type = cfg.final_layer_type
        if self.final_layer_type == "wavenet":
            self.t_embedder2 = TimestepEmbedder(cfg.wavenet_hidden_dim)
            self.conv1 = nn.Linear(cfg.hidden_dim, cfg.wavenet_hidden_dim, bias=True)
            self.conv2 = nn.Conv1d(cfg.wavenet_hidden_dim, cfg.in_channels, kernel_size=1)
            self.wavenet = WN(
                hidden_channels=cfg.wavenet_hidden_dim,
                kernel_size=cfg.wavenet_kernel_size,
                dilation_rate=cfg.wavenet_dilation_rate,
                n_layers=cfg.wavenet_num_layers,
                gin_channels=cfg.wavenet_hidden_dim,
                p_dropout=0.0,
            )
            self.final_layer = FinalLayer(cfg.wavenet_hidden_dim, cfg.wavenet_hidden_dim)
            self.res_projection = nn.Linear(cfg.hidden_dim, cfg.wavenet_hidden_dim, bias=True)
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.in_channels),
            )

        self.input_pos = mx.arange(16384)

    def __call__(
        self,
        x: mx.array,
        prompt_x: mx.array,
        x_lens: mx.array,
        t: mx.array,
        style: mx.array,
        cond: mx.array,
    ) -> mx.array:
        # x, prompt_x: (B, C, T)
        # cond: (B, T, content_dim)
        B, C, T = x.shape

        t1 = self.t_embedder(t.astype(mx.float32))  # (B, D)
        cond_proj = self.cond_projection(cond)

        x_t = x.transpose(0, 2, 1)  # (B, T, C)
        p_t = prompt_x.transpose(0, 2, 1)
        x_in = mx.concatenate([x_t, p_t, cond_proj, mx.repeat(style[:, None, :], T, axis=1)], axis=-1)
        x_in = self.cond_x_merge_linear(x_in)

        # Attention mask (non-causal)
        x_mask = sequence_mask(x_lens, max_length=int(T)).astype(mx.bool_)
        key_mask = x_mask[:, None, None, :]  # (B,1,1,T)
        attn_mask = mx.where(key_mask, 0.0, -1e9).astype(mx.float32)
        attn_mask = mx.broadcast_to(attn_mask, (B, 1, T, T))

        input_pos = self.input_pos[:T]
        x_res = self.transformer(x_in, t1, input_pos, attn_mask)

        if self.long_skip_connection:
            x_res = self.skip_linear(mx.concatenate([x_res, x_t], axis=-1))

        if self.final_layer_type == "wavenet":
            h = self.conv1(x_res)  # (B, T, H)
            t2 = self.t_embedder2(t.astype(mx.float32))
            g = t2[:, None, :]  # (B,1,H)
            x_mask_c = x_mask[:, :, None].astype(h.dtype)
            h = self.wavenet(h, x_mask_c, g=g) + self.res_projection(x_res)
            h = self.final_layer(h, t1)
            y = self.conv2(h)
        else:
            y = self.final_mlp(x_res)
        return y.transpose(0, 2, 1)
