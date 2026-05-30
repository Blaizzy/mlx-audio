import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _approx_gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + scale) + shift


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(_approx_gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.proj = nn.Linear(self.inner_dim, dim)

    def to_heads(self, x: mx.array) -> mx.array:
        b, t, c = x.shape
        x = x.reshape(b, t, self.num_heads, c // self.num_heads)
        return mx.transpose(x, (0, 2, 1, 3))

    def __call__(self, x: mx.array, attn_mask: Optional[mx.array]) -> mx.array:
        b, t, _ = x.shape
        q = self.to_heads(self.to_q(x))
        k = self.to_heads(self.to_k(x))
        v = self.to_heads(self.to_v(x))

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        scores = (q @ mx.swapaxes(k, -1, -2)) * self.scale
        if attn_mask is not None:
            if attn_mask.ndim == 3:
                attn_mask = mx.expand_dims(attn_mask, 1)
            scores = mx.where(attn_mask, scores, -mx.inf)
        attn = mx.softmax(scores, axis=-1)
        x = attn @ v
        x = mx.transpose(x, (0, 2, 1, 3)).reshape(b, t, self.inner_dim)
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ]
        self.frequency_embedding_size = frequency_embedding_size
        self.scale = 1000

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
        ).astype(t.dtype)
        args = mx.expand_dims(t, -1) * mx.expand_dims(freqs, 0)
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        x = self.timestep_embedding(t * self.scale, self.frequency_embedding_size)
        for layer in self.mlp:
            x = layer(x)
        return x


class CausalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block = [
            None,
            nn.Conv1d(in_channels, out_channels, kernel_size),
            None,
            nn.LayerNorm(out_channels),
            nn.Mish(),
            None,
            nn.Conv1d(out_channels, out_channels, kernel_size),
            None,
        ]

    def _causal_conv(self, x: mx.array, conv: nn.Conv1d) -> mx.array:
        x = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        return conv(x)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        if mask is not None:
            x = x * mask
        x = self._causal_conv(x, self.block[1])
        x = self.block[3](x)
        x = self.block[4](x)
        x = self._causal_conv(x, self.block[6])
        if mask is not None:
            x = x * mask
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=True,
            qk_norm=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
        )
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.conv = CausalConvBlock(hidden_size, hidden_size, kernel_size=3)
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size),
        ]

    def __call__(
        self, x: mx.array, c: mx.array, attn_mask: Optional[mx.array]
    ) -> mx.array:
        mod = c
        for layer in self.adaLN_modulation:
            mod = layer(mod)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = mx.split(mod, 9, axis=-1)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), attn_mask
        )
        x = x + gate_conv * self.conv(modulate(self.norm3(x), shift_conv, scale_conv))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        ]
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear = nn.Linear(hidden_size, out_channels)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mod = c
        for layer in self.adaLN_modulation:
            mod = layer(mod)
        shift, scale = mx.split(mod, 2, axis=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        depth: int = 16,
        num_heads: int = 8,
        head_dim: int = 64,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.in_proj = nn.Linear(in_channels, hidden_size)
        self.blocks = [
            DiTBlock(hidden_size, num_heads, head_dim, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ]
        self.final_layer = FinalLayer(hidden_size, out_channels)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        mu: mx.array,
        t: mx.array,
        spks: Optional[mx.array] = None,
        cond: Optional[mx.array] = None,
    ) -> mx.array:
        t = mx.expand_dims(self.t_embedder(t), 1)
        pieces = [x, mu]
        if spks is not None:
            spks = mx.broadcast_to(
                mx.expand_dims(spks, -1), (spks.shape[0], spks.shape[1], x.shape[-1])
            )
            pieces.append(spks)
        if cond is not None:
            pieces.append(cond)
        x = mx.concatenate(pieces, axis=1)
        return self.blocks_forward(x, t, mask.astype(mx.bool_))

    def blocks_forward(
        self, x: mx.array, t: mx.array, mask: Optional[mx.array]
    ) -> mx.array:
        x = mx.transpose(x, (0, 2, 1))
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, t, mask)
        x = self.final_layer(x, t)
        return mx.transpose(x, (0, 2, 1))
