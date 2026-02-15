"""Core modules for the MOSS audio tokenizer codec.

This file intentionally mirrors upstream module naming where practical so
checkpoint weights can load with minimal remapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache

from .config import MossAudioTokenizerModuleConfig


@dataclass(frozen=True)
class MossAudioTokenizerTransformerConfig:
    input_dimension: int
    output_dimension: int
    d_model: int
    num_heads: int
    num_layers: int
    dim_feedforward: int
    causal: bool = True
    norm: str = "layer_norm"
    positional_embedding: str = "rope"
    max_period: float = 10000.0
    gating: str = "none"
    layer_scale: Optional[float] = None
    conv_layout: bool = True
    context: Optional[int] = None

    @classmethod
    def from_module_config(
        cls,
        module_config: MossAudioTokenizerModuleConfig,
        *,
        context: Optional[int],
    ) -> "MossAudioTokenizerTransformerConfig":
        if module_config.module_type != "Transformer":
            raise ValueError(
                f"Expected Transformer module config, got {module_config.module_type}"
            )
        required_int_fields = {
            "input_dimension": module_config.input_dimension,
            "output_dimension": module_config.output_dimension,
            "d_model": module_config.d_model,
            "num_heads": module_config.num_heads,
            "num_layers": module_config.num_layers,
        }
        for field_name, value in required_int_fields.items():
            if value is None:
                raise ValueError(
                    f"Transformer module config is missing {field_name}: {module_config}"
                )

        dim_feedforward = module_config.dim_feedforward
        if dim_feedforward is None:
            raise ValueError(
                "Transformer module config is missing dim_feedforward: "
                f"{module_config}"
            )

        return cls(
            input_dimension=int(module_config.input_dimension),
            output_dimension=int(module_config.output_dimension),
            d_model=int(module_config.d_model),
            num_heads=int(module_config.num_heads),
            num_layers=int(module_config.num_layers),
            dim_feedforward=int(dim_feedforward),
            causal=bool(
                module_config.causal if module_config.causal is not None else True
            ),
            norm=str(
                module_config.norm if module_config.norm is not None else "layer_norm"
            ),
            positional_embedding=str(
                module_config.positional_embedding
                if module_config.positional_embedding is not None
                else "rope"
            ),
            max_period=float(
                module_config.max_period
                if module_config.max_period is not None
                else 10000
            ),
            gating=str(
                module_config.gating if module_config.gating is not None else "none"
            ),
            layer_scale=(
                float(module_config.layer_scale)
                if module_config.layer_scale is not None
                else None
            ),
            conv_layout=bool(
                module_config.conv_layout
                if module_config.conv_layout is not None
                else True
            ),
            context=context,
        )

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        return self.d_model // self.num_heads


class MossAudioTokenizerLayerScale(nn.Module):
    """Per-channel learned residual scaling."""

    def __init__(self, channels: int, init: float):
        super().__init__()
        self.scale = mx.ones((channels,)) * init

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale


def _create_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5)
    if norm_type == "rms_norm":
        return nn.RMSNorm(dim, eps=1e-8)
    raise ValueError(f"Unsupported norm type: {norm_type}")


def _apply_weights_per_step(
    modules: Sequence[nn.Module],
    schedule: Optional[list[int]],
    x: mx.array,
    offset: int,
) -> mx.array:
    if len(modules) == 1:
        return modules[0](x)

    outputs = []
    _, time_steps, _ = x.shape
    for step in range(time_steps):
        module_index = step + offset
        if schedule is not None:
            if module_index < 0 or module_index >= len(schedule):
                raise ValueError(
                    "weights_per_step_schedule is too short for "
                    f"module_index={module_index}."
                )
            module_index = schedule[module_index]
        if module_index < 0 or module_index >= len(modules):
            raise ValueError(
                f"module_index={module_index} is out of range for {len(modules)} modules."
            )
        outputs.append(modules[module_index](x[:, step : step + 1]))
    return mx.concatenate(outputs, axis=1)


class MossAudioTokenizerMultiheadAttention(nn.Module):
    """Causal MHA with optional RoPE and bounded KV cache context."""

    def __init__(self, config: MossAudioTokenizerTransformerConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.causal = config.causal
        self.context = config.context
        self.weights_per_step_schedule: Optional[list[int]] = None

        # Kept as lists to match upstream checkpoint key names:
        # self_attn.in_projs.0.weight and self_attn.out_projs.0.weight
        self.in_projs = [nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)]
        self.out_projs = [nn.Linear(self.embed_dim, self.embed_dim, bias=False)]

        self.rope = None
        if config.positional_embedding in {"rope", "sin_rope"}:
            self.rope = nn.RoPE(self.head_dim, traditional=True, base=config.max_period)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[KVCache | RotatingKVCache] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, time_steps, hidden_dim = x.shape
        if hidden_dim != self.embed_dim:
            raise ValueError(f"Expected hidden dim {self.embed_dim}, got {hidden_dim}")

        offset = 0 if cache is None else int(cache.offset)
        projected = _apply_weights_per_step(
            self.in_projs,
            self.weights_per_step_schedule,
            x,
            offset,
        )
        projected = projected.reshape(
            batch_size, time_steps, 3, self.num_heads, self.head_dim
        ).transpose(2, 0, 3, 1, 4)

        q = projected[0]
        k = projected[1]
        v = projected[2]

        if self.rope is not None:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if mask is None and self.causal:
            key_steps = k.shape[2]
            key_offset = 0 if cache is None else int(cache.offset) - key_steps
            key_positions = mx.arange(key_steps, dtype=mx.int32) + key_offset
            query_positions = mx.arange(time_steps, dtype=mx.int32) + offset
            delta = query_positions[:, None] - key_positions[None, :]
            allowed = (key_positions[None, :] >= 0) & (delta >= 0)
            if self.context is not None:
                allowed = allowed & (delta < self.context)
            mask = mx.where(allowed, 0.0, -1e9).astype(x.dtype)
            mask = mask[None, None, :, :]

        attended = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, time_steps, -1)
        return _apply_weights_per_step(
            self.out_projs,
            self.weights_per_step_schedule,
            attended,
            offset,
        )


class MossAudioTokenizerTransformerLayer(nn.Module):
    """Transformer block aligned to the upstream module contract."""

    def __init__(self, config: MossAudioTokenizerTransformerConfig):
        super().__init__()
        if config.gating != "none":
            raise ValueError(
                f"Unsupported gating mode for MOSS audio tokenizer: {config.gating}"
            )

        self.self_attn = MossAudioTokenizerMultiheadAttention(config)
        self.norm1 = _create_norm(config.norm, config.d_model)
        self.norm2 = _create_norm(config.norm, config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward, bias=False)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model, bias=False)

        if config.layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = MossAudioTokenizerLayerScale(
                config.d_model, config.layer_scale
            )
            self.layer_scale_2 = MossAudioTokenizerLayerScale(
                config.d_model, config.layer_scale
            )

    def __call__(
        self,
        x: mx.array,
        cache: Optional[KVCache | RotatingKVCache] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        attn_update = self.self_attn(self.norm1(x), cache=cache, mask=mask)
        x = x + self.layer_scale_1(attn_update)
        mlp_update = self.linear2(nn.gelu_approx(self.linear1(self.norm2(x))))
        return x + self.layer_scale_2(mlp_update)


class MossAudioTokenizerTransformer(nn.Module):
    """Stacked transformer with per-layer KV caches."""

    def __init__(self, config: MossAudioTokenizerTransformerConfig):
        super().__init__()
        self.config = config
        self.layers = [
            MossAudioTokenizerTransformerLayer(config) for _ in range(config.num_layers)
        ]

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        if self.config.context is None:
            return [KVCache() for _ in self.layers]
        return [
            RotatingKVCache(max_size=self.config.context, keep=0) for _ in self.layers
        ]

    def __call__(
        self,
        x: mx.array,
        cache: Optional[list[KVCache | RotatingKVCache]] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if cache is None:
            per_layer_cache: list[Optional[KVCache | RotatingKVCache]] = [
                None for _ in self.layers
            ]
        else:
            if len(cache) != len(self.layers):
                raise ValueError(
                    "Cache depth mismatch: expected "
                    f"{len(self.layers)} layers, got {len(cache)}."
                )
            per_layer_cache = list(cache)

        for layer, layer_cache in zip(self.layers, per_layer_cache):
            x = layer(x, cache=layer_cache, mask=mask)
        return x


class MossAudioTokenizerProjectedTransformer(nn.Module):
    """Transformer block with optional input/output projections."""

    def __init__(
        self,
        config: MossAudioTokenizerTransformerConfig,
        *,
        module_type: str = "Transformer",
    ):
        super().__init__()
        self.module_type = module_type
        self.downsample_ratio = 1
        self.conv_layout = config.conv_layout
        self.input_dimension = config.input_dimension
        self.output_dimension = config.output_dimension

        if config.input_dimension == config.d_model:
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(
                config.input_dimension, config.d_model, bias=False
            )
        self.transformer = MossAudioTokenizerTransformer(config)
        if config.output_dimension == config.d_model:
            self.output_proj = None
        else:
            self.output_proj = nn.Linear(
                config.d_model, config.output_dimension, bias=False
            )

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        return self.transformer.make_cache()

    def __call__(
        self,
        x: mx.array,
        input_lengths: mx.array,
        cache: Optional[list[KVCache | RotatingKVCache]] = None,
        mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        if self.conv_layout:
            x = x.swapaxes(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        x = self.transformer(x, cache=cache, mask=mask)
        if self.output_proj is not None:
            x = self.output_proj(x)
        if self.conv_layout:
            x = x.swapaxes(1, 2)
        return x, input_lengths


class MossAudioTokenizerPatchedPretransform(nn.Module):
    """Patch/unpatch module used for deterministic stride changes."""

    def __init__(
        self,
        patch_size: int,
        is_downsample: bool,
        *,
        module_type: str = "PatchedPretransform",
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.downsample_ratio = self.patch_size
        self.is_downsample = is_downsample
        self.module_type = module_type

    def encode(self, x: mx.array, input_lengths: mx.array) -> tuple[mx.array, mx.array]:
        batch_size, channels, _ = x.shape
        patch = self.patch_size
        x = (
            x.reshape(batch_size, channels, -1, patch)
            .transpose(0, 1, 3, 2)
            .reshape(batch_size, channels * patch, -1)
        )
        output_lengths = input_lengths // patch
        return x, output_lengths

    def decode(self, x: mx.array, input_lengths: mx.array) -> tuple[mx.array, mx.array]:
        batch_size, channels_times_patch, length = x.shape
        patch = self.patch_size
        channels = channels_times_patch // patch
        x = (
            x.reshape(batch_size, channels, patch, length)
            .transpose(0, 1, 3, 2)
            .reshape(batch_size, channels, length * patch)
        )
        output_lengths = input_lengths * patch
        return x, output_lengths

    def __call__(
        self,
        x: mx.array,
        input_lengths: mx.array,
        cache=None,
        mask=None,
    ) -> tuple[mx.array, mx.array]:
        del cache, mask
        if self.is_downsample:
            return self.encode(x, input_lengths)
        return self.decode(x, input_lengths)
