from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def swish(x: mx.array) -> mx.array:
    return x * nn.sigmoid(x)


class Conv1d(nn.Module):
    """Minimal Conv1d with groups support (MLX layout).

    Expects input shape (B, T, C_in).
    Weight shape is (C_out, K, C_in/groups).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")

        # Init matches torch-ish uniform
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if "bias" in self:
            y = y + self.bias
        return y


@dataclass
class Wav2Vec2BertConfig:
    # Core
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    feature_projection_input_dim: int = 160
    layer_norm_eps: float = 1e-5

    # Attention
    attention_dropout: float = 0.0
    position_embeddings_type: Optional[str] = "relative_key"  # rotary|relative|relative_key|None
    rotary_embedding_base: int = 10000
    max_source_positions: int = 5000
    left_max_position_embeddings: int = 64
    right_max_position_embeddings: int = 8

    # Conformer conv
    conv_depthwise_kernel_size: int = 31
    conformer_conv_dropout: float = 0.1

    # Dropouts (inference only; kept for compatibility)
    hidden_dropout: float = 0.0
    activation_dropout: float = 0.0
    feat_proj_dropout: float = 0.0


class Wav2Vec2BertFeatureProjection(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            config.feature_projection_input_dim, eps=config.layer_norm_eps
        )
        self.projection = nn.Linear(
            config.feature_projection_input_dim, config.hidden_size
        )

    def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array]:
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        return hidden_states, norm_hidden_states


class Wav2Vec2BertFeedForward(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig, *, hidden_size: Optional[int] = None):
        super().__init__()
        hs = hidden_size if hidden_size is not None else config.hidden_size
        self.intermediate_dense = nn.Linear(hs, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, hs)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = swish(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        return hidden_states


class Wav2Vec2BertConvolutionModule(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError(
                "conv_depthwise_kernel_size must be odd for SAME padding"
            )

        self.hidden_size = config.hidden_size
        self.kernel_size = config.conv_depthwise_kernel_size

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pointwise_conv1 = Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.depthwise_conv = Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_depthwise_kernel_size,
            groups=config.hidden_size,
            bias=False,
            padding=0,
        )
        self.depthwise_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.pointwise_conv2 = Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            bias=False,
        )

    def __call__(
        self, hidden_states: mx.array, *, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        # hidden_states: (B, T, C)
        hidden_states = self.layer_norm(hidden_states)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None]

        hidden_states = self.pointwise_conv1(hidden_states)
        a, b = hidden_states.split(2, axis=-1)
        hidden_states = a * nn.sigmoid(b)

        # Causal left pad
        pad_left = self.kernel_size - 1
        hidden_states = mx.pad(hidden_states, ((0, 0), (pad_left, 0), (0, 0)))
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_layer_norm(hidden_states)
        hidden_states = swish(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states


class Wav2Vec2BertSelfAttention(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.scale = self.head_size**-0.5

        self.position_embeddings_type = config.position_embeddings_type
        if self.position_embeddings_type not in (None, "relative_key"):
            raise NotImplementedError(
                f"position_embeddings_type={self.position_embeddings_type} not implemented"
            )

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        if self.position_embeddings_type == "relative_key":
            self.left_max_position_embeddings = config.left_max_position_embeddings
            self.right_max_position_embeddings = config.right_max_position_embeddings
            num_positions = (
                self.left_max_position_embeddings
                + self.right_max_position_embeddings
                + 1
            )
            self.distance_embedding = nn.Embedding(num_positions, self.head_size)

    def _relative_key_scores(self, q: mx.array) -> mx.array:
        # q: (B, H, T, Dh) -> (B, H, T, T)
        T = q.shape[2]
        pos_l = mx.arange(T).reshape(T, 1)
        pos_r = mx.arange(T).reshape(1, T)
        dist = pos_r - pos_l
        dist = mx.clip(
            dist,
            -self.left_max_position_embeddings,
            self.right_max_position_embeddings,
        )
        dist = dist + self.left_max_position_embeddings

        pos_emb = self.distance_embedding(dist)  # (T, T, Dh)
        pos_emb = pos_emb.astype(q.dtype)
        return mx.einsum("bhqd,qkd->bhqk", q, pos_emb) * self.scale

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        # hidden_states: (B, T, C)
        B, T, C = hidden_states.shape

        q = self.linear_q(hidden_states)
        k = self.linear_k(hidden_states)
        v = self.linear_v(hidden_states)

        q = q.reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)

        mask = None
        if attention_mask is not None:
            # attention_mask: (B, T) with 1=keep, 0=pad
            # Convert to additive mask: (B, 1, 1, T)
            mask = (1.0 - attention_mask.astype(mx.float32))[:, None, None, :] * (-1e9)

        if self.position_embeddings_type == "relative_key":
            pos_scores = self._relative_key_scores(q)
            mask = pos_scores if mask is None else (mask + pos_scores)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.linear_out(out)


class Wav2Vec2BertEncoderLayer(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        d = config.hidden_size

        self.ffn1_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.ffn1 = Wav2Vec2BertFeedForward(config)

        self.self_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.self_attn = Wav2Vec2BertSelfAttention(config)

        self.conv_module = Wav2Vec2BertConvolutionModule(config)

        self.ffn2_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.ffn2 = Wav2Vec2BertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: Optional[mx.array] = None,
        conv_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        # 1) FFN1
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual

        # 2) Self-attn
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states + residual

        # 3) Conv
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = hidden_states + residual

        # 4) FFN2
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class Wav2Vec2BertEncoder(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        self.layers = [Wav2Vec2BertEncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[mx.array, Optional[List[mx.array]]]:
        all_hidden_states: Optional[List[mx.array]] = [] if output_hidden_states else None

        conv_attention_mask = attention_mask

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None]

        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                conv_attention_mask=conv_attention_mask,
            )
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask[:, :, None]

        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states


class Wav2Vec2BertModel(nn.Module):
    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        self.config = config
        self.feature_projection = Wav2Vec2BertFeatureProjection(config)
        self.encoder = Wav2Vec2BertEncoder(config)

        # Present in HF checkpoints (used for SpecAugment during training).
        # We keep it to allow strict weight loading.
        self.masked_spec_embed = mx.zeros((config.hidden_size,), dtype=mx.float32)

    def __call__(
        self,
        input_features: mx.array,
        *,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[mx.array, Optional[List[mx.array]]]:
        hidden_states, _ = self.feature_projection(input_features)
        return self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Transpose conv weights from torch layout when needed."""
        curr = dict(tree_flatten(self.parameters()))
        out = {}
        for k, v in weights.items():
            if k in curr and v.ndim == 3 and curr[k].ndim == 3 and v.shape != curr[k].shape:
                # Torch Conv1d: (O, I/groups, K) -> MLX: (O, K, I/groups)
                if v.shape[0] == curr[k].shape[0] and v.shape[2] == curr[k].shape[1]:
                    v = v.transpose(0, 2, 1)
            out[k] = v
        return out
