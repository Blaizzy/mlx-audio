"""Inference-only w2v-BERT encoder, including relative-key attention and masks.

Based on the w2v-BERT implementation used by Confucius4 and Hugging Face's
Wav2Vec2BertModel. All runtime weights have their LoRA updates merged.
"""

import math

import mlx.core as mx
from mlx import nn

from .config import EncoderConfig


class FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            config.feature_projection_input_dim, eps=config.layer_norm_eps
        )
        self.projection = nn.Linear(
            config.feature_projection_input_dim, config.hidden_size
        )

    def __call__(self, x):
        return self.projection(self.layer_norm(x))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x):
        return self.output_dense(nn.silu(self.intermediate_dense(x)))


class RelativeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.left = config.left_max_position_embeddings
        self.right = config.right_max_position_embeddings
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.distance_embedding = nn.Embedding(
            self.left + self.right + 1, self.head_dim
        )

    def __call__(self, x, attention_mask=None):
        batch, length, dim = x.shape
        q, k, v = [
            projection(x)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
            for projection in (self.linear_q, self.linear_k, self.linear_v)
        ]
        positions = mx.arange(length)
        distance = (
            mx.clip(positions[None, :] - positions[:, None], -self.left, self.right)
            + self.left
        )
        # Project the small relative-position table first. Gathering embeddings
        # before the projection would allocate [T, T, head_dim].
        relative = q @ self.distance_embedding.weight.T
        bias = mx.take_along_axis(relative, distance[None, None], axis=-1)
        bias = bias * (self.head_dim**-0.5)
        if attention_mask is not None:
            bias = mx.where(
                attention_mask[:, None, None, :], bias, mx.finfo(q.dtype).min
            )
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(self.head_dim), mask=bias
        )
        return self.linear_out(out.transpose(0, 2, 1, 3).reshape(batch, length, dim))


class ConvolutionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.kernel_size = config.conv_depthwise_kernel_size
        self.layer_norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, self.kernel_size, groups=dim, bias=False
        )
        self.depthwise_layer_norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, 1, bias=False)

    def __call__(self, x, attention_mask=None):
        x = self.layer_norm(x)
        if attention_mask is not None:
            x = mx.where(attention_mask[:, :, None], x, 0)
        a, b = mx.split(self.pointwise_conv1(x), 2, axis=-1)
        x = a * mx.sigmoid(b)
        x = mx.pad(x, ((0, 0), (self.kernel_size - 1, 0), (0, 0)))
        x = self.depthwise_layer_norm(self.depthwise_conv(x))
        return self.pointwise_conv2(nn.silu(x))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, eps = config.hidden_size, config.layer_norm_eps
        self.ffn1_layer_norm = nn.LayerNorm(dim, eps=eps)
        self.ffn1 = FeedForward(config)
        self.self_attn_layer_norm = nn.LayerNorm(dim, eps=eps)
        self.self_attn = RelativeAttention(config)
        self.conv_module = ConvolutionModule(config)
        self.ffn2_layer_norm = nn.LayerNorm(dim, eps=eps)
        self.ffn2 = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(dim, eps=eps)

    def __call__(self, x, attention_mask=None):
        x = x + 0.5 * self.ffn1(self.ffn1_layer_norm(x))
        x = x + self.self_attn(self.self_attn_layer_norm(x), attention_mask)
        x = x + self.conv_module(x, attention_mask)
        x = x + 0.5 * self.ffn2(self.ffn2_layer_norm(x))
        return self.final_layer_norm(x)


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.feature_projection = FeatureProjection(config)
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, input_features, attention_mask=None):
        x = self.feature_projection(input_features)
        if attention_mask is not None:
            x = mx.where(attention_mask[:, :, None], x, 0)
        for layer in self.layers:
            x = layer(x, attention_mask)
            mx.eval(x)
        return x
