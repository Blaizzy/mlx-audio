"""Attention and transformer modules for MeloTTS."""

import math

import mlx.core as mx
import mlx.nn as nn

from .hifigan import Conv1dPT


class LayerNorm(nn.Module):
    """Channel-first layer norm (normalizes over C in (B, C, T))."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x):
        # x: (B, C, T) -> normalize over C
        mean = mx.mean(x, axis=1, keepdims=True)
        variance = mx.var(x, axis=1, keepdims=True)
        x = (x - mean) / mx.sqrt(variance + self.eps)
        return self.weight[None, :, None] * x + self.bias[None, :, None]


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.k_channels = channels // n_heads

        self.conv_q = Conv1dPT(channels, channels, 1)
        self.conv_k = Conv1dPT(channels, channels, 1)
        self.conv_v = Conv1dPT(channels, channels, 1)
        self.conv_o = Conv1dPT(channels, out_channels, 1)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = (
                mx.random.normal(
                    shape=(n_heads_rel, window_size * 2 + 1, self.k_channels)
                )
                * rel_stddev
            )
            self.emb_rel_v = (
                mx.random.normal(
                    shape=(n_heads_rel, window_size * 2 + 1, self.k_channels)
                )
                * rel_stddev
            )

    def __call__(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, _ = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s = key.shape
        t_t = query.shape[2]

        query = query.reshape(b, self.n_heads, self.k_channels, t_t).transpose(
            0, 1, 3, 2
        )
        key = key.reshape(b, self.n_heads, self.k_channels, t_s).transpose(0, 1, 3, 2)
        value = value.reshape(b, self.n_heads, self.k_channels, t_s).transpose(
            0, 1, 3, 2
        )

        query_scaled = query / math.sqrt(self.k_channels)
        scores = mx.matmul(query_scaled, key.transpose(0, 1, 3, 2))

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query_scaled, key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local

        if self.proximal_bias:
            scores = scores + self._attention_bias_proximal(t_s)

        if mask is not None:
            scores = mx.where(mask == 0, -1e4, scores)

        p_attn = mx.softmax(scores, axis=-1)

        output = mx.matmul(p_attn, value)

        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )

        output = output.transpose(0, 1, 3, 2).reshape(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        # y: (n_heads_rel, 2*L-1, d_k) or (2*L-1, d_k) if heads_share
        y_sel = y[0] if y.ndim == 3 else y
        return mx.matmul(x, y_sel)

    def _matmul_with_relative_keys(self, x, y):
        y_sel = y[0] if y.ndim == 3 else y
        return mx.matmul(x, y_sel.T)

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            padded = mx.pad(
                relative_embeddings,
                [(0, 0), (pad_length, pad_length), (0, 0)],
            )
        else:
            padded = relative_embeddings
        return padded[:, slice_start:slice_end]

    def _relative_position_to_absolute_position(self, x):
        b, heads, length, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1)])
        x_flat = x.reshape(b, heads, length * 2 * length)
        x_flat = mx.pad(x_flat, [(0, 0), (0, 0), (0, length - 1)])
        x_final = x_flat.reshape(b, heads, length + 1, 2 * length - 1)
        x_final = x_final[:, :, :length, length - 1 :]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        b, heads, length, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, length - 1)])
        x_flat = x.reshape(b, heads, length**2 + length * (length - 1))
        x_flat = mx.pad(x_flat, [(0, 0), (0, 0), (length, 0)])
        x_final = x_flat.reshape(b, heads, length, 2 * length)
        x_final = x_final[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        r = mx.arange(length, dtype=mx.float32)
        diff = r[None, :] - r[:, None]
        return -(mx.log1p(mx.abs(diff)))[None, None, :, :]


class FFN(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.causal = causal
        padding = self._causal_padding() if causal else (kernel_size - 1) // 2

        self.conv_1 = Conv1dPT(
            in_channels, filter_channels, kernel_size, padding=padding
        )
        self.conv_2 = Conv1dPT(
            filter_channels, out_channels, kernel_size, padding=padding
        )
        self.drop = nn.Dropout(p_dropout)
        self.activation = activation

    def _causal_padding(self):
        return self.kernel_size - 1

    def __call__(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        if self.activation == "gelu":
            x = x * mx.sigmoid(1.702 * x)
        else:
            x = nn.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(nn.Module):
    """Transformer encoder with relative position bias and optional speaker conditioning."""

    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=4,
        gin_channels=0,
        cond_layer_idx=2,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.cond_layer_idx = (
            min(cond_layer_idx, n_layers) if gin_channels > 0 else n_layers
        )
        self.drop = nn.Dropout(p_dropout)

        self.attn_layers = []
        self.norm_layers_1 = []
        self.ffn_layers = []
        self.norm_layers_2 = []

        for i in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

        if gin_channels > 0:
            self.spk_emb_linear = nn.Linear(gin_channels, hidden_channels)

    def __call__(self, x, x_mask, g=None):
        attn_mask = mx.expand_dims(x_mask, 2) * mx.expand_dims(x_mask, -1)
        x = x * x_mask

        for i in range(self.n_layers):
            if i == self.cond_layer_idx and g is not None and self.gin_channels > 0:
                g_proj = self.spk_emb_linear(g.transpose(0, 2, 1)).transpose(0, 2, 1)
                x = (x + g_proj) * x_mask

            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)

        x = x * x_mask
        return x
