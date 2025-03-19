import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import BaseModelArgs
from .istftnet import AdainResBlk1d, ConvWeighted


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

    def __call__(self, x):
        return self.linear_layer(x)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = []
        for _ in range(depth):
            self.cnn.append(
                [
                    ConvWeighted(
                        channels, channels, kernel_size=kernel_size, padding=padding
                    ),
                    nn.LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                ]
            )
        # MLX doesn't have built-in LSTM, so we'll implement a simplified version
        self.lstm = LSTM(channels, channels // 2)

    def __call__(self, x, input_lengths, m):
        x = self.embedding(x)
        x = mx.transpose(x, (0, 2, 1))
        m = mx.expand_dims(m, axis=1)
        x = mx.where(m, 0.0, x)

        for conv in self.cnn:
            for layer in conv:
                if isinstance(layer, (ConvWeighted, nn.LayerNorm)):
                    x = x.swapaxes(2, 1)
                    x = (
                        layer(x, mx.conv1d)
                        if isinstance(layer, ConvWeighted)
                        else layer(x)
                    )
                    x = x.swapaxes(2, 1)
                else:
                    x = layer(x)

                x = mx.where(m, 0.0, x)

        x = x.swapaxes(2, 1)
        x, _ = self.lstm(x)
        x = x.swapaxes(2, 1)
        x_pad = mx.zeros([x.shape[0], x.shape[1], m.shape[-1]])
        x_pad[:, :, : x.shape[-1]] = x
        x = mx.where(m, 0.0, x_pad)
        return x


class AdaLayerNorm(nn.Module):
    # Works fine in MLX
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def __call__(self, x, s):
        h = self.fc(s)
        h = mx.reshape(h, (h.shape[0], h.shape[1], 1))
        gamma, beta = mx.split(h, 2, axis=1)
        gamma = gamma.transpose(2, 0, 1)
        beta = beta.transpose(2, 0, 1)

        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)

        return (1 + gamma) * x + beta


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        # Initialize scale for weight initialization
        scale = 1.0 / math.sqrt(hidden_size)

        # Forward direction weights and biases
        self.Wx_forward = mx.random.uniform(
            low=-scale, high=scale, shape=(4 * hidden_size, input_size)
        )
        self.Wh_forward = mx.random.uniform(
            low=-scale, high=scale, shape=(4 * hidden_size, hidden_size)
        )
        self.bias_ih_forward = (
            mx.random.uniform(low=-scale, high=scale, shape=(4 * hidden_size,))
            if bias
            else None
        )
        self.bias_hh_forward = (
            mx.random.uniform(low=-scale, high=scale, shape=(4 * hidden_size,))
            if bias
            else None
        )

        # Backward direction weights and biases
        self.Wx_backward = mx.random.uniform(
            low=-scale, high=scale, shape=(4 * hidden_size, input_size)
        )
        self.Wh_backward = mx.random.uniform(
            low=-scale, high=scale, shape=(4 * hidden_size, hidden_size)
        )
        self.bias_ih_backward = (
            mx.random.uniform(low=-scale, high=scale, shape=(4 * hidden_size,))
            if bias
            else None
        )
        self.bias_hh_backward = (
            mx.random.uniform(low=-scale, high=scale, shape=(4 * hidden_size,))
            if bias
            else None
        )

    def _extra_repr(self):
        return (
            f"input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, bias={self.bias}"
        )

    def _forward_direction(self, x, hidden=None, cell=None):
        """Process sequence in forward direction"""
        # Pre-compute input projections
        if self.bias_ih_forward is not None and self.bias_hh_forward is not None:
            x_proj = mx.addmm(
                self.bias_ih_forward + self.bias_hh_forward, x, self.Wx_forward.T
            )
        else:
            x_proj = x @ self.Wx_forward.T

        all_hidden = []
        all_cell = []

        seq_len = x.shape[-2]

        # Initialize hidden and cell states if not provided
        if hidden is None:
            hidden = mx.zeros((x.shape[0], self.hidden_size))
        if cell is None:
            cell = mx.zeros((x.shape[0], self.hidden_size))

        # Process sequence in forward direction (0 to seq_len-1)
        for idx in range(seq_len):
            ifgo = x_proj[..., idx, :]
            ifgo = ifgo + hidden @ self.Wh_forward.T

            # Split gates
            i, f, g, o = mx.split(ifgo, 4, axis=-1)

            # Apply activations
            i = mx.sigmoid(i)
            f = mx.sigmoid(f)
            g = mx.tanh(g)
            o = mx.sigmoid(o)

            # Update cell and hidden states
            cell = f * cell + i * g
            hidden = o * mx.tanh(cell)

            all_cell.append(cell)
            all_hidden.append(hidden)

        return mx.stack(all_hidden, axis=-2), mx.stack(all_cell, axis=-2)

    def _backward_direction(self, x, hidden=None, cell=None):
        """Process sequence in backward direction"""
        # Pre-compute input projections
        if self.bias_ih_backward is not None and self.bias_hh_backward is not None:
            x_proj = mx.addmm(
                self.bias_ih_backward + self.bias_hh_backward, x, self.Wx_backward.T
            )
        else:
            x_proj = x @ self.Wx_backward.T

        all_hidden = []
        all_cell = []

        seq_len = x.shape[-2]

        # Initialize hidden and cell states if not provided
        if hidden is None:
            hidden = mx.zeros((x.shape[0], self.hidden_size))
        if cell is None:
            cell = mx.zeros((x.shape[0], self.hidden_size))

        # Process sequence in backward direction (seq_len-1 to 0)
        for idx in range(seq_len - 1, -1, -1):
            ifgo = x_proj[..., idx, :]
            ifgo = ifgo + hidden @ self.Wh_backward.T

            # Split gates
            i, f, g, o = mx.split(ifgo, 4, axis=-1)

            # Apply activations
            i = mx.sigmoid(i)
            f = mx.sigmoid(f)
            g = mx.tanh(g)
            o = mx.sigmoid(o)

            # Update cell and hidden states
            cell = f * cell + i * g
            hidden = o * mx.tanh(cell)

            # Insert at beginning to maintain original sequence order
            all_cell.insert(0, cell)
            all_hidden.insert(0, hidden)

        return mx.stack(all_hidden, axis=-2), mx.stack(all_cell, axis=-2)

    def __call__(
        self,
        x,
        hidden_forward=None,
        cell_forward=None,
        hidden_backward=None,
        cell_backward=None,
    ):
        """
        Process input sequence in both directions and concatenate the results.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden_forward: Initial hidden state for forward direction
            cell_forward: Initial cell state for forward direction
            hidden_backward: Initial hidden state for backward direction
            cell_backward: Initial cell state for backward direction

        Returns:
            Tuple of:
                - Combined output hidden states (batch_size, seq_len, 2*hidden_size)
                - Tuple of final states ((forward_hidden, forward_cell), (backward_hidden, backward_cell))
        """

        if x.ndim == 2:
            x = mx.expand_dims(x, axis=0)  # (1, seq_len, input_size)

        # Forward direction
        forward_hidden, forward_cell = self._forward_direction(
            x, hidden_forward, cell_forward
        )

        # Backward direction
        backward_hidden, backward_cell = self._backward_direction(
            x, hidden_backward, cell_backward
        )

        # Concatenate outputs along the feature dimension
        output = mx.concatenate([forward_hidden, backward_hidden], axis=-1)

        # Return combined output and final states for both directions
        return output, (
            (forward_hidden[..., -1, :], forward_cell[..., -1, :]),
            (backward_hidden[..., 0, :], backward_cell[..., 0, :]),
        )


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )
        self.lstm = LSTM(d_hid + style_dim, d_hid // 2)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = LSTM(d_hid + style_dim, d_hid // 2)

        # F0 and N blocks
        self.F0 = [
            AdainResBlk1d(
                d_hid, d_hid, style_dim, dropout_p=dropout, conv_type=mx.conv1d
            ),
            AdainResBlk1d(
                d_hid,
                d_hid // 2,
                style_dim,
                upsample=True,
                dropout_p=dropout,
                conv_type=mx.conv1d,
            ),
            AdainResBlk1d(
                d_hid // 2,
                d_hid // 2,
                style_dim,
                dropout_p=dropout,
                conv_type=mx.conv1d,
            ),
        ]

        self.N = [
            AdainResBlk1d(
                d_hid, d_hid, style_dim, dropout_p=dropout, conv_type=mx.conv1d
            ),
            AdainResBlk1d(
                d_hid,
                d_hid // 2,
                style_dim,
                upsample=True,
                dropout_p=dropout,
                conv_type=mx.conv1d,
            ),
            AdainResBlk1d(
                d_hid // 2,
                d_hid // 2,
                style_dim,
                dropout_p=dropout,
                conv_type=mx.conv1d,
            ),
        ]

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, padding=0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, padding=0)

    def __call__(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        x, _ = self.lstm(d, text_lengths)

        # Apply dropout during inference
        x = mx.dropout(x, p=0.5)

        duration = self.duration_proj(x)
        en = mx.matmul(mx.transpose(d), alignment)
        return mx.squeeze(duration, axis=-1), en

    def F0Ntrain(self, x, s):
        x = mx.array(x)
        s = mx.array(s)
        x, _ = self.shared(mx.transpose(x, (0, 2, 1)))

        # F0 prediction
        F0 = mx.transpose(x, (0, 2, 1))
        for block in self.F0:
            F0 = block(F0, s)

        F0 = F0.swapaxes(2, 1)
        F0 = self.F0_proj(F0)
        F0 = F0.swapaxes(2, 1)

        # N prediction
        N = mx.transpose(x, (0, 2, 1))
        for block in self.N:
            N = block(N, s)
        N = N.swapaxes(2, 1)
        N = self.N_proj(N)
        N = N.swapaxes(2, 1)

        return mx.squeeze(F0, axis=1), mx.squeeze(N, axis=1)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = []
        for _ in range(nlayers):
            self.lstms.extend(
                [LSTM(d_model + sty_dim, d_model // 2), AdaLayerNorm(sty_dim, d_model)]
            )
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def __call__(self, x, style, text_lengths, m):
        x = x.transpose(2, 0, 1)
        s = mx.broadcast_to(style, (x.shape[0], x.shape[1], style.shape[-1]))
        x = mx.concatenate([x, s], axis=-1)
        x = mx.where(m[..., None].transpose(1, 0, 2), 0.0, x)
        x = x.transpose(1, 2, 0)

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(0, 2, 1), style).transpose(0, 2, 1)
                x = mx.concatenate([x, s.transpose(1, 2, 0)], axis=1)
                x = mx.where(m[..., None].transpose(0, 2, 1), 0.0, x)
            else:
                x = x.transpose(0, 2, 1)[0]
                x, _ = block(x)
                x = x.transpose(0, 2, 1)
                x_pad = mx.zeros([x.shape[0], x.shape[1], m.shape[-1]])
                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad
        return x.transpose(0, 2, 1)


# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
# TODO: Implement this in MLX


@dataclass
class AlbertModelArgs(BaseModelArgs):
    num_hidden_layers: int
    num_attention_heads: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    model_type: str = "albert"
    embedding_size: int = 128
    inner_group_num: int = 1
    num_hidden_groups: int = 1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    vocab_size: int = 30522
    dropout: float = 0.0


class AlbertEmbeddings(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = mx.arange(seq_length, dtype=mx.int32)[None, :]
        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertSelfAttention(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.reshape(new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def __call__(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = mx.matmul(query_layer, key_layer.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = mx.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = mx.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        context_layer = self.dense(context_layer)
        context_layer = self.LayerNorm(context_layer + hidden_states)

        return context_layer


class AlbertSelfOutput(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AlbertIntermediate(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class AlbertOutput(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AlbertLayer(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.attention = AlbertSelfAttention(config)

        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()

    def __call__(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ff_chunk(attention_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        return hidden_states

    def ff_chunk(self, attention_output: mx.array) -> mx.array:
        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        return ffn_output


class AlbertLayerGroup(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.albert_layers = [
            AlbertLayer(config) for _ in range(config.inner_group_num)
        ]

    def __call__(self, hidden_states, attention_mask=None):
        for layer_module in self.albert_layers:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class AlbertEncoder(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(
            config.embedding_size, config.hidden_size
        )
        self.albert_layer_groups = [
            AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)
        ]

    def __call__(self, hidden_states, attention_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(
                self.config.num_hidden_layers / self.config.num_hidden_groups
            )

            # Index of the hidden group
            group_idx = int(
                i / (self.config.num_hidden_layers / self.config.num_hidden_groups)
            )

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states, attention_mask
            )
            hidden_states = layer_group_output
        return hidden_states


class CustomAlbert(nn.Module):
    def __init__(self, config: AlbertModelArgs):
        super().__init__()
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = nn.tanh(self.pooler(sequence_output[:, 0]))

        return sequence_output, pooled_output

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights
