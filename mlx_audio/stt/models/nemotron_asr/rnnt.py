"""RNN-T prediction (decoder) and joint networks.

Layout mirrors NeMo's ``RNNTDecoder`` / ``RNNTJoint`` so checkpoint keys map directly:
``decoder.prediction.embed``, ``decoder.prediction.dec_rnn.lstm.{i}``,
``joint.enc``, ``joint.pred``, ``joint.joint_net.2``.
"""

import mlx.core as mx
import mlx.nn as nn

from .config import JointArgs, PredictArgs


class LSTM(nn.Module):
    """Multi-layer LSTM holding one ``nn.LSTM`` per layer under ``self.lstm``."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = [
            nn.LSTM(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ]

    def __call__(self, x: mx.array, h_c=None):
        # x: (B, L, input_size) -> internally time-major for nn.LSTM.
        x = mx.transpose(x, (1, 0, 2))
        if h_c is None:
            h = [None] * self.num_layers
            c = [None] * self.num_layers
        else:
            h, c = h_c

        outputs = x
        next_h, next_c = [], []
        for i in range(self.num_layers):
            all_h, all_c = self.lstm[i](outputs, hidden=h[i], cell=c[i])
            outputs = all_h
            next_h.append(all_h[-1])
            next_c.append(all_c[-1])

        outputs = mx.transpose(outputs, (1, 0, 2))
        return outputs, (mx.stack(next_h, axis=0), mx.stack(next_c, axis=0))


class PredictNetwork(nn.Module):
    def __init__(self, args: PredictArgs):
        super().__init__()
        self.pred_hidden = args.pred_hidden
        vocab = args.vocab_size + 1 if args.blank_as_pad else args.vocab_size
        self.prediction = {
            "embed": nn.Embedding(vocab, args.pred_hidden),
            "dec_rnn": LSTM(args.pred_hidden, args.pred_hidden, args.pred_rnn_layers),
        }

    def __call__(self, y: mx.array | None, h_c=None):
        if y is not None:
            embedded = self.prediction["embed"](y)
        else:
            batch = 1 if h_c is None else h_c[0].shape[1]
            embedded = mx.zeros((batch, 1, self.pred_hidden))
        return self.prediction["dec_rnn"](embedded, h_c)


class JointNetwork(nn.Module):
    def __init__(self, args: JointArgs):
        super().__init__()
        self._num_classes = args.num_classes + 1  # + blank

        act = args.activation.lower()
        if act == "relu":
            activation = nn.ReLU()
        elif act == "sigmoid":
            activation = nn.Sigmoid()
        elif act == "tanh":
            activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported joint activation: {args.activation}")

        self.enc = nn.Linear(args.encoder_hidden, args.joint_hidden)
        self.pred = nn.Linear(args.pred_hidden, args.joint_hidden)
        self.joint_net = [
            activation,
            nn.Identity(),
            nn.Linear(args.joint_hidden, self._num_classes),
        ]

    def __call__(self, enc: mx.array, pred: mx.array) -> mx.array:
        enc = self.enc(enc)
        pred = self.pred(pred)
        x = mx.expand_dims(enc, 2) + mx.expand_dims(pred, 1)
        for layer in self.joint_net:
            x = layer(x)
        return x
