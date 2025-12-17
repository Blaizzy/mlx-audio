import mlx.core as mx
import mlx.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Positionwise feed forward layer.

    Feed forward applied on each position of the sequence.
    The output dim is same as the input dim.
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: nn.Module = None,
    ):
        """
        Args:
            idim: Input dimension
            hidden_units: Number of hidden units
            dropout_rate: Dropout rate
            activation: Activation function (defaults to ReLU)
        """
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation if activation is not None else nn.ReLU()
        self.dropout_rate = dropout_rate
        self.w_2 = nn.Linear(hidden_units, idim)

    def __call__(self, xs: mx.array) -> mx.array:
        """
        Forward function.

        Args:
            xs: Input tensor (B, L, D)

        Returns:
            Output tensor (B, L, D)
        """
        x = self.w_1(xs)
        x = self.activation(x)

        if self.training and self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x)

        return self.w_2(x)
