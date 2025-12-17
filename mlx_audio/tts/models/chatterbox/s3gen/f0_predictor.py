import mlx.core as mx
import mlx.nn as nn


class ConvRNNF0Predictor(nn.Module):

    def __init__(
        self,
        num_class: int = 1,
        in_channels: int = 80,
        cond_channels: int = 512,
    ):

        super().__init__()
        self.num_class = num_class

        self.condnet = [
            nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1),
            nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
            nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
        ]

        # Final classifier
        self.classifier = nn.Linear(cond_channels, num_class)

    def __call__(self, x: mx.array) -> mx.array:

        # x is (B, C, T), but MLX Conv1d expects (B, T, C)
        x = mx.swapaxes(x, 1, 2)  # (B, C, T) -> (B, T, C)

        # Convolutional stack with ELU activations
        for conv in self.condnet:
            x = nn.elu(conv(x))

        # x is now (B, T, C) which is correct for linear layer

        # Classify and take absolute value
        x = self.classifier(x)
        x = mx.squeeze(x, axis=-1)  # (B, T, 1) -> (B, T)

        return mx.abs(x)
