# Ported from https://github.com/resemble-ai/chatterbox

import mlx.core as mx
import mlx.nn as nn


class ConvRNNF0Predictor(nn.Module):
    """
    Convolutional F0 (fundamental frequency / pitch) predictor.

    Predicts F0 from mel-spectrogram features using stacked convolutions.

    Weight structure matches PyTorch nn.Sequential naming:
    - condnet.0, condnet.2, condnet.4, condnet.6, condnet.8 (Conv1d layers)
    - ELU activations are at indices 1, 3, 5, 7, 9 (not saved in weights)
    """

    def __init__(
        self,
        num_class: int = 1,
        in_channels: int = 80,
        cond_channels: int = 512,
    ):
        """
        Args:
            num_class: Number of output classes (1 for F0 regression)
            in_channels: Number of input mel channels
            cond_channels: Number of conditioning channels
        """
        super().__init__()
        self.num_class = num_class

        # Stack of convolutional layers with ELU activation
        # Use list to match PyTorch nn.Sequential numbering: 0, 2, 4, 6, 8
        # (ELU layers are at 1, 3, 5, 7, 9 but not stored in weights)
        self.condnet = [
            nn.Conv1d(
                in_channels, cond_channels, kernel_size=3, padding=1
            ),  # condnet.0
            nn.Conv1d(
                cond_channels, cond_channels, kernel_size=3, padding=1
            ),  # condnet.2
            nn.Conv1d(
                cond_channels, cond_channels, kernel_size=3, padding=1
            ),  # condnet.4
            nn.Conv1d(
                cond_channels, cond_channels, kernel_size=3, padding=1
            ),  # condnet.6
            nn.Conv1d(
                cond_channels, cond_channels, kernel_size=3, padding=1
            ),  # condnet.8
        ]

        # Final classifier
        self.classifier = nn.Linear(cond_channels, num_class)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Predict F0 from mel-spectrogram.

        Args:
            x: Mel-spectrogram (B, C, T)

        Returns:
            F0 predictions (B, T)
        """
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
