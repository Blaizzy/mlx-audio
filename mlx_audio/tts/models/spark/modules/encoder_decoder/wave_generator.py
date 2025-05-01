import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.descript.dac import (
    ResidualUnit,
    Snake1d,
    WNConv1d,
    WNConvTranspose1d,
)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        kernel_size: int = 2,
        stride: int = 1,
    ):
        super().__init__()
        self.block = [
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        ]

    def __call__(self, x):
        x = x.transpose(0, 2, 1)
        for module in self.block:
            x = module(x)
        return x.transpose(0, 2, 1)


class WaveGenerator(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        kernel_sizes,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, rates)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, kernel_size, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = layers

    def __call__(self, x):
        x = x.transpose(0, 2, 1)
        for module in self.model:
            x = module(x)
        return x.transpose(0, 2, 1)


if __name__ == "__main__":
    test_input = mx.random.normal((8, 1024, 50), dtype=mx.float32)
    wave_generator = WaveGenerator(1024, 16, [2, 2], [7, 7])
    output = wave_generator(test_input)
    print(output.shape)
    if output.shape == (8, 1, 203):
        print("WaveGenerator test passed")
    else:
        print("WaveGenerator test failed")
