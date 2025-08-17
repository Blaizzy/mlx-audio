import math

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.descript.nn.layers import WNConvTranspose1d


class SamplingBlock(nn.Module):
    """Sampling block for upsampling or downsampling"""

    def __init__(
        self,
        dim: int,
        groups: int = 1,
        upsample_scale: int = 1,
        downsample_scale: int = 1,
    ) -> None:
        """
        Args:
            dim: input dimension
            groups: number of groups
            upsample_scale: upsampling scale
            downsample_scale: downsampling scale
        """
        super(SamplingBlock, self).__init__()

        self.upsample_scale = upsample_scale
        self.downsample_scale = downsample_scale

        if self.upsample_scale > 1:
            self.de_conv_upsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    dim,
                    dim,
                    kernel_size=upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    groups=groups,
                ),
            )

        if self.downsample_scale > 1:
            self.conv_downsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=2 * downsample_scale,
                    stride=downsample_scale,
                    padding=downsample_scale // 2 + downsample_scale % 2,
                    groups=groups,
                ),
            )

    @staticmethod
    def repeat_upsampler(x, upsample_scale):
        # MLX doesn't have repeat_interleave, so we need to implement it manually
        batch_size, seq_len, channels = x.shape
        # Create a new tensor with the expanded shape
        output = mx.zeros((batch_size, seq_len * upsample_scale, channels))
        # Fill the output tensor by repeating each element
        for i in range(seq_len):
            for j in range(upsample_scale):
                output[:, i * upsample_scale + j, :] = x[:, i, :]
        return output

    @staticmethod
    def skip_downsampler(x, downsample_scale):
        return nn.AvgPool1d(kernel_size=downsample_scale, stride=downsample_scale)(x)

    def __call__(self, x):
        x = x.transpose(0, 2, 1)
        if self.upsample_scale > 1:
            repeat_res = self.repeat_upsampler(x, self.upsample_scale)
            deconv_res = self.de_conv_upsampler(x)
            upmerge_res = repeat_res + deconv_res
        else:
            upmerge_res = x
            repeat_res = x

        if self.downsample_scale > 1:
            conv_res = self.conv_downsampler(upmerge_res)
            skip2_res = self.skip_downsampler(upmerge_res, self.downsample_scale)
            skip1_res = self.skip_downsampler(repeat_res, self.downsample_scale)
        else:
            conv_res = upmerge_res
            skip2_res = upmerge_res
            skip1_res = repeat_res

        final_res = conv_res + skip1_res + skip2_res

        return final_res.transpose(0, 2, 1)


# test
if __name__ == "__main__":
    test_input = mx.random.randint(
        0, 100, (8, 1024, 50)
    )  # Batch size = 8, 1024 channels, length = 50
    model = SamplingBlock(1024, 1024, upsample_scale=2)
    model_down = SamplingBlock(1024, 1024, downsample_scale=2)
    output = model(test_input)
    output_down = model_down(test_input)
    print("shape after upsample * 2", output.shape)  # torch.Size([8, 1024, 100])
    print("shape after downsample * 2", output_down.shape)  # torch.Size([8, 1024, 25])
    if output.shape == (8, 1024, 100) and output_down.shape == (8, 1024, 25):
        print("test successful")
    else:
        print("test failed")
