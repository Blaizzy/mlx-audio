# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F


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
                nn.ConvTranspose1d(
                    dim,
                    dim,
                    kernel_size=upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
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
        return x.repeat_interleave(upsample_scale, dim=2)

    @staticmethod
    def skip_downsampler(x, downsample_scale):
        return F.avg_pool1d(x, kernel_size=downsample_scale, stride=downsample_scale)

    def forward(self, x):
        x = x.transpose(1, 2)
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

        return final_res


# test
if __name__ == "__main__":
    test_input = torch.randn(8, 1024, 50)  # Batch size = 8, 1024 channels, length = 50
    model = SamplingBlock(1024, 1024, upsample_scale=2)
    model_down = SamplingBlock(1024, 1024, downsample_scale=2)
    output = model(test_input)
    output_down = model_down(test_input)
    print("shape after upsample * 2", output.shape)  # torch.Size([8, 1024, 100])
    print("shape after downsample * 2", output_down.shape)  # torch.Size([8, 1024, 25])
    if output.shape == torch.Size([8, 1024, 100]) and output_down.shape == torch.Size(
        [8, 1024, 25]
    ):
        print("test successful")
