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

from typing import List

from sparktts.modules.blocks.vocos import VocosBackbone
from sparktts.modules.blocks.samper import SamplingBlock


class Decoder(nn.Module):
    """Decoder module with convnext and upsampling blocks

    Args:
        sample_ratios (List[int]): sample ratios
            example: [2, 2] means downsample by 2x and then upsample by 2x
    """

    def __init__(
        self,
        input_channels: int,
        vocos_dim: int,
        vocos_intermediate_dim: int,
        vocos_num_layers: int,
        out_channels: int,
        condition_dim: int = None,
        sample_ratios: List[int] = [1, 1],
        use_tanh_at_final: bool = False,
    ):
        super().__init__()

        self.linear_pre = nn.Linear(input_channels, vocos_dim)
        modules = [
            nn.Sequential(
                SamplingBlock(
                    dim=vocos_dim,
                    groups=vocos_dim,
                    upsample_scale=ratio,
                ),
                VocosBackbone(
                    input_channels=vocos_dim,
                    dim=vocos_dim,
                    intermediate_dim=vocos_intermediate_dim,
                    num_layers=2,
                    condition_dim=None,
                ),
            )
            for ratio in sample_ratios
        ]

        self.downsample = nn.Sequential(*modules)

        self.vocos_backbone = VocosBackbone(
            input_channels=vocos_dim,
            dim=vocos_dim,
            intermediate_dim=vocos_intermediate_dim,
            num_layers=vocos_num_layers,
            condition_dim=condition_dim,
        )
        self.linear = nn.Linear(vocos_dim, out_channels)
        self.use_tanh_at_final = use_tanh_at_final

    def forward(self, x: torch.Tensor, c: torch.Tensor = None):
        """encoder forward.

        Args:
            x (torch.Tensor): (batch_size, input_channels, length)

        Returns:
            x (torch.Tensor): (batch_size, encode_channels, length)
        """
        x = self.linear_pre(x.transpose(1, 2))
        x = self.downsample(x).transpose(1, 2)
        x = self.vocos_backbone(x, condition=c)
        x = self.linear(x).transpose(1, 2)
        if self.use_tanh_at_final:
            x = torch.tanh(x)

        return x


# test
if __name__ == "__main__":
    test_input = torch.randn(8, 1024, 50)  # Batch size = 8, 1024 channels, length = 50
    condition = torch.randn(8, 256)
    decoder = Decoder(
        input_channels=1024,
        vocos_dim=384,
        vocos_intermediate_dim=2048,
        vocos_num_layers=12,
        out_channels=256,
        condition_dim=256,
        sample_ratios=[2, 2],
    )
    output = decoder(test_input, condition)
    print(output.shape)  # torch.Size([8, 256, 200])
    if output.shape == torch.Size([8, 256, 200]):
        print("Decoder test passed")
    else:
        print("Decoder test failed")