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


class Encoder(nn.Module):
    """Encoder module with convnext and downsampling blocks"""

    def __init__(
        self,
        input_channels: int,
        vocos_dim: int,
        vocos_intermediate_dim: int,
        vocos_num_layers: int,
        out_channels: int,
        sample_ratios: List[int] = [1, 1],
    ):
        super().__init__()
        """
        Encoder module with VocosBackbone and sampling blocks.

        Args:
            sample_ratios (List[int]): sample ratios
                example: [2, 2] means downsample by 2x and then upsample by 2x
        """
        self.encoder = VocosBackbone(
            input_channels=input_channels,
            dim=vocos_dim,
            intermediate_dim=vocos_intermediate_dim,
            num_layers=vocos_num_layers,
            condition_dim=None,
        )

        modules = [
            nn.Sequential(
                SamplingBlock(
                    dim=vocos_dim,
                    groups=vocos_dim,
                    downsample_scale=ratio,
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

        self.project = nn.Linear(vocos_dim, out_channels)

    def forward(self, x: torch.Tensor, *args):
        """
        Args:
            x (torch.Tensor): (batch_size, input_channels, length)

        Returns:
            x (torch.Tensor): (batch_size, encode_channels, length)
        """
        x = self.encoder(x)
        x = self.downsample(x)
        x = self.project(x)
        return x.transpose(1, 2)


# test
if __name__ == "__main__":
    test_input = torch.randn(8, 1024, 50)  # Batch size = 8, 1024 channels, length = 50
    encoder = Encoder(
        input_channels=1024,
        vocos_dim=384,
        vocos_intermediate_dim=2048,
        vocos_num_layers=12,
        out_channels=256,
        sample_ratios=[2, 2],
    )

    output = encoder(test_input)
    print(output.shape)  # torch.Size([8, 256, 12])
    if output.shape == torch.Size([8, 256, 12]):
        print("test successful")