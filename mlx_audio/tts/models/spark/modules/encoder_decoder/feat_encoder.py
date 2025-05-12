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


from typing import List

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.vocos.vocos import VocosBackbone
from mlx_audio.tts.models.spark.modules.blocks.sampler import SamplingBlock


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
        )

        modules = []

        for ratio in sample_ratios:
            modules.append(
                [
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
                        bias=True,
                    ),
                ]
            )

        self.downsample = modules

        self.project = nn.Linear(vocos_dim, out_channels)

    def __call__(self, x: mx.array, *args):
        """
        Args:
            x (mx.array): (batch_size, input_channels, length)

        Returns:
            x (mx.array): (batch_size, encode_channels, length)
        """

        x = self.encoder(x)

        for modules in self.downsample:
            for module in modules:
                x = x.transpose(0, 2, 1)
                x = module(x)

        x = self.project(x)
        return x.transpose(0, 2, 1)

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "dwconv.weight" in k:
                if v.shape[1] < v.shape[-1]:
                    sanitized_weights[k] = v.transpose(0, 2, 1)
                else:
                    sanitized_weights[k] = v
            elif "embed.weight" in k:
                if v.shape[1] > v.shape[-1]:
                    sanitized_weights[k] = v.transpose(0, 2, 1)
                else:
                    sanitized_weights[k] = v

            else:
                sanitized_weights[k] = v

        return sanitized_weights


# test
if __name__ == "__main__":
    test_input = mx.random.normal(
        (8, 1024, 50), dtype=mx.float32
    )  # Batch size = 8, 1024 channels, length = 50
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
    if output.shape == (8, 256, 12):
        print("test successful")
    else:
        print("test failed")
