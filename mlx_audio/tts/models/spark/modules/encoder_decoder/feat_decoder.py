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


from collections import OrderedDict
from typing import List

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.vocos.vocos import VocosBackbone
from mlx_audio.tts.models.spark.modules.blocks.sampler import SamplingBlock


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
        modules = []
        for ratio in sample_ratios:
            module_list = [
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
                ),
            ]
            modules.append(module_list)

        self.downsample = modules

        self.vocos_backbone = VocosBackbone(
            input_channels=vocos_dim,
            dim=vocos_dim,
            intermediate_dim=vocos_intermediate_dim,
            num_layers=vocos_num_layers,
            adanorm_num_embeddings=condition_dim,
        )
        self.linear = nn.Linear(vocos_dim, out_channels)
        self.use_tanh_at_final = use_tanh_at_final

    def __call__(self, x: mx.array, c: mx.array = None):
        """encoder forward.

        Args:
            x (mx.array): (batch_size, input_channels, length)

        Returns:
            x (mx.array): (batch_size, encode_channels, length)
        """
        x = self.linear_pre(x.transpose(0, 2, 1))
        for modules in self.downsample:
            for module in modules:
                x = module(x)

        x = self.vocos_backbone(x.transpose(0, 2, 1), bandwidth_id=c)
        x = self.linear(x).transpose(0, 2, 1)
        if self.use_tanh_at_final:
            x = mx.tanh(x)

        return x


# test
if __name__ == "__main__":
    test_input = mx.random.normal(
        (8, 1024, 50), dtype=mx.float32
    )  # Batch size = 8, 1024 channels, length = 50
    condition = mx.random.randint(0, 100, (256, 8))  # 8, 256
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
    if output.shape == (8, 256, 200):
        print("Decoder test passed")
    else:
        print("Decoder test failed")
