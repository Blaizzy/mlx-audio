# Copyright (c) 2021 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Bing Han (hanbing97@sjtu.edu.cn)
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

""" This implementation is adapted from github repo:
    https://github.com/lawlict/ECAPA-TDNN.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.spark.modules.speaker import pooling_layers as pooling_layers


class Res2Conv1dReluBn(nn.Module):
    """
    in_channels == out_channels == channels
    """

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        scale=4,
    ):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.channels = channels
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
            )
            self.bns.append(nn.BatchNorm(self.width))
        # self.convs = [*self.convs]  # nn.ModuleList(self.convs)
        # self.bns = [*self.bns]  # nn.ModuleList(self.bns)

    def __call__(self, x):
        out = []

        spx = mx.split(x, self.scale, axis=1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # Order: conv -> relu -> bn
            if i >= 1:
                sp = sp + spx[i]

            sp = conv(sp.transpose(0, 2, 1))
            sp = bn(nn.relu(sp)).transpose(0, 2, 1)
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = mx.concatenate(out, axis=1)
        return out


""" Conv1d + BatchNorm1d + ReLU
"""


class Conv1dReluBn(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x):
        x = self.conv(x.swapaxes(1, 2)).swapaxes(1, 2)
        x = nn.relu(x)
        x = self.bn(x.swapaxes(1, 2)).swapaxes(1, 2)
        return x


""" The SE connection of 1D case.
"""


class SE_Connect(nn.Module):

    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def __call__(self, x):
        out = mx.mean(x, axis=2)
        out = nn.relu(self.linear1(out))
        out = mx.sigmoid(self.linear2(out))
        out = x * out[:, :, None]
        return out


""" SE-Res2Block of the ECAPA-TDNN architecture.
"""


class SE_Res2Block(nn.Module):

    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()
        self.se_res2block = [
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            Res2Conv1dReluBn(
                channels, kernel_size, stride, padding, dilation, scale=scale
            ),
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            SE_Connect(channels),
        ]

    def __call__(self, x):
        res = x
        for module in self.se_res2block:
            x = module(x)
        return x + res


class ECAPA_TDNN(nn.Module):

    def __init__(
        self,
        channels=512,
        feat_dim=80,
        embed_dim=192,
        pooling_func="ASTP",
        global_context_att=False,
        emb_bn=False,
    ):
        super().__init__()

        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8
        )
        self.layer3 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8
        )
        self.layer4 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8
        )

        cat_channels = channels * 3
        out_channels = 512 * 3
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=out_channels, global_context_att=global_context_att
        )
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.emb_bn = emb_bn
        if emb_bn:  # better in SSL for SV
            self.bn2 = nn.BatchNorm(embed_dim)
        else:
            self.bn2 = nn.Identity()

    def __call__(self, x, return_latent=False):
        x = x.transpose(0, 2, 1)  # (B,T,F) -> (B,F,T)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = mx.concatenate([out2, out3, out4], axis=1)

        out = self.conv(out.transpose(0, 2, 1)).transpose(0, 2, 1)
        latent = nn.relu(out)
        out = self.pool(latent)
        out = self.bn(out)
        out = self.linear(out)
        if self.emb_bn:
            out = self.bn2(out)

        if return_latent:
            return out, latent
        return out


def ECAPA_TDNN_c1024(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    return ECAPA_TDNN(
        channels=1024,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_GLOB_c1024(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    return ECAPA_TDNN(
        channels=1024,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_c512(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    return ECAPA_TDNN(
        channels=512,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        emb_bn=emb_bn,
    )


def ECAPA_TDNN_GLOB_c512(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    return ECAPA_TDNN(
        channels=512,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        emb_bn=emb_bn,
    )


if __name__ == "__main__":
    from mlx.utils import tree_flatten

    x = mx.zeros(shape=(1, 200, 100))
    model = ECAPA_TDNN_GLOB_c512(feat_dim=100, embed_dim=256, pooling_func="ASTP")
    model.eval()
    out, latent = model(x, True)
    print(out.shape)
    print(latent.shape)
    # Count parameters for MLX model
    num_params = 0

    weights = dict(tree_flatten(model.parameters()))

    for k, v in weights.items():
        num_params += v.size
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 200, 80)
    # flops, params = profile(model, inputs=(x_np, ))
    # print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
