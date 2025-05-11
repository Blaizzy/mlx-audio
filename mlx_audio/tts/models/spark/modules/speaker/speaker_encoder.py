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

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.spark.modules.residual_fsq import ResidualFSQ
from mlx_audio.tts.models.spark.modules.speaker.ecapa_tdnn import ECAPA_TDNN_GLOB_c512
from mlx_audio.tts.models.spark.modules.speaker.perceiver_encoder import (
    PerceiverResampler,
)

# from mlx_audio.codec.models.descript.nn.quantize import ResidualVectorQuantize


"""
x-vector + d-vector
"""


class SpeakerEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 100,
        out_dim: int = 512,
        latent_dim: int = 128,
        token_num: int = 32,
        fsq_levels: List[int] = [4, 4, 4, 4, 4, 4],
        fsq_num_quantizers: int = 1,
    ):
        super(SpeakerEncoder, self).__init__()

        self.speaker_encoder = ECAPA_TDNN_GLOB_c512(
            feat_dim=input_dim, embed_dim=out_dim
        )
        self.perceiver_sampler = PerceiverResampler(
            dim=latent_dim, dim_context=512 * 3, num_latents=token_num
        )
        self.quantizer = ResidualFSQ(
            dim=latent_dim,
            num_quantizers=fsq_num_quantizers,
            levels=fsq_levels,
            is_channel_first=True,
            quantize_dropout=False,
        )

        self.project = nn.Linear(latent_dim * token_num, out_dim)

    def get_codes_from_indices(self, indices: mx.array) -> mx.array:
        zq = self.quantizer.get_codes_from_indices(indices.transpose(1, 2))
        return zq.transpose(0, 2, 1)

    def get_indices(self, mels: mx.array) -> mx.array:
        mels = mels.transpose(0, 2, 1)
        x = self.perceiver_sampler(mels).transpose(0, 2, 1)
        zq, indices = self.quantizer(x)
        return indices

    def __call__(self, mels: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Args:
            mels: (B, D_mel, T1)

        Return:
            x_vector: (B, out_dim)
            d_vector: (B, out_dim)
        """
        # mels = mels.transpose(1,2)

        x_vector, features = self.speaker_encoder(mels, True)
        x = self.perceiver_sampler(features.transpose(0, 2, 1)).transpose(0, 2, 1)
        z_q, indices = self.quantizer(x)  # zq: (B, latent_dim, T2, latent_dim)
        x = z_q.reshape(z_q.shape[0], -1)
        d_vector = self.project(x)

        return x_vector, d_vector

    def tokenize(self, mels: mx.array) -> mx.array:
        """tokenize the input mel spectrogram"""
        _, features = self.speaker_encoder(mels, True)
        x = self.perceiver_sampler(features.transpose(0, 2, 1)).transpose(0, 2, 1)
        z_q, indices = self.quantizer(x)
        return indices

    def detokenize(self, indices: mx.array) -> mx.array:
        zq = self.quantizer.get_output_from_indices(indices.swapaxes(-1, -2)).swapaxes(
            -1, -2
        )
        x = zq.reshape(zq.shape[0], -1)
        d_vector = self.project(x)
        return d_vector

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if (
                ".conv.weight" in k
                or ("convs." in k and "weight" in k)
                or ("speaker_encoder.pool.linear" in k and "weight" in k)
            ):
                if v.shape[1] > v.shape[-1]:
                    sanitized_weights[k] = v.transpose(0, 2, 1)
                else:
                    sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v
        return sanitized_weights


if __name__ == "__main__":
    from mlx.utils import tree_flatten

    model = SpeakerEncoder(
        input_dim=100,
        latent_dim=128,
        token_num=32,
        fsq_levels=[4, 4, 4, 4, 4, 4],
        fsq_num_quantizers=1,
    )
    mel = mx.random.normal(shape=(8, 200, 100), scale=1.0)
    x_vector, d_vector = model(mel)
    print("x-vector shape", x_vector.shape)
    print("d-vector shape", d_vector.shape)

    indices = model.tokenize(mel)
    print("indices shape", indices.shape)
    d_vector_post = model.detokenize(indices)
    print("d-vector shape", d_vector_post.shape)
    if d_vector_post.all() == d_vector.all():
        print("d-vector post and d-vector are the same")
    else:
        print("d-vector post and d-vector are different")

    num_params = 0

    weights = dict(tree_flatten(model.parameters()))

    for k, v in weights.items():
        num_params += v.size
    print("{} M".format(num_params / 1e6))
