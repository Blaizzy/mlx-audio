from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.codec.models.bigvgan.bigvgan import BigVGAN, BigVGANConfig
from mlx_audio.codec.models.bigvgan.conv import WNConv1d
from mlx_audio.tts.models.indextts.ecapa_tdnn.ecapa_tdnn import ECPATDNN, ECPATDNNArgs


@dataclass
class BigVGANConditioningConfig(BigVGANConfig):
    gpt_dim: int = 1
    speaker_embedding_dim: int = 1
    cond_d_vector_in_each_upsampling_layer: bool = True


class BigVGANConditioning(BigVGAN):
    def __init__(self, config: BigVGANConditioningConfig):
        super().__init__(config)

        self.conv_pre = WNConv1d(
            config.gpt_dim, config.upsample_initial_channel, 7, 1, 3
        )

        self.cond_in_each_up_layer = config.cond_d_vector_in_each_upsampling_layer

        self.speaker_encoder = ECPATDNN(
            ECPATDNNArgs(config.num_mels, lin_neurons=config.speaker_embedding_dim)
        )
        self.cond_layer = nn.Conv1d(
            config.speaker_embedding_dim, config.upsample_initial_channel, 1
        )

        if config.cond_d_vector_in_each_upsampling_layer:
            self.conds = [
                nn.Conv1d(
                    config.speaker_embedding_dim,
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    1,
                )
                for i in range(len(self.ups))
            ]
        else:
            self.conds = []

    def __call__(
        self, x: mx.array, mel_refer: mx.array
    ) -> mx.array:  # (batch, num_mels, seq)
        x = x.transpose(0, 2, 1)
        mel_refer = mel_refer.transpose(0, 2, 1)

        speaker_embedding = self.speaker_encoder(mel_refer)

        x = self.conv_pre(x)
        x += self.cond_layer(speaker_embedding)

        for step in range(self.num_upsamples):
            for idx in range(len(self.ups[step])):
                x = self.ups[step][idx](x)

            if self.cond_in_each_up_layer:
                x += self.conds[step](speaker_embedding)

            xs = self.resblocks[step * self.num_kernels](x)
            for idx in range(1, self.num_kernels):
                xs += self.resblocks[step * self.num_kernels + idx](x)

            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)

        if self.use_tanh_at_final:
            x = mx.tanh(x)
        else:
            x = mx.clip(x, -1.0, 1.0)

        return x.transpose(0, 2, 1)

    def sanitize(self, weights: dict[str, mx.array]):
        new_weights = {}

        curr_weights = dict(tree_flatten(self.parameters()))

        for key, value in weights.items():
            if "num_batches_tracked" in key:
                continue

            key = (
                key.replace("norm.norm", "norm")
                .replace("conv.conv", "conv")
                .replace("conv1.conv", "conv1")
                .replace("conv2.conv", "conv2")
                .replace("fc.conv", "fc")
                .replace("asp_bn.norm", "asp_bn")
            )

            if (
                "conv" in key
                or "cond_layer" in key
                or "lowpass.filter" in key
                or "upsample.filter" in key
                or "conds" in key
                or "fc" in key
            ):
                if value.ndim == 3:
                    if value.shape != curr_weights[key].shape:
                        value = value.transpose(0, 2, 1)
                elif value.ndim == 4:
                    if value.shape != curr_weights[key].shape:
                        value = value.transpose(0, 2, 3, 1)

            if "ups." in key:
                if value.ndim == 3:
                    if value.shape != curr_weights[key].shape:
                        value = value.transpose(1, 2, 0)

            new_weights[key] = value

        del curr_weights

        return new_weights
