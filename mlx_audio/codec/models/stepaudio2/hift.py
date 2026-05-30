from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.chatterbox.s3gen.f0_predictor import ConvRNNF0Predictor
from mlx_audio.tts.models.chatterbox.s3gen.hifigan import HiFTGenerator


class StepAudio2HiFTGenerator(HiFTGenerator):
    def __init__(self):
        super().__init__(
            sampling_rate=24000,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=ConvRNNF0Predictor(),
            use_interpolation=True,
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        from .convert import sanitize_hift_weights

        return sanitize_hift_weights(self, weights)

    def decode(self, x: mx.array, s: mx.array) -> mx.array:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = mx.concatenate([s_stft_real, s_stft_imag], axis=1)

        x = mx.swapaxes(x, 1, 2)
        x = self.conv_pre(x)
        x = mx.swapaxes(x, 1, 2)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)
            x = mx.swapaxes(x, 1, 2)
            x = self.ups[i](x)
            x = mx.swapaxes(x, 1, 2)

            if i == self.num_upsamples - 1:
                x = mx.concatenate([x[:, :, 1:2], x], axis=2)

            si = mx.swapaxes(s_stft, 1, 2)
            si = self.source_downs[i](si)
            si = mx.swapaxes(si, 1, 2)
            si = self.source_resblocks[i](si)
            x = x + si

            start_idx = i * self.num_kernels
            x = mx.mean(
                mx.stack(
                    [self.resblocks[start_idx + j](x) for j in range(self.num_kernels)],
                    axis=0,
                ),
                axis=0,
            )

        x = nn.leaky_relu(x)
        x = mx.swapaxes(x, 1, 2)
        x = self.conv_post(x)
        x = mx.swapaxes(x, 1, 2)

        n_fft_half = self.istft_params["n_fft"] // 2 + 1
        magnitude = mx.exp(x[:, :n_fft_half, :])
        phase = mx.sin(x[:, n_fft_half:, :])
        x = self._istft(magnitude, phase)
        return mx.clip(x, -self.audio_limit, self.audio_limit)
