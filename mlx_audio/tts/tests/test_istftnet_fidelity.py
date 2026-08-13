"""Fidelity regression tests for the iSTFTNet ports (kokoro / kitten_tts).

Pins the two numerically-checkable divergences fixed against the PyTorch
reference implementation: the grouped-ConvTranspose1d emulation alignment and
the COLA (window-squared) iSTFT normalization.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.tts.models.kitten_tts.istftnet import MLXSTFT as KittenMLXSTFT
from mlx_audio.tts.models.kitten_tts.istftnet import ConvWeighted as KittenConvWeighted
from mlx_audio.tts.models.kokoro.istftnet import MLXSTFT as KokoroMLXSTFT
from mlx_audio.tts.models.kokoro.istftnet import ConvWeighted as KokoroConvWeighted


@pytest.mark.parametrize("conv_cls", [KokoroConvWeighted, KittenConvWeighted])
def test_conv_transpose_emulation_matches_torch(conv_cls):
    # torch.nn.ConvTranspose1d(1, 1, 3, stride=2, padding=1, output_padding=1)
    # with weight [1, 2, 3] and input [1, 2, 3, 4] yields the values below.
    # The AdainResBlk1d pool emulates it as unpadded-conv-then-slice[1:].
    conv = conv_cls(1, 1, kernel_size=3, stride=2, padding=0)
    conv.weight_v = mx.array([1.0, 2.0, 3.0]).reshape(1, 3, 1)
    conv.weight_g = mx.array([float(np.sqrt(14.0))]).reshape(1, 1, 1)

    x = mx.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1)  # (B, T, C)
    out = conv(x, mx.conv_transpose1d)[:, 1:, :]

    expected = [2.0, 5.0, 4.0, 9.0, 6.0, 13.0, 8.0, 12.0]
    np.testing.assert_allclose(np.array(out).reshape(-1), expected, rtol=1e-4)


@pytest.mark.parametrize("stft_cls", [KokoroMLXSTFT, KittenMLXSTFT])
def test_mlxstft_roundtrip_preserves_amplitude(stft_cls):
    # transform -> inverse must reconstruct the signal at unity gain.
    # Plain-window (non-COLA) normalization reconstructs at sum(w^2)/sum(w)
    # = 0.75 for this periodic-hann win=4*hop configuration.
    stft = stft_cls(filter_length=20, hop_length=5, win_length=20)
    t = np.arange(2000, dtype=np.float32)
    x = (0.5 * np.sin(2 * np.pi * 220 * t / 24000)).astype(np.float32)

    recon = np.array(stft.inverse(*stft.transform(mx.array(x)[None, :])))
    recon = recon.reshape(-1)[: x.shape[0]]

    np.testing.assert_allclose(recon[20:-20], x[20:-20], atol=1e-3)
