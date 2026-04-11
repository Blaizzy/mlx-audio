import math

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.dacvae.codec import Snake1d, WNConv1d


class ResidualUnit(nn.Module):
    """Residual unit with dilated conv, matching keys res_unitN.snake1/conv1/snake2/conv2."""

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.snake1 = Snake1d(dim)
        self.conv1 = WNConv1d(
            dim,
            dim,
            kernel_size=7,
            dilation=dilation,
            pad_mode="none",
            norm="none",
        )
        self.snake2 = Snake1d(dim)
        self.conv2 = WNConv1d(dim, dim, kernel_size=1, norm="none")

    def __call__(self, x: mx.array) -> mx.array:
        y = self.snake1(x)
        y = self.conv1(y)
        y = self.snake2(y)
        y = self.conv2(y)
        # Trim x to match y if lengths differ due to padding
        pad = (x.shape[1] - y.shape[1]) // 2
        if pad > 0:
            x = x[:, pad:-pad, :]
        return x + y


class AcousticEncoderBlock(nn.Module):
    """Encoder block: 3 residual units + snake + strided conv downsampling.

    Matches keys: block.N.res_unit1/2/3, block.N.snake1, block.N.conv1
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        self.res_unit1 = ResidualUnit(in_dim, dilation=1)
        self.res_unit2 = ResidualUnit(in_dim, dilation=3)
        self.res_unit3 = ResidualUnit(in_dim, dilation=9)
        self.snake1 = Snake1d(in_dim)
        pad = math.ceil(stride / 2)
        self.conv1 = WNConv1d(
            in_dim,
            out_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=pad,
            norm="none",
        )
        # WNConv1d with pad_mode="none" overrides padding via
        # (kernel-stride)//2 which is wrong for odd strides.
        # Force the correct value here.
        self.conv1.padding = pad

    def __call__(self, x: mx.array) -> mx.array:
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        x = self.snake1(x)
        x = self.conv1(x)
        return x


class AcousticDecoderBlock(nn.Module):
    """Decoder block: snake + transposed conv upsample + 3 residual units.

    Matches keys: block.N.snake1, block.N.conv_t1, block.N.res_unit1/2/3

    Uses nn.ConvTranspose1d with padding=stride//2. For odd strides the raw output
    is 1 sample longer than stride*T_in, so we trim in the forward pass.
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        self._stride = stride
        self.snake1 = Snake1d(in_dim)
        self.conv_t1 = nn.ConvTranspose1d(
            in_dim,
            out_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
        )
        self.res_unit1 = ResidualUnit(out_dim, dilation=1)
        self.res_unit2 = ResidualUnit(out_dim, dilation=3)
        self.res_unit3 = ResidualUnit(out_dim, dilation=9)

    def __call__(self, x: mx.array) -> mx.array:
        t_in = x.shape[1]
        x = self.snake1(x)
        x = self.conv_t1(x)
        # Trim to exact expected length to handle odd-stride rounding
        expected = t_in * self._stride
        if x.shape[1] > expected:
            x = x[:, :expected, :]
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        return x


class AcousticEncoder(nn.Module):
    """DAC-style acoustic encoder: waveform [B,T,1] → latent [B,T//960,256].

    Matches keys: acoustic_encoder.conv1, .block.N.*, .snake, .conv2
    """

    # Encoder channel progression: 1→64→128→256→512→1024→2048→256
    _STRIDES = [8, 5, 4, 2, 3]
    _CHANNELS = [64, 128, 256, 512, 1024, 2048]

    def __init__(self):
        super().__init__()
        self.conv1 = WNConv1d(1, 64, kernel_size=7, padding=3, norm="none")

        self.block = [
            AcousticEncoderBlock(
                self._CHANNELS[i], self._CHANNELS[i + 1], self._STRIDES[i]
            )
            for i in range(len(self._STRIDES))
        ]

        self.snake1 = Snake1d(2048)
        self.conv2 = WNConv1d(2048, 256, kernel_size=3, padding=1, norm="none")

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        for blk in self.block:
            x = blk(x)
        x = self.snake1(x)
        x = self.conv2(x)
        return x


class AcousticDecoder(nn.Module):
    """DAC-style acoustic decoder: latent [B,T_tok,256] → waveform [B,T_tok*960,1].

    Matches keys: acoustic_decoder.conv1, .block.N.*, .snake1, .conv2
    """

    # Decoder channel progression: 256→1024→512→256→128→64→32→1
    _STRIDES = [8, 5, 4, 2, 3]
    _IN_CHANNELS = [1024, 512, 256, 128, 64]
    _OUT_CHANNELS = [512, 256, 128, 64, 32]

    def __init__(self):
        super().__init__()
        self.conv1 = WNConv1d(256, 1024, kernel_size=7, padding=3, norm="none")

        self.block = [
            AcousticDecoderBlock(
                self._IN_CHANNELS[i], self._OUT_CHANNELS[i], self._STRIDES[i]
            )
            for i in range(len(self._STRIDES))
        ]

        self.snake1 = Snake1d(32)
        self.conv2 = WNConv1d(32, 1, kernel_size=7, padding=3, norm="none")

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        for blk in self.block:
            x = blk(x)
        x = self.snake1(x)
        x = self.conv2(x)
        return x


class VectorQuantizer(nn.Module):
    """Single VQ codebook with project_in/codebook/project_out.

    Matches keys: quantizer.quantizers.N.project_in, .codebook, .project_out
    """

    def __init__(
        self, latent_dim: int = 1024, codebook_size: int = 1024, codebook_dim: int = 64
    ):
        super().__init__()
        self.project_in = nn.Linear(latent_dim, codebook_dim, bias=True)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.project_out = nn.Linear(codebook_dim, latent_dim, bias=True)

    def decode_codes(self, codes: mx.array) -> mx.array:
        """codes: [B, T] int32 → [B, T, latent_dim]"""
        return self.project_out(self.codebook(codes))

    def encode(self, z: mx.array) -> mx.array:
        """z: [B, T, latent_dim] → [B, T] int32 nearest-neighbor indices"""
        z_q = self.project_in(z)  # [B, T, codebook_dim]
        # Compute squared distances to each codebook entry
        dists = (
            mx.sum(z_q**2, axis=-1, keepdims=True)
            + mx.sum(self.codebook.weight**2, axis=-1)
            - 2 * (z_q @ self.codebook.weight.T)
        )
        return mx.argmin(dists, axis=-1).astype(mx.int32)

    def __call__(self, z: mx.array) -> mx.array:
        idx = self.encode(z)
        return self.decode_codes(idx)


class ResidualVectorQuantizer(nn.Module):
    """8-codebook residual vector quantizer.

    Matches keys: quantizer.quantizers.0..7.*
    """

    def __init__(
        self,
        n_codebooks: int = 8,
        latent_dim: int = 1024,
        codebook_size: int = 1024,
        codebook_dim: int = 64,
    ):
        super().__init__()
        self.quantizers = [
            VectorQuantizer(latent_dim, codebook_size, codebook_dim)
            for _ in range(n_codebooks)
        ]

    def decode(self, codes: mx.array) -> mx.array:
        """codes: [B, T, n_codebooks] int32 → [B, T, latent_dim]"""
        return sum(
            self.quantizers[i].decode_codes(codes[:, :, i])
            for i in range(len(self.quantizers))
        )

    def encode(self, z: mx.array) -> mx.array:
        """z: [B, T, latent_dim] → [B, T, n_codebooks] int32 via greedy residual quantization"""
        tokens = []
        residual = z
        for vq in self.quantizers:
            idx = vq.encode(residual)
            tokens.append(idx)
            recon = vq.decode_codes(idx)
            residual = residual - recon
        return mx.stack(tokens, axis=-1).astype(mx.int32)

    def __call__(self, z: mx.array) -> mx.array:
        codes = self.encode(z)
        return self.decode(codes)
