"""Quantizer modules for the MOSS audio tokenizer."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.mimi.modules.conv import Conv1d

from .config import MossAudioTokenizerQuantizerConfig


def _l2_normalize(x: mx.array, axis: int) -> mx.array:
    norm = mx.sqrt(mx.sum(x.astype(mx.float32) ** 2, axis=axis, keepdims=True) + 1e-12)
    return x / norm


def _mask_from_lengths(lengths: mx.array, max_time: int, dtype) -> mx.array:
    mask = mx.arange(max_time, dtype=lengths.dtype)[None, :] < lengths[:, None]
    return mask.astype(dtype)[:, None, :]


class MossAudioTokenizerVectorQuantize(nn.Module):
    """Single RVQ codebook (inference-only)."""

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        if input_dim != codebook_dim:
            self.in_proj = Conv1d(input_dim, codebook_dim, 1, bias=True)
            self.out_proj = Conv1d(codebook_dim, input_dim, 1, bias=True)
        else:
            self.in_proj = None
            self.out_proj = None

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, indices: mx.array) -> mx.array:
        # (B, T) -> (B, D, T)
        quantized = self.codebook(indices).swapaxes(1, 2).astype(mx.float32)
        if self.out_proj is not None:
            quantized = self.out_proj(quantized)
        return quantized.astype(mx.float32)

    def __call__(self, z: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        z = z.astype(mx.float32)
        z_e = self.in_proj(z) if self.in_proj is not None else z
        encodings = z_e.swapaxes(1, 2).reshape(-1, z_e.shape[1]).astype(mx.float32)
        codebook = self.codebook.weight.astype(mx.float32)

        distances = (
            mx.sum(encodings**2, axis=1, keepdims=True)
            - 2.0 * (encodings @ codebook.transpose())
            + mx.sum(codebook**2, axis=1)[None, :]
        )
        indices = mx.argmax(-distances, axis=1).reshape(z.shape[0], -1).astype(mx.int32)
        z_q = self.decode_code(indices)
        return z_q, indices, z_e.astype(mx.float32)


class MossAudioTokenizerLFQ(nn.Module):
    """LFQ codebook used by ResidualLFQ."""

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        if input_dim != codebook_dim:
            self.in_proj = Conv1d(input_dim, codebook_dim, 1, bias=True)
            self.out_proj = Conv1d(codebook_dim, input_dim, 1, bias=True)
        else:
            self.in_proj = None
            self.out_proj = None

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code_wo_out_proj(self, indices: mx.array) -> mx.array:
        return self.codebook(indices).swapaxes(1, 2).astype(mx.float32)

    def decode_code(self, indices: mx.array) -> mx.array:
        z_q = self.decode_code_wo_out_proj(indices).astype(mx.float32)
        if self.out_proj is not None:
            z_q = self.out_proj(z_q)
        return z_q.astype(mx.float32)

    def _decode_latents(self, latents: mx.array) -> tuple[mx.array, mx.array]:
        encodings = latents.swapaxes(1, 2).reshape(-1, latents.shape[1]).astype(mx.float32)
        codebook = self.codebook.weight.astype(mx.float32)

        encodings = _l2_normalize(encodings, axis=-1)
        codebook = _l2_normalize(codebook, axis=-1)

        distances = (
            mx.sum(encodings**2, axis=1, keepdims=True)
            - 2.0 * (encodings @ codebook.transpose())
            + mx.sum(codebook**2, axis=1)[None, :]
        )
        indices = mx.argmax(-distances, axis=1).reshape(latents.shape[0], -1).astype(
            mx.int32
        )
        z_q = self.decode_code_wo_out_proj(indices)
        return z_q.astype(mx.float32), indices

    def __call__(self, z: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        z = z.astype(mx.float32)
        z_e = self.in_proj(z) if self.in_proj is not None else z
        z_q, indices = self._decode_latents(z_e.astype(mx.float32))
        if self.out_proj is not None:
            z_q = self.out_proj(z_q)
        return z_q.astype(mx.float32), indices, z_e.astype(mx.float32)


class MossAudioTokenizerResidualVQ(nn.Module):
    """Residual VQ stack."""

    def __init__(
        self,
        input_dim: int,
        rvq_dim: int,
        output_dim: int,
        num_quantizers: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rvq_dim = rvq_dim
        self.output_dim = output_dim
        self.num_quantizers = num_quantizers

        self.input_proj = (
            Conv1d(input_dim, rvq_dim, 1, bias=True) if input_dim != rvq_dim else None
        )
        self.output_proj = (
            Conv1d(rvq_dim, output_dim, 1, bias=True)
            if rvq_dim != output_dim
            else None
        )
        self.quantizers = [
            MossAudioTokenizerVectorQuantize(
                input_dim=rvq_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
            )
            for _ in range(num_quantizers)
        ]

    def __call__(
        self,
        z: mx.array,
        input_length: mx.array,
        n_quantizers: Optional[int] = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        z = self.input_proj(z) if self.input_proj is not None else z
        z = z.astype(mx.float32)
        batch_size, _, max_time = z.shape
        mask = _mask_from_lengths(input_length, max_time, z.dtype)

        quantized_out = mx.zeros_like(z).astype(mx.float32)
        residual = z.astype(mx.float32)
        all_indices = []
        n_quantizers = int(n_quantizers or self.num_quantizers)

        for quantizer in self.quantizers[:n_quantizers]:
            z_q_i, indices_i, _ = quantizer(residual * mask)
            quantized_out = quantized_out + z_q_i * mask
            residual = residual - z_q_i * mask
            all_indices.append(indices_i.astype(mx.int32))

        if all_indices:
            audio_codes = mx.stack(all_indices, axis=0)
        else:
            audio_codes = mx.zeros((0, batch_size, max_time), dtype=mx.int32)

        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out)
        return quantized_out.astype(mx.float32), audio_codes, input_length.astype(mx.int32)

    def decode_codes(self, codes: mx.array) -> mx.array:
        nq, batch_size, time_steps = codes.shape
        embeddings = mx.zeros((batch_size, self.rvq_dim, time_steps), dtype=mx.float32)
        for i, quantizer in enumerate(self.quantizers[:nq]):
            embeddings = embeddings + quantizer.decode_code(codes[i]).astype(mx.float32)
        if self.output_proj is not None:
            embeddings = self.output_proj(embeddings)
        return embeddings.astype(mx.float32)


class MossAudioTokenizerResidualLFQ(nn.Module):
    """Residual LFQ stack."""

    def __init__(
        self,
        input_dim: int,
        rvq_dim: int,
        output_dim: int,
        num_quantizers: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rvq_dim = rvq_dim
        self.output_dim = output_dim
        self.num_quantizers = num_quantizers

        self.input_proj = (
            Conv1d(input_dim, rvq_dim, 1, bias=True) if input_dim != rvq_dim else None
        )
        self.output_proj = (
            Conv1d(rvq_dim, output_dim, 1, bias=True)
            if rvq_dim != output_dim
            else None
        )
        self.quantizers = [
            MossAudioTokenizerLFQ(
                input_dim=rvq_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
            )
            for _ in range(num_quantizers)
        ]

    def __call__(
        self,
        z: mx.array,
        input_length: mx.array,
        n_quantizers: Optional[int] = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        z = self.input_proj(z) if self.input_proj is not None else z
        z = z.astype(mx.float32)
        batch_size, _, max_time = z.shape
        mask = _mask_from_lengths(input_length, max_time, z.dtype)

        quantized_out = mx.zeros_like(z).astype(mx.float32)
        residual = z.astype(mx.float32)
        all_indices = []
        n_quantizers = int(n_quantizers or self.num_quantizers)

        for quantizer in self.quantizers[:n_quantizers]:
            z_q_i, indices_i, _ = quantizer(residual * mask)
            quantized_out = quantized_out + z_q_i * mask
            residual = residual - z_q_i * mask
            all_indices.append(indices_i.astype(mx.int32))

        if all_indices:
            audio_codes = mx.stack(all_indices, axis=0)
        else:
            audio_codes = mx.zeros((0, batch_size, max_time), dtype=mx.int32)

        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out)
        return quantized_out.astype(mx.float32), audio_codes, input_length.astype(mx.int32)

    def decode_codes(self, codes: mx.array) -> mx.array:
        nq, batch_size, time_steps = codes.shape
        embeddings = mx.zeros((batch_size, self.rvq_dim, time_steps), dtype=mx.float32)
        for i, quantizer in enumerate(self.quantizers[:nq]):
            embeddings = embeddings + quantizer.decode_code(codes[i]).astype(mx.float32)
        if self.output_proj is not None:
            embeddings = self.output_proj(embeddings)
        return embeddings.astype(mx.float32)


def build_moss_audio_tokenizer_quantizer(
    config: MossAudioTokenizerQuantizerConfig,
) -> MossAudioTokenizerResidualLFQ | MossAudioTokenizerResidualVQ:
    quantizer_type = config.quantizer_type
    if quantizer_type in {"rlfq", "random_prefix_rlfq"}:
        return MossAudioTokenizerResidualLFQ(
            input_dim=config.input_dim,
            rvq_dim=config.rvq_dim,
            output_dim=config.output_dim,
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
        )
    if quantizer_type in {"rvq", "spec_rvq"}:
        return MossAudioTokenizerResidualVQ(
            input_dim=config.input_dim,
            rvq_dim=config.rvq_dim,
            output_dim=config.output_dim,
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
        )
    raise ValueError(f"Unsupported quantizer_type: {quantizer_type}")
