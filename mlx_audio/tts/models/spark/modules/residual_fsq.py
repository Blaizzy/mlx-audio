import random
from typing import List

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.spark.modules.finite_scalar_quantization import FSQ


def exists(val):
    return val is not None


def first(l):
    return l[0]


def default(val, d):
    return val if exists(val) else d


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


class ResidualFSQ(nn.Module):
    """Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(
        self,
        *,
        levels: List[int],
        num_quantizers,
        dim=None,
        is_channel_first=False,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        **kwargs,
    ):
        super().__init__()
        codebook_dim = len(levels)
        dim = default(dim, codebook_dim)

        requires_projection = codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.has_projections = requires_projection

        self.is_channel_first = is_channel_first
        self.num_quantizers = num_quantizers

        self.levels = levels
        self.layers = []

        # Convert ListConfig to a regular list before passing to mx.array
        levels_tensor = mx.array(list(levels))

        scales = []

        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** -ind)

            fsq = FSQ(levels=levels, dim=codebook_dim, **kwargs)

            self.layers.append(fsq)

        assert all([not fsq.has_projections for fsq in self.layers])

        self.codebook_size = self.layers[0].codebook_size

        self._scales = mx.array(scales)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

    @property
    def codebooks(self):
        codebooks = [layer._implicit_codebook for layer in self.layers]
        codebooks = mx.stack(codebooks, axis=0)
        return codebooks

    def get_codes_from_indices(self, indices):
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        # MLX doesn't have pack function, so we need to reshape manually
        original_shape = indices.shape
        indices = mx.reshape(indices, (indices.shape[0], -1, indices.shape[-1]))

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert (
                self.quantize_dropout > 0.0
            ), "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations"
            indices = mx.pad(
                indices,
                ((0, 0), (0, 0), (0, self.num_quantizers - quantize_dim)),
                constant_value=-1,
            )

        # take care of quantizer dropout

        mask = indices == -1
        # MLX doesn't have masked_fill, so we use where
        indices = mx.where(
            mask, mx.zeros_like(indices), indices
        )  # have it fetch a dummy code to be masked out later

        # MLX doesn't have get_at function, so we need to manually gather codes
        all_codes = []
        for q in range(self.codebooks.shape[0]):
            q_codes = []
            for b in range(indices.shape[0]):
                n_codes = []
                for n in range(indices.shape[1]):
                    idx = indices[b, n, q]
                    n_codes.append(self.codebooks[q, idx])
                q_codes.append(mx.stack(n_codes))
            all_codes.append(mx.stack(q_codes))
        all_codes = mx.stack(all_codes)[:, :, :, 0, :]  # Shape: (q, b, n, d)

        # mask out any codes that were dropout-ed
        # Reshape mask for broadcasting: q b n 1

        mask_reshaped = mx.reshape(
            mask, (mask.shape[2], mask.shape[0], mask.shape[1], 1)
        )

        all_codes = mx.where(mask_reshaped, mx.zeros_like(all_codes), all_codes)

        # scale the codes
        # Reshape scales for broadcasting: q 1 1 d
        scales = mx.reshape(
            self._scales, (self._scales.shape[0], 1, 1, self._scales.shape[1])
        )
        all_codes = all_codes * scales

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)
        # Reshape all_codes back to original dimensions
        if len(original_shape) > 3:  # If we had height, width dimensions
            all_codes = mx.reshape(
                all_codes,
                (
                    all_codes.shape[0],
                    original_shape[0],
                    *original_shape[1:-1],
                    all_codes.shape[-1],
                ),
            )

        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = mx.sum(codes, axis=0)
        return self.project_out(codes_summed)

    def __call__(
        self, x, return_all_codes=False, rand_quantize_dropout_fixed_seed=None
    ):
        num_quant, quant_dropout_multiple_of = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
        )

        # handle channel first

        if self.is_channel_first:
            # Manually implement rearrange and pack functionality
            # First, move dimension d from position 1 to the end
            shape = x.shape
            # Assuming shape is (b, d, ...)
            new_shape = (shape[0],) + shape[2:] + (shape[1],)
            x = mx.transpose(x, (0,) + tuple(range(2, len(shape))) + (1,))

            # Pack operation: flatten all dimensions between b and d
            # This is equivalent to pack([x], "b * d")
            ps = x.shape
            middle_dims = x.shape[1:-1]
            flattened_dim = 1
            for dim in middle_dims:
                flattened_dim *= dim
            x = mx.reshape(x, (x.shape[0], flattened_dim, x.shape[-1]))

        # maybe project in

        x = self.project_in(x)

        quantized_out = 0.0
        residual = x

        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices

        if should_quantize_dropout:

            # check if seed is manually passed in

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(
                self.quantize_dropout_cutoff_index, num_quant
            )

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    round_up_multiple(
                        rand_quantize_dropout_index + 1, quant_dropout_multiple_of
                    )
                    - 1
                )

            null_indices = mx.full(x.shape[:2], -1, dtype=mx.int32)

        # go through the layers
        for quantizer_index, (layer, scale) in enumerate(
            zip(self.layers, self._scales)
        ):

            if (
                should_quantize_dropout
                and quantizer_index > rand_quantize_dropout_index
            ):
                all_indices.append(null_indices)
                continue

            quantized, indices = layer(residual / scale)

            quantized = quantized * scale

            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # stack all indices

        all_indices = mx.stack(all_indices, axis=-1)

        # channel first out

        if self.is_channel_first:
            # MLX doesn't have unpack, so we need to reshape manually
            # Assuming ps contains the original batch dimensions
            # Reshape to combine all dimensions between batch and the last dimension
            batch_size = ps[0] if isinstance(ps, tuple) else ps
            quantized_out = mx.reshape(
                quantized_out, (batch_size, -1, quantized_out.shape[-1])
            ).swapaxes(
                2, 1
            )  # swap to match torch output
            all_indices = mx.reshape(
                all_indices, (batch_size, -1, all_indices.shape[-1])
            ).swapaxes(
                2, 1
            )  # swap to match torch output

        # return
        ret = (quantized_out, all_indices)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers

        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)


if __name__ == "__main__":
    model = ResidualFSQ(
        levels=[4, 4, 4, 4, 4, 4],
        num_quantizers=1,
        dim=30,
        is_channel_first=True,
        quantize_dropout=False,
    )
    x = mx.random.normal((2, 30, 10))
    quantize, embed_ind = model(x)

    emb_from_ind = model.get_output_from_indices(embed_ind.transpose(0, 2, 1))

    print(quantize == emb_from_ind.transpose(0, 2, 1))

    print("quantize shape", quantize.shape)
    print("embed_ind", embed_ind)
