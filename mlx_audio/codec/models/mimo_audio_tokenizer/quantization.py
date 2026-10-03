import mlx.core as mx
import mlx.nn as nn


class ResidualVectorQuantizer(nn.Module):
    """Inference-only Euclidean RVQ, with FP32 search and residual accumulation."""

    def __init__(self, dimension: int, sizes: list[int]):
        super().__init__()
        self.codebooks = [nn.Embedding(size, dimension) for size in sizes]

    def encode(self, features: mx.array, num_quantizers: int | None = None):
        count = len(self.codebooks) if num_quantizers is None else num_quantizers
        if not 1 <= count <= len(self.codebooks):
            raise ValueError("num_quantizers is outside the codec's codebook range")
        residual = features.astype(mx.float32)
        codes = []
        for codebook in self.codebooks[:count]:
            weight = codebook.weight.astype(mx.float32)
            # The omitted squared input norm is constant across codebook entries.
            distance = mx.sum(weight * weight, axis=-1) - 2 * (residual @ weight.T)
            indices = mx.argmin(distance, axis=-1)
            residual = residual - weight[indices]
            codes.append(indices)
        return mx.stack(codes, axis=0)

    def decode(self, codes: mx.array):
        if codes.ndim != 2 or not 1 <= codes.shape[0] <= len(self.codebooks):
            raise ValueError("Codes must have shape (codebooks, frames)")
        values = [
            self.codebooks[i].weight.astype(mx.float32)[codes[i]]
            for i in range(codes.shape[0])
        ]
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result
