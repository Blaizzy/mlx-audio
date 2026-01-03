import mlx.core as mx
import mlx.nn as nn


class ScaledSinuEmbedding(nn.Module):
    """MLX implementation of ScaledSinuEmbedding.

    ScaledSinuEmbedding provides sinusoidal positional encodings for inputs.
    It generates position embeddings using sine and cosine functions with
    different frequencies and applies learnable scaling.

    Performance optimizations:
    - Efficient broadcasting operations
    - Optional caching for repeated sequence lengths
    - Reduced memory allocations

    Arguments:
        dim: Dimension of the positional embeddings
        use_cache: Whether to cache embeddings for repeated sequence lengths
        max_cache_size: Maximum number of different sequence lengths to cache
    """

    def __init__(self, dim: int, use_cache: bool = True, max_cache_size: int = 50):
        super().__init__()
        self.dim = dim
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size

        # Initialize learnable scale parameter
        self.scale = mx.ones((1,))

        # Calculate inverse frequencies for sinusoidal embeddings
        # inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = mx.arange(0, dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (10000.0 ** (positions / dim))
        self.inv_freq = (
            inv_freq  # Store as instance variable (equivalent to register_buffer)
        )

        # Cache for computed embeddings
        self._embedding_cache = {}

    def __call__(self, x: mx.array) -> mx.array:
        """forward pass for the ScaledSinuEmbedding layer.

        Performance optimizations:
        - Use efficient broadcasting with [:, None] syntax
        - Optional caching for repeated sequence lengths
        - Reduce intermediate array allocations
        - Maintain exact numerical compatibility

        Args:
            x: Input tensor of shape (batch_size, sequence_length, ...)

        Returns:
            Positional encoding tensor of shape (sequence_length, dim)
        """
        # Extract sequence length from input
        seq_len = x.shape[1]

        # Check cache if enabled
        if self.use_cache and seq_len in self._embedding_cache:
            base_embeddings = self._embedding_cache[seq_len]
        else:
            # Create position indices efficiently
            positions = mx.arange(seq_len, dtype=mx.float32)

            # Compute sinusoidal values using efficient broadcasting
            # Direct broadcasting with [:, None] syntax
            sinusoids = positions[:, None] * self.inv_freq

            # Compute sin and cos and concatenate in one operation
            # This reduces memory allocations compared to separate operations
            base_embeddings = mx.concatenate(
                [mx.sin(sinusoids), mx.cos(sinusoids)], axis=-1
            )

            # Cache if enabled and under limit
            if self.use_cache and len(self._embedding_cache) < self.max_cache_size:
                self._embedding_cache[seq_len] = base_embeddings

        # Apply learnable scaling
        return base_embeddings * self.scale

    def clear_cache(self):
        """Clear the embedding cache to free memory."""
        self._embedding_cache.clear()
