import mlx.core as mx


@mx.compile
def create_window(win_type: str, win_len: int, periodic: bool = False) -> mx.array:
    """Create window function"""
    if win_type == "hamming":
        n = mx.arange(win_len, dtype=mx.float32)
        if periodic:
            return 0.54 - 0.46 * mx.cos(2 * mx.pi * n / win_len)
        else:
            return 0.54 - 0.46 * mx.cos(2 * mx.pi * n / (win_len - 1))
    elif win_type == "hann":
        n = mx.arange(win_len, dtype=mx.float32)
        if periodic:
            return 0.5 * (1 - mx.cos(2 * mx.pi * n / win_len))
        else:
            return 0.5 * (1 - mx.cos(2 * mx.pi * n / (win_len - 1)))
    else:
        raise ValueError(f"Unsupported window type: {win_type}")


@mx.compile
def stft(
    x: mx.array,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: mx.array,
    center: bool = True,
):
    """
    STFT with maximum optimizations:
    - Minimal memory allocations
    - Vectorized operations
    - Optimized for repeated calls
    """
    batch_size, signal_len = x.shape

    # 1. Efficient padding using slice operations
    if center:
        pad_amount = n_fft // 2
        # Direct slice-based reflection (fastest approach)
        x_padded = mx.concatenate(
            [
                x[:, 1 : pad_amount + 1][:, ::-1],  # Left reflection
                x,  # Original signal
                x[:, -pad_amount - 1 : -1][:, ::-1],  # Right reflection
            ],
            axis=-1,
        )
        padded_len = signal_len + 2 * pad_amount
    else:
        x_padded = x
        padded_len = signal_len

    # 2. Single-shot framing and windowing
    num_frames = (padded_len - win_length) // hop_length + 1

    # Create frames with strides
    frames = mx.as_strided(
        x_padded,
        shape=(batch_size, num_frames, win_length),
        strides=(padded_len, hop_length, 1),
    )

    # 3. Apply window and handle FFT size in one operation
    if win_length == n_fft:
        # Perfect case - no padding needed
        windowed_frames = frames * window
    else:
        # Apply original window to frames
        windowed_frames = frames * window[:win_length]
        # Pad frames to n_fft
        windowed_frames = mx.concatenate(
            [windowed_frames, mx.zeros((batch_size, num_frames, n_fft - win_length))],
            axis=-1,
        )

    # 4. FFT with immediate transpose for memory efficiency
    stft_complex = mx.fft.rfft(windowed_frames, n=n_fft, axis=-1).transpose(0, 2, 1)

    return mx.real(stft_complex), mx.imag(stft_complex)


class STFTCache:
    """
    Advanced caching for STFT operations. Handles multiple signal lengths efficiently.
    """

    def __init__(self):
        self.padding_cache = {}

    def get_padding_indices(self, signal_len: int, n_fft: int, center: bool = True):
        """Get cached padding indices or create new ones"""
        key = (signal_len, n_fft, center)
        if key not in self.padding_cache:
            if not center:
                self.padding_cache[key] = (None, None, signal_len)
            else:
                pad_amount = n_fft // 2
                left_indices = mx.arange(1, pad_amount + 1)
                right_indices = mx.arange(signal_len - pad_amount - 1, signal_len - 1)
                padded_len = signal_len + 2 * pad_amount
                self.padding_cache[key] = (left_indices, right_indices, padded_len)
        return self.padding_cache[key]

    def stft(
        self,
        x: mx.array,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: mx.array,
        center: bool = True,
    ):
        """STFT with caching (currently just calls main stft)"""
        return stft(x, n_fft, hop_length, win_length, window, center)


def create_istft_norm_buffer(
    n_fft: int, hop_length: int, win_length: int, window: mx.array, num_frames: int
) -> mx.array:
    """
    Pre-compute normalization buffer for repeated iSTFT calls.
    This can be cached and reused for significant performance gains.
    """
    frame_length = window.shape[0]
    ola_len = (num_frames - 1) * hop_length + frame_length

    window_squared = window**2
    norm_buffer = mx.zeros(ola_len, dtype=mx.float32)

    # Vectorized creation
    positions = (
        mx.arange(num_frames)[:, None] * hop_length + mx.arange(frame_length)[None, :]
    )
    positions_flat = positions.reshape(-1)
    window_sq_tiled = mx.tile(window_squared, num_frames)
    norm_buffer = norm_buffer.at[positions_flat].add(window_sq_tiled)

    return mx.maximum(norm_buffer, 1e-10)


@mx.compile
def istft(
    real_part: mx.array,
    imag_part: mx.array,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: mx.array,
    center: bool = True,
    audio_length: int = None,
) -> mx.array:
    """
    Pure MLX iSTFT with multiple performance improvements:
    1. Pre-computed normalization buffer (can be cached)
    2. Reduced memory allocations
    3. Vectorized operations where possible
    """
    # Step 1: Get windowed time-domain frames
    stft_complex = real_part + 1j * imag_part
    time_frames = mx.fft.irfft(stft_complex.transpose(0, 2, 1), n=n_fft, axis=-1)
    windowed_frames = time_frames * window

    batch_size, num_frames, frame_length = windowed_frames.shape
    ola_len = (num_frames - 1) * hop_length + frame_length

    # Step 2: Pre-compute normalization buffer
    window_squared = window**2
    norm_buffer = mx.zeros(ola_len, dtype=mx.float32)

    # Vectorized normalization buffer creation
    positions = (
        mx.arange(num_frames)[:, None] * hop_length + mx.arange(frame_length)[None, :]
    )
    positions_flat = positions.reshape(-1)
    window_sq_tiled = mx.tile(window_squared, num_frames)
    norm_buffer = norm_buffer.at[positions_flat].add(window_sq_tiled)

    # Step 3: Overlap-add using advanced indexing
    output = mx.zeros((batch_size, ola_len), dtype=mx.float32)

    # Reshape for vectorized scatter
    windowed_flat = windowed_frames.reshape(batch_size, -1)

    # Use scatter_add for all batches at once
    for b in range(batch_size):
        output = output.at[b, positions_flat].add(windowed_flat[b])

    # Step 4: Normalize (avoid division by zero)
    norm_buffer = mx.maximum(norm_buffer, 1e-10)
    output = output / norm_buffer[None, :]

    # Step 5: Final trimming
    if center:
        start_cut = n_fft // 2
        output = output[:, start_cut:]

    if audio_length is not None:
        output = output[:, :audio_length]

    return output


class ISTFTCache:
    """
    Advanced caching for iSTFT operations. Fully vectorized Overlap-Add for MLX. Handles multiple configurations efficiently.
    Automatically caches normalization buffers and position indices for maximum performance.
    """

    def __init__(self):
        self.norm_buffer_cache = {}
        self.position_cache = {}

    def get_positions(self, num_frames: int, frame_length: int, hop_length: int):
        """Get cached position indices or create new ones"""
        key = (num_frames, frame_length, hop_length)

        if key not in self.position_cache:
            # Create base indices for one frame sequence
            positions = (
                mx.arange(num_frames)[:, None] * hop_length
                + mx.arange(frame_length)[None, :]
            )
            self.position_cache[key] = positions.reshape(-1)

        return self.position_cache[key]

    def get_norm_buffer(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: mx.array,
        num_frames: int,
    ):
        """Get cached normalization buffer or create new one"""
        # Create a unique key for the window configuration
        # Note: .tolist() is slow, but this only happens once per new configuration
        window_hash = hash(tuple(window.tolist()))
        key = (n_fft, hop_length, win_length, window_hash, num_frames)

        if key not in self.norm_buffer_cache:
            # Logic inlined to ensure self-containment
            frame_length = window.shape[0]
            ola_len = (num_frames - 1) * hop_length + frame_length

            # Get positions (reuse cache logic)
            positions_flat = self.get_positions(num_frames, frame_length, hop_length)

            # Create buffer
            window_squared = window**2
            norm_buffer = mx.zeros(ola_len, dtype=mx.float32)
            window_sq_tiled = mx.tile(window_squared, num_frames)

            # Vectorized scatter for norm buffer
            norm_buffer = norm_buffer.at[positions_flat].add(window_sq_tiled)
            norm_buffer = mx.maximum(norm_buffer, 1e-10)

            self.norm_buffer_cache[key] = norm_buffer

        return self.norm_buffer_cache[key]

    def istft(
        self,
        real_part: mx.array,
        imag_part: mx.array,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: mx.array,
        center: bool = True,
        audio_length: int = None,
    ) -> mx.array:
        """
        iSTFT with automatic caching and vectorized overlap-add.
        """
        # Robust Window Padding (Safety Check)
        if window.shape[0] < n_fft:
            pad = n_fft - window.shape[0]
            window = mx.concatenate([window, mx.zeros((pad,), dtype=window.dtype)])

        # Inverse FFT
        stft_complex = real_part + 1j * imag_part
        time_frames = mx.fft.irfft(stft_complex.transpose(0, 2, 1), n=n_fft, axis=-1)

        # Apply synthesis window
        windowed_frames = time_frames * window

        batch_size, num_frames, frame_length = windowed_frames.shape
        ola_len = (num_frames - 1) * hop_length + frame_length

        # Get Cached Items
        # Note: We pass the *padded* window to ensure the normalization buffer matches
        norm_buffer = self.get_norm_buffer(
            n_fft, hop_length, win_length, window, num_frames
        )
        positions_flat = self.get_positions(num_frames, frame_length, hop_length)

        # Vectorized Overlap-Add (The Speedup)
        # Instead of looping "for b in range(batch_size)", we scatter all at once.

        # Create global indices for the entire batch
        batch_offsets = mx.arange(batch_size) * ola_len

        # Broadcast positions: (1, P) + (B, 1) -> (B, P)
        global_indices = positions_flat[None, :] + batch_offsets[:, None]

        # Flatten everything to 1D for a single GPU operation
        output = mx.zeros((batch_size * ola_len), dtype=mx.float32)
        output = output.at[global_indices.reshape(-1)].add(windowed_frames.reshape(-1))
        output = output.reshape(batch_size, ola_len)

        # Apply Normalization
        output = output / norm_buffer[None, :]

        # Final Trimming
        if center:
            start_cut = n_fft // 2
            output = output[:, start_cut:]

        if audio_length is not None:
            output = output[:, :audio_length]

        return output

    def clear_cache(self):
        """Clear all cached data to free memory"""
        self.norm_buffer_cache.clear()
        self.position_cache.clear()

    def cache_info(self):
        """Get information about cached items"""
        return {
            "norm_buffers": len(self.norm_buffer_cache),
            "position_indices": len(self.position_cache),
            "total_cached_items": len(self.norm_buffer_cache)
            + len(self.position_cache),
        }
