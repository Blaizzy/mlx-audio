from typing import List, Optional, Tuple, Union

import mlx.core as mx


def interpolate(
    input: mx.array,
    size: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
    scale_factor: Optional[Union[float, List[float], Tuple[float, ...]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> mx.array:
    """Interpolate array with correct shape handling.

    Args:
        input (mx.array): Input tensor [N, C, ...] where ... represents spatial dimensions
        size (int or tuple): Output size
        scale_factor (float or tuple): Multiplier for spatial size
        mode (str): 'nearest' or 'linear'
        align_corners (bool): If True, align corners of input and output tensors
    """
    ndim = input.ndim
    if ndim < 3:
        raise ValueError(f"Expected at least 3D input (N, C, D1), got {ndim}D")

    spatial_dims = ndim - 2

    # Handle size and scale_factor
    if size is not None and scale_factor is not None:
        raise ValueError("Only one of size or scale_factor should be defined")
    elif size is None and scale_factor is None:
        raise ValueError("One of size or scale_factor must be defined")

    # Convert single values to tuples
    if size is not None and not isinstance(size, (list, tuple)):
        size = [size] * spatial_dims
    if scale_factor is not None and not isinstance(scale_factor, (list, tuple)):
        scale_factor = [scale_factor] * spatial_dims

    # Calculate output size from scale factor if needed
    if size is None:
        size = []
        for i in range(spatial_dims):
            # Use ceiling instead of floor to match PyTorch behavior
            curr_size = max(1, int(mx.ceil(input.shape[i + 2] * scale_factor[i])))
            size.append(curr_size)

    # Handle 1D case (N, C, W)
    if spatial_dims == 1:
        return interpolate1d(input, size[0], mode, align_corners)
    else:
        raise ValueError(
            f"Only 1D interpolation currently supported, got {spatial_dims}D"
        )


def interpolate1d(
    input: mx.array,
    size: int,
    mode: str = "linear",
    align_corners: Optional[bool] = None,
) -> mx.array:
    """1D interpolation implementation."""
    batch_size, channels, in_width = input.shape

    # Handle edge cases
    if size < 1:
        size = 1
    if in_width < 1:
        in_width = 1

    if mode == "nearest":
        if size == 1:
            indices = mx.array([0])
        else:
            scale = in_width / size
            indices = mx.floor(mx.arange(size) * scale).astype(mx.int32)
            indices = mx.clip(indices, 0, in_width - 1)
        return input[:, :, indices]

    # Linear interpolation
    if align_corners and size > 1:
        x = mx.arange(size) * ((in_width - 1) / (size - 1))
    else:
        if size == 1:
            x = mx.array([0.0])
        else:
            x = mx.arange(size) * (in_width / size)
            if not align_corners:
                x = x + 0.5 * (in_width / size) - 0.5

    # Handle the case where input width is 1
    if in_width == 1:
        output = mx.broadcast_to(input, (batch_size, channels, size))
        return output

    x_low = mx.floor(x).astype(mx.int32)
    x_high = mx.minimum(x_low + 1, in_width - 1)
    x_frac = x - x_low

    # Pre-compute indices to avoid repeated computation
    y_low = input[:, :, x_low]
    y_high = input[:, :, x_high]

    # Vectorized interpolation
    output = y_low * (1 - x_frac)[None, None, :] + y_high * x_frac[None, None, :]

    return output
