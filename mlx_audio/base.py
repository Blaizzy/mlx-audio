import inspect
from dataclasses import dataclass


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def check_array_shape(arr):
    """
    Check if a conv weight array is already in MLX format.

    For 1D convolutions:
        MLX format: (out_channels, kernel_size, in_channels)
        PyTorch format: (out_channels, in_channels, kernel_size)

    For 2D convolutions:
        MLX format: (out_channels, kH, kW, in_channels)
        PyTorch format: (out_channels, in_channels, kH, kW)

    Returns True if the array appears to be in MLX format (no transpose needed).
    Returns False if the array appears to be in PyTorch format (needs transpose).

    Heuristic: kernel dimensions are typically small (1, 3, 5, 7, 9, 11),
    while channel dimensions are typically larger (64, 128, 256, 512, etc.).
    """
    shape = arr.shape

    # Common kernel sizes for convolutions
    KERNEL_SIZE_THRESHOLD = 15

    if len(shape) == 4:
        # 2D convolution: check if dims 1,2 are kernel-like (small) vs dim 3 being channel-like
        out_channels, dim1, dim2, dim3 = shape
        # MLX format: (out_channels, kH, kW, in_channels) - dim1, dim2 are small kernels
        # PyTorch format: (out_channels, in_channels, kH, kW) - dim3, dim2 are small kernels
        if (
            dim1 <= KERNEL_SIZE_THRESHOLD
            and dim2 <= KERNEL_SIZE_THRESHOLD
            and dim3 > KERNEL_SIZE_THRESHOLD
        ):
            return True  # MLX format
        elif (
            dim2 <= KERNEL_SIZE_THRESHOLD
            and dim3 <= KERNEL_SIZE_THRESHOLD
            and dim1 > KERNEL_SIZE_THRESHOLD
        ):
            return False  # PyTorch format
        # Fallback to original logic for ambiguous cases
        if (out_channels >= dim1) and (out_channels >= dim2) and (dim1 == dim2):
            return True
        return False

    elif len(shape) == 3:
        # 1D convolution: (out_channels, kernel_size, in_channels) for MLX
        #                 (out_channels, in_channels, kernel_size) for PyTorch
        out_channels, dim1, dim2 = shape
        # If middle dim is small (kernel-like) and last dim is large (channel-like): MLX format
        if dim1 <= KERNEL_SIZE_THRESHOLD and dim2 > KERNEL_SIZE_THRESHOLD:
            return True  # MLX format
        # If last dim is small (kernel-like) and middle dim is large (channel-like): PyTorch format
        elif dim2 <= KERNEL_SIZE_THRESHOLD and dim1 > KERNEL_SIZE_THRESHOLD:
            return False  # PyTorch format

        # Ambiguous case: both dims are small (both could be kernel-like)
        # Special handling when one dim is 1:
        # - in_channels=1 is common for certain operations
        # - kernel_size=1 (pointwise conv) is less common than kernel_size=3,5,7
        # So if dim1=1 and dim2>1, assume dim1 is in_channels (PyTorch format)
        if dim1 == 1 and dim2 > 1:
            return False  # Assume PyTorch format: (out, in=1, kernel)
        if dim2 == 1 and dim1 > 1:
            return True  # Assume MLX format: (out, kernel, in=1)

        # Both dims are similar and neither is 1
        # Kernel is typically smaller than or equal to in_channels
        if dim1 <= dim2:
            return True  # Assume MLX format (kernel in middle is smaller or equal)
        return False  # Assume PyTorch format

    else:
        return False
