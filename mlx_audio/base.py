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
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) == 4:
        out_channels, kH, KW, _ = shape
        # Check if out_channels is the largest, and kH and KW are the same
        if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
            return True
        else:
            return False
    # Check if the shape has 3 dimensions
    elif len(shape) == 3:
        _, kW, out_channels = shape
        # Check if out_channels is the largest
        if kW >= out_channels:
            return True
        else:
            return False
    else:
        return False
