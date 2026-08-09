# Copyright © 2023-2024 Apple Inc.
# Vendored from mlx-lm v0.31.3 (ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd),
# mlx_lm/models/activations.py. Modified: kept only swiglu; dropped xielu/XieLU
# (used solely by the apertus backbone, which mlx-audio does not vendor).
# MIT licensed.

from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    return nn.silu(gate) * x
