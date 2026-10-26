"""Common output contract for file-based STS generation."""

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np


@dataclass
class STSOutput:
    """Audio, text, or named audio stems returned by ``model.generate()``.

    Audio arrays use [samples] or [samples, channels] layout. ``audios`` holds
    separate named outputs, such as speaker stems, rather than audio channels.
    Conversational models may return richer results with the same attributes.
    """

    audio: mx.array | np.ndarray | None = None
    sample_rate: int = 24000
    text: str | None = None
    audios: dict[str, mx.array | np.ndarray] = field(default_factory=dict)
