"""RNN-T prediction + joint networks for Nemotron 3.5 ASR.

Identical in structure to Parakeet's RNN-T (2-layer LSTM prednet + joint with
separate enc/pred projections), so we reuse those implementations directly.
Weight keys match: decoder.prediction.{embed,dec_rnn.lstm.N.*}, joint.{enc,pred,joint_net.2}.
"""

from mlx_audio.stt.models.parakeet.rnnt import (  # noqa: F401
    JointArgs,
    JointNetwork,
    JointNetworkArgs,
    PredictArgs,
    PredictNetwork,
    PredictNetworkArgs,
)

__all__ = [
    "PredictNetwork",
    "JointNetwork",
    "PredictArgs",
    "JointArgs",
    "PredictNetworkArgs",
    "JointNetworkArgs",
]
