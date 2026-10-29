"""Speech checkpoint precision contracts on top of the shared Nemotron-H model."""

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.lm.models.nemotron_h import Model as NemotronH
from mlx_audio.lm.models.nemotron_h import NemotronHMamba2Mixer
from mlx_audio.lm.models.ssm import ssm_update


class MambaMixer(NemotronHMamba2Mixer):
    def _ssm(self, hidden_states, B, C, dt, cache, mask):
        b, t, _ = hidden_states.shape
        # NeMo retains A_log, D, dt_bias and the recurrent state in FP32.
        # Accumulate both the scan and skip connection before rounding to BF16.
        y, state = ssm_update(
            hidden_states.reshape(b, t, self.num_heads, self.head_dim).astype(
                mx.float32
            ),
            self.A_log,
            B.reshape(b, t, self.n_groups, self.ssm_state_size).astype(mx.float32),
            C.reshape(b, t, self.n_groups, self.ssm_state_size).astype(mx.float32),
            self.D,
            dt,
            self.dt_bias,
            cache[1] if cache is not None else None,
            self.time_step_limit,
            mask,
        )
        if cache is not None:
            cache[1] = state
        return y.reshape(b, t, self.intermediate_size).astype(hidden_states.dtype)


class GatedRMSNorm(nn.Module):
    """Equivalent to NeMo's fused gate-before-normalization kernel."""

    def __init__(self, width, group_size, eps):
        super().__init__()
        self.weight = mx.ones(width)
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x, gate):
        dtype = x.dtype
        x = x.astype(mx.float32) * nn.silu(gate.astype(mx.float32))
        x = mx.unflatten(x, -1, (-1, self.group_size))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps).flatten(-2)
        return (x * self.weight).astype(dtype)


class LanguageModel(NemotronH):
    def __init__(self, args):
        super().__init__(args)
        for layer in self.layers:
            if layer.block_type == "M":
                mixer = MambaMixer(args)
                mixer.norm = GatedRMSNorm(
                    mixer.intermediate_size,
                    mixer.intermediate_size // args.n_groups,
                    args.layer_norm_epsilon,
                )
                layer.mixer = mixer
