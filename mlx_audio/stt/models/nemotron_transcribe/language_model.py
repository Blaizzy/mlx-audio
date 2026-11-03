"""Speech checkpoint precision contracts on top of the shared Nemotron-H model."""

import mlx.core as mx
from mlx import nn

from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import ArraysCache, KVCache
from mlx_audio.lm.models.nemotron_h import Model as NemotronH
from mlx_audio.lm.models.nemotron_h import NemotronHBlock, NemotronHMamba2Mixer
from mlx_audio.lm.models.ssm import ssm_update


class MambaCache(ArraysCache):
    """Retain short verification inputs so a rejected suffix can be undone.

    Recompute only the recurrent scan of the accepted prefix, without replaying
    the backbone's projections, experts or attention. Ordinary decoding keeps
    no rollback inputs.
    """

    def __init__(self):
        super().__init__(size=2)
        self.recording = False
        self.conv_inputs = self.ssm_inputs = None

    def finish_verification(self, mixer, keep, rejected):
        self.recording = False
        if rejected:
            self[0] = self.conv_inputs[:, keep : keep + mixer.conv_kernel_size - 1]
            x, B, C, dt, state = self.ssm_inputs
            temporary = ArraysCache(size=2)
            temporary[1] = state
            mixer._ssm(
                x[:, :keep], B[:, :keep], C[:, :keep], dt[:, :keep], temporary, None
            )
            self[1] = temporary[1]
        self.conv_inputs = self.ssm_inputs = None


class MambaMixer(NemotronHMamba2Mixer):
    def _conv(self, conv_input, cache, mask):
        if isinstance(cache, MambaCache) and cache.recording:
            previous = cache[0]
            if previous is None:
                previous = mx.zeros(
                    (conv_input.shape[0], self.conv_kernel_size - 1, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            cache.conv_inputs = mx.concatenate([previous, conv_input], axis=1)
        return super()._conv(conv_input, cache, mask)

    def _ssm(self, hidden_states, B, C, dt, cache, mask):
        b, t, _ = hidden_states.shape
        if isinstance(cache, MambaCache) and cache.recording:
            cache.ssm_inputs = (hidden_states, B, C, dt, cache[1])
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
    def __init__(self, args, mtp_config=None):
        super().__init__(args)
        self.mtp = None
        if mtp_config and mtp_config.get("enabled"):
            self.mtp = MultiTokenPrediction(args)
        for layer in self.layers:
            if layer.block_type == "M":
                mixer = MambaMixer(args)
                mixer.norm = GatedRMSNorm(
                    mixer.intermediate_size,
                    mixer.intermediate_size // args.n_groups,
                    args.layer_norm_epsilon,
                )
                layer.mixer = mixer

    def make_cache(self):
        return [
            MambaCache() if layer.block_type == "M" else KVCache()
            for layer in self.layers
            if layer.block_type in ("M", "*")
        ]

    def finish_verification(self, cache, keep, rejected):
        index = 0
        for layer in self.layers:
            if layer.block_type == "M":
                cache[index].finish_verification(layer.mixer, keep, rejected)
                index += 1
            elif layer.block_type == "*":
                cache[index].trim(rejected)
                index += 1


class MultiTokenPrediction(nn.Module):
    """NVIDIA's repeated attention/MoE depth, sharing target embeddings/head.

    Fusion consumes the normalized target hidden state and the *next* input
    embedding, including the shifted audio embeddings during prefill.
    """

    def __init__(self, args):
        super().__init__()
        self.layers = [NemotronHBlock(args, kind) for kind in ("*", "E")]
        first, last = self.layers
        first.enorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        first.hnorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        first.eh_proj = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        last.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(self, embeddings, hidden, cache):
        first, last = self.layers
        hidden = first.eh_proj(
            mx.concatenate([first.enorm(embeddings), first.hnorm(hidden)], axis=-1)
        )
        hidden = first(
            hidden, mask=create_attention_mask(hidden, cache[0]), cache=cache[0]
        )
        return last.final_layernorm(last(hidden))
