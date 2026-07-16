"""Pure-MLX VAE semantic encoder for dots.tts (``patch_encoder.*``).

Mirrors ``dots_tts.modules.backbone.semantic_encoder.VAESemanticEncoder``: takes a
generated VAE latent stream ``[B, T, 128]`` and produces a compact ``[B, T/4, 1536]``
embedding fed back into the LLM. The total time downsample of ``patch_size = 4`` is
split into a causal stride-2 conv (``ds_proj``) and a 2-step group-reshape in the
output projection (``out_ds_rate = patch_size // in_ds_rate = 2``).

Imports ONLY mlx (+ shared ``layers.py`` primitives) — never torch.

CONFIG-VS-CODE NOTE (authoritative): the checkpoint's ``PatchEncoder`` config block
sets ``qk_norm=True`` / ``rotary_bias=True``, but the upstream ``SuperviseEncoder``
(``semantic_encoder.py:124-133``) forwards ONLY ``num_layers / num_heads /
hidden_size / ffn_hidden_size / norm_layer`` into ``TransformerEncoderLayer``, which
in turn passes only ``attn_drop / norm_layer`` (+ empty ``**kwargs``) into
``MultiHeadAttention``. So the encoder's attention uses the MHA defaults:
**NO qk_norm, NO rotary, NO qkv_bias** — positions come ONLY from the causal mask.
Verified against the checkpoint key manifest (zero ``patch_encoder.*.q_norm`` /
``rotary`` keys); the loader re-asserts this. The FFN activation is **SiLU** (not GELU).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from .layers import Conv1d, Linear, Mlp, MultiHeadAttention, RMSNorm


@dataclass
class PatchEncoderDecodeState:
    """Incremental decode state for the streaming patch encoder.

    Mirrors upstream ``SemanticEncoderDecodeState`` minus ``positions``/``seq_len``:
    our encoder has no rotary / no qk-norm, so position enters only via the causal
    mask and each ``KVCache`` tracks its own offset.

      * ``conv_tail`` — the ``ds_proj`` causal conv's left-context: the last
        ``left_padding`` raw frames (NLC ``[1, left_padding, in_dim]``). Zeros at init,
        which equals the full conv's zero left-pad on the first patch.
      * ``layer_caches`` — one ``mlx_lm`` ``KVCache`` per encoder layer (the same cache
        class the LLM trunk uses).
    """

    conv_tail: mx.array
    layer_caches: list = field(default_factory=list)


class TransformerEncoderLayer:
    """Pre-norm transformer block (affine RMSNorm + MHA + SiLU FFN).

    Mirrors ``semantic_encoder.TransformerEncoderLayer.forward``::

        h = attn_norm(x); h = attn(q=h, mask=causal); x = x + h
        h = ffn_norm(x);  h = ffn(h);                 x = x + h

    ``attn_norm`` / ``ffn_norm`` are affine ``RMSNorm(hidden_size)`` (norm_layer=
    RMSNorm, eps = finfo(fp32).eps). ``attn`` is a plain causal ``MultiHeadAttention``
    (no qk_norm, no rotary, no qkv_bias). ``ffn`` is a SiLU ``Mlp``.
    """

    def __init__(
        self,
        attn_norm: RMSNorm,
        attn: MultiHeadAttention,
        ffn_norm: RMSNorm,
        ffn: Mlp,
    ):
        self.attn_norm = attn_norm
        self.attn = attn
        self.ffn_norm = ffn_norm
        self.ffn = ffn

    def __call__(self, x: mx.array, mask: mx.array, *, hp: bool = False) -> mx.array:
        h = self.attn_norm(x)
        h = self.attn(h, mask=mask, hp=hp)
        x = x + h
        h = self.ffn_norm(x)
        h = self.ffn(h, hp=hp)
        return x + h

    def decode_step(
        self, x_new: mx.array, kv_cache, mask: mx.array, *, hp: bool = False
    ):
        """Streaming block: attn.step (KV-cached) + FFN, same residual structure as __call__."""
        h = self.attn_norm(x_new)
        h = self.attn.step(h, kv_cache, mask, hp=hp)
        x_new = x_new + h
        h = self.ffn_norm(x_new)
        h = self.ffn(h, hp=hp)
        return x_new + h


class SuperviseEncoder:
    """Stack of ``TransformerEncoderLayer`` run under a shared causal mask.

    Mirrors ``semantic_encoder.SuperviseEncoder.forward`` for the non-streaming
    (recompute-full) path: builds the ``[T, T]`` lower-triangular bool mask once
    (the encoder is causal, ``config.causal=True``) and applies every layer with it.
    """

    def __init__(self, layers: list[TransformerEncoderLayer], *, causal: bool = True):
        self.layers = layers
        self.causal = causal

    def __call__(self, x: mx.array, *, hp: bool = False) -> mx.array:
        t = x.shape[1]
        if self.causal:
            # lower-triangular bool mask, True = attend (matches torch.tril(ones(T,T))).
            ones = mx.ones((t, t), dtype=mx.bool_)
            mask = mx.tril(ones)[None]  # [1, T, T] -> broadcast over batch + heads
        else:
            mask = None
        for layer in self.layers:
            x = layer(x, mask, hp=hp)
        return x

    def decode_step(
        self, x_new: mx.array, layer_caches: list, *, hp: bool = False
    ) -> mx.array:
        """Thread a new token block through every layer with a shared block mask.

        ``x_new`` is [1, n, hidden]; ``layer_caches`` is one KVCache per layer (all at the
        same offset). The mask [n, t_past + n] is True over the t_past past columns (attend
        to all history) and lower-triangular over the n new columns (causal within the
        block). Works for prefill (t_past=0, n=T -> full causal) and decode (t_past=T, n=2).
        """
        if not self.causal:
            raise ValueError("streaming decode_step requires a causal encoder.")
        n = x_new.shape[1]
        t_past = layer_caches[0].offset
        mask = mx.concatenate(
            [
                mx.ones((n, t_past), dtype=mx.bool_),
                mx.tril(mx.ones((n, n), dtype=mx.bool_)),
            ],
            axis=1,
        )
        for layer, cache in zip(self.layers, layer_caches):
            x_new = layer.decode_step(x_new, cache, mask, hp=hp)
        return x_new


class VAESemanticEncoder:
    """VAE latent ``[B, T, 128]`` -> LLM embedding ``[B, T/4, 1536]``.

    Pipeline (``forward``):
      1. ``_downsample``  causal stride-2 ``ds_proj`` conv (time T -> T/2).
      2. ``in_proj``      ``Linear(128 -> 1024)``.
      3. ``encoder``      24-layer causal ``SuperviseEncoder``.
      4. ``_project_embeddings``  group 2 consecutive timesteps into the feature dim
         (``rearrange "b (s d) h -> b s (d h)"``, d=out_ds_rate=2) then ``out_proj``
         ``Linear(2048 -> 1536)`` (time T/2 -> T/4).
    """

    def __init__(
        self,
        ds_proj: Conv1d,
        in_proj: Linear,
        encoder: SuperviseEncoder,
        out_proj: Linear,
        *,
        out_ds_rate: int = 2,
        patch_size: int = 4,
    ):
        self.ds_proj = ds_proj
        self.in_proj = in_proj
        self.encoder = encoder
        self.out_proj = out_proj
        self.out_ds_rate = out_ds_rate
        self.patch_size = patch_size

    def init_decode_state(
        self, *, dtype: mx.Dtype = mx.float32
    ) -> PatchEncoderDecodeState:
        """Fresh streaming-decode state: zero conv_tail + one empty KVCache per layer."""
        in_dim = self.ds_proj.weight.shape[-1]
        conv_tail = mx.zeros((1, self.ds_proj.left_padding, in_dim), dtype=dtype)
        layer_caches = [KVCache() for _ in self.encoder.layers]
        return PatchEncoderDecodeState(conv_tail=conv_tail, layer_caches=layer_caches)

    def _downsample(self, x: mx.array) -> mx.array:
        # Upstream applies ds_proj on [B, C, T] (transpose) then transposes back.
        # Our Conv1d is channels-last (NLC), so x[B, T, C] feeds it directly.
        # (The conv's precision is fixed at construction via its own ``hp`` flag.)
        return self.ds_proj(x)

    def _downsample_step(self, patch: mx.array, conv_tail: mx.array):
        """Streaming causal stride-2 conv over one patch (NLC), conv-only (no in_proj).

        Mirrors upstream ``_downsample_step``: prepend the carried left-context, run the
        conv with padding=0 (the tail supplies the left context), and carry the new tail.
        Returns ``(down [1, out_ds_rate, in_dim], new_conv_tail [1, left_padding, in_dim])``.
        Numerically identical to ``_downsample`` over the concatenated stream because the
        conv is causal and the tail reproduces its exact left-context.
        """
        conv_input = mx.concatenate([conv_tail, patch], axis=1)  # [1, lp+P, in_dim]
        y = mx.conv1d(
            conv_input,
            self.ds_proj.weight,
            stride=self.ds_proj.stride,
            padding=0,
            dilation=self.ds_proj.dilation,
            groups=self.ds_proj.groups,
        )
        if self.ds_proj.bias is not None:
            y = y + self.ds_proj.bias
        new_conv_tail = patch[:, -self.ds_proj.left_padding :, :]
        return y, new_conv_tail

    def _project_embeddings(self, z: mx.array, *, hp: bool) -> mx.array:
        if self.out_ds_rate > 1:
            b, sd, h = z.shape
            d = self.out_ds_rate
            s = sd // d
            # rearrange "b (s d) h -> b s (d h)": split the time axis into (s, d),
            # then merge the d sub-steps into the feature axis (d varies fastest).
            z = z.reshape(b, s, d, h).reshape(b, s, d * h)
        return self.out_proj(z, hp=hp)

    def prefill(
        self, x: mx.array, state: PatchEncoderDecodeState, *, hp: bool = False
    ) -> None:
        """Populate the decode state from the prompt's denormalized latents.

        Runs the full causal downsample + in_proj over the prompt, threads all resulting
        tokens through the encoder into the per-layer KV caches (offset 0 -> S*out_ds_rate),
        and sets conv_tail = the prompt's last left_padding raw frames. ``x`` is
        [1, S*patch_size, in_dim] (the same denormalized stream that seeds the recompute-full
        history). Mutates ``state``; no return value (the prompt embeddings are computed for
        the LLM scatter separately in model._prefill).
        """
        t = x.shape[1]
        if t == 0:
            return
        if t % self.patch_size != 0:
            raise ValueError(
                f"prefill expects T divisible by patch_size={self.patch_size}, got T={t}."
            )
        down = self._downsample(x)  # [1, S*out_ds_rate, in_dim]
        step_inputs = self.in_proj(down, hp=hp)  # [1, S*out_ds_rate, hidden]
        self.encoder.decode_step(step_inputs, state.layer_caches, hp=hp)
        state.conv_tail = x[:, -self.ds_proj.left_padding :, :]

    def decode_patch(
        self, patch: mx.array, state: PatchEncoderDecodeState, *, hp: bool = False
    ) -> mx.array:
        """Incrementally encode ONE generated patch -> one LLM token [1, 1, out_dim].

        Streaming equivalent of recompute-full: conv step (with carried tail) + in_proj +
        per-layer KV-cached decode_step + out_ds_rate grouping + out_proj. Mutates
        ``state.conv_tail`` and the layer caches.
        """
        if patch.shape[1] != self.patch_size:
            raise ValueError(
                f"decode_patch expects patch length {self.patch_size}, got {patch.shape[1]}."
            )
        down, state.conv_tail = self._downsample_step(patch, state.conv_tail)
        step_inputs = self.in_proj(down, hp=hp)  # [1, out_ds_rate, hidden]
        enc = self.encoder.decode_step(step_inputs, state.layer_caches, hp=hp)
        return self._project_embeddings(enc, hp=hp)  # [1, 1, out_dim]

    def __call__(self, x: mx.array, *, hp: bool = False) -> mx.array:
        # The recompute-full / patch decode loops always feed a time dimension that is
        # a multiple of patch_size (one or more whole VAE latent patches); make that
        # precondition explicit so a misaligned feed fails loudly instead of silently
        # mis-grouping the out_ds_rate reshape.
        t = x.shape[1]
        if t % self.patch_size != 0:
            raise ValueError(
                f"VAESemanticEncoder expects T divisible by patch_size={self.patch_size}, "
                f"got T={t}."
            )
        x = self._downsample(x)
        x = self.in_proj(x, hp=hp)
        z = self.encoder(x, hp=hp)
        return self._project_embeddings(z, hp=hp)
