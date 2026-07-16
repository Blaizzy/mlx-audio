"""Pure-MLX AudioVAE encode + decode paths (BigVGAN-style alias-free causal vocoder).

Imports ONLY mlx — never torch, never numpy. Ports two halves of
``dots_tts.modules.vocoder.bigvgan.AudioVAE``:

  decode (``inference_from_latents``, do_sample=False):
    post_proj (Conv1d k=1) -> [B,C,T]->[B,T,C]
      -> dec_mi_layer (Linear, SLSTM x4, Linear)
      -> [B,T,C]->[B,C,T] -> Decoder -> clamp(-1, 1)  (48 kHz waveform)

  encode (``extract_latents``, do_sample=False):
    audio_encoder (Encoder: Conv1d_S stack + ResStacks, /1920) -> [B,C,T]->[B,T,C]
      -> enc_mi_layer (Linear, SLSTM x4, Linear) -> [B,T,C]->[B,C,T]
      -> pre_proj (Conv1d k=1, -> 2*latent_dim)  (the mean/log_std pair [B,256,T])

All tensors flow internally as NLC ``[B, T, C]`` (MLX-native). The SnakeBeta
*activation* math is forced fp32 (see ``_snakebeta``); everything else runs at the
runtime ``dtype`` EXCEPT the tiny alias-free filter taps, which ``loader.py`` keeps
fp32 unconditionally (cheap + parity-stable). Alias-free low-pass filters are LOADED
from the checkpoint (no Kaiser recompute). Weights are bound by ``loader.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from .config import VocoderConfig
from .layers import Conv1d, ConvTranspose1d, hp_matmul, replicate_pad


def _snakebeta(x: mx.array, alpha: mx.array, beta: mx.array) -> mx.array:
    """SnakeBeta (logscale) over an NLC tensor: ``x + 1/(exp(beta)+eps) * sin(x*exp(alpha))^2``.

    ``alpha`` / ``beta`` are ``[C]``; broadcast over (B, T). Computed in fp32.
    """
    x32 = x.astype(mx.float32)
    a = mx.exp(alpha.astype(mx.float32))
    b = mx.exp(beta.astype(mx.float32))
    s = mx.sin(x32 * a)
    return x32 + (1.0 / (b + 1e-9)) * (s * s)


@dataclass
class _AliasFreeResampler:
    """Causal alias-free up/down sampler around a SnakeBeta activation.

    Filters are depthwise MLX-layout kernels ``[C, k, 1]`` loaded from the
    checkpoint. ``up_filter`` drives ``UpSample1d`` (conv_transpose, scale x ratio,
    causal trim of ``k_up - ratio``); ``down_filter`` drives the ``LowPassFilter1d``
    (causal left-pad ``k_down - 1``, replicate pad, depthwise conv stride=ratio).
    """

    up_filter: mx.array
    down_filter: mx.array
    alpha: mx.array
    beta: mx.array
    ratio: int = 2

    @property
    def _k_up(self) -> int:
        return self.up_filter.shape[1]

    @property
    def _k_down(self) -> int:
        return self.down_filter.shape[1]

    def __call__(self, x: mx.array) -> mx.array:
        c = x.shape[2]
        x32 = x.astype(mx.float32)
        # The depthwise up/down filters + the replicate (edge-clamp) pad bypass the
        # layers.py Conv1d classes on purpose: those handle channel-mixing zero-pad
        # convs, whereas here we need grouped (depthwise, groups=c) convs with a
        # replicate left-pad, so we call mx.conv* directly. Filter taps are fp32
        # (kept so by loader.py), matching the fp32 x32 here.
        # UpSample1d (causal pad=0): conv_transpose, scale x ratio, trim last (k-stride).
        up = self.ratio * mx.conv_transpose1d(
            x32, self.up_filter, stride=self.ratio, padding=0, groups=c
        )
        up = up[:, : -(self._k_up - self.ratio), :]
        # SnakeBeta activation at the upsampled rate.
        act = _snakebeta(up, self.alpha, self.beta)
        # DownSample1d -> LowPassFilter1d (causal): replicate-pad k-1, conv stride=ratio.
        padded = replicate_pad(act, self._k_down - 1, 0)
        down = mx.conv1d(
            padded, self.down_filter, stride=self.ratio, padding=0, groups=c
        )
        return down


@dataclass
class _AMPBlock1:
    """Anti-aliased multi-periodicity composition block (kernel_size, dilation=(1,3,5)).

    ``x = x + convs2[j](acts2[j](convs1[j](acts1[j](x))))`` for j in 0..2, where
    acts1 = activations[::2], acts2 = activations[1::2] (6 resamplers total).
    """

    convs1: list[Conv1d]
    convs2: list[Conv1d]
    acts: list[_AliasFreeResampler]  # 6 total

    def __call__(self, x: mx.array) -> mx.array:
        acts1 = self.acts[0::2]
        acts2 = self.acts[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


@dataclass
class _SLSTM:
    """Unidirectional residual LSTM (skip=True) over NLC ``[B, T, C]``.

    PyTorch gate order is i, f, g, o. Uses both ``bias_ih`` + ``bias_hh``. Weights
    per layer are stored in torch layout: ``weight_ih`` ``[4H, in]``, ``weight_hh``
    ``[4H, H]`` (here in == H). ``y = lstm(x) + x``.
    """

    weight_ih: list[mx.array]  # per layer, [4H, H]
    weight_hh: list[mx.array]  # per layer, [4H, H]
    bias_ih: list[mx.array]  # per layer, [4H]
    bias_hh: list[mx.array]  # per layer, [4H]
    num_layers: int
    hp: bool = False  # true fp32 matmul reduction (encode path; max-abs gated)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        b, t, h = x.shape
        mm = hp_matmul if self.hp else (lambda a, w_t: a @ w_t)
        for layer in range(self.num_layers):
            w_ih = self.weight_ih[layer]
            w_hh = self.weight_hh[layer]
            bias = self.bias_ih[layer] + self.bias_hh[layer]  # [4H]
            # Precompute the input projection for every timestep: [B, T, 4H].
            gates_in = mm(x, w_ih.T) + bias
            hx = mx.zeros((b, h), dtype=x.dtype)
            cx = mx.zeros((b, h), dtype=x.dtype)
            outputs = []
            for ts in range(t):
                gates = gates_in[:, ts, :] + mm(hx, w_hh.T)
                i_g, f_g, g_g, o_g = mx.split(gates, 4, axis=-1)
                i_g = mx.sigmoid(i_g)
                f_g = mx.sigmoid(f_g)
                g_g = mx.tanh(g_g)
                o_g = mx.sigmoid(o_g)
                cx = f_g * cx + i_g * g_g
                hx = o_g * mx.tanh(cx)
                outputs.append(hx)
            x = mx.stack(outputs, axis=1)
        return x + residual


def _leaky_relu(x: mx.array, slope: float) -> mx.array:
    """LeakyReLU: ``max(x, 0) + slope * min(x, 0)`` (mirrors ``nn.LeakyReLU``)."""
    return mx.where(x >= 0, x, x * slope)


@dataclass
class _ResStack:
    """Causal residual conv stack (upstream ``ResStack``, ``nums`` blocks).

    Each block: ``x = x + conv2(LeakyReLU(conv1(LeakyReLU(x))))`` where conv1 has
    dilation ``base**i`` (k=3, causal left-pad ``dil*2``) and conv2 has dilation 1
    (k=3, causal left-pad 2). The two LeakyReLUs use the default slope 0.01.
    """

    convs1: list[Conv1d]  # dilated, per block
    convs2: list[Conv1d]  # dilation=1, per block

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2 in zip(self.convs1, self.convs2):
            h = _leaky_relu(x, 0.01)
            h = c1(h)
            h = _leaky_relu(h, 0.01)
            h = c2(h)
            x = x + h
        return x


@dataclass
class _Encoder:
    """AudioVAE waveform encoder (upstream ``Encoder.generator`` Sequential).

    Structure (NLC tensors):
      pre = Conv1d_S(1->base, k=3, s=1, causal) ; LeakyReLU(0.2)
      per stage i: Conv1d_S(in->out, k=2*down, s=down, causal) ; ResStack ; LeakyReLU(0.2)
      post (lookahead=2): Conv1d_S(last->latent_dim, k=5, s=1, NON-causal)

    The lookahead post-conv is non-causal (kernel = 2*lookahead+1 = 5), matching
    upstream where ``causal=False`` is hard-set for the lookahead branch.
    """

    pre_conv: Conv1d
    down_convs: list[Conv1d]
    res_stacks: list[_ResStack]
    post_conv: Conv1d
    act_slope: float = 0.2

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pre_conv(x)
        x = _leaky_relu(x, self.act_slope)
        for dc, rs in zip(self.down_convs, self.res_stacks):
            x = dc(x)
            x = rs(x)
            x = _leaky_relu(x, self.act_slope)
        x = self.post_conv(x)
        return x


@dataclass
class _Decoder:
    """BigVGAN decoder: conv_pre -> [upsample + AMP] x num_stages -> act_post -> conv_post."""

    conv_pre: Conv1d
    ups: list[ConvTranspose1d]
    resblocks: list[_AMPBlock1]  # len = num_stages * num_kernels
    act_post: _AliasFreeResampler
    conv_post: Conv1d
    num_kernels: int

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_pre(x)
        num_upsamples = len(self.ups)
        for i in range(num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                xs = rb(x) if xs is None else xs + rb(x)
            x = xs / self.num_kernels
        x = self.act_post(x)
        x = self.conv_post(x)
        x = mx.clip(x, -1.0, 1.0)  # use_tanh_at_final=False
        return x


@dataclass
class AudioVAE:
    """AudioVAE encode + decode paths (waveform <-> latent)."""

    config: VocoderConfig
    post_proj: Conv1d
    # dec_mi_layer = Sequential(Linear, SLSTM, Linear)
    mi_lin0_w: mx.array  # [inter, latent]
    mi_lin0_b: mx.array
    slstm: _SLSTM
    mi_lin2_w: mx.array  # [latent, inter]
    mi_lin2_b: mx.array
    decoder: _Decoder
    # Encode path (built only when load_audiovae(with_encoder=True)).
    encoder: _Encoder | None = None
    # enc_mi_layer = Sequential(Linear, SLSTM, Linear)
    enc_lin0_w: mx.array | None = None  # [inter, latent]
    enc_lin0_b: mx.array | None = None
    enc_slstm: _SLSTM | None = None
    enc_lin2_w: mx.array | None = None  # [latent, inter]
    enc_lin2_b: mx.array | None = None
    pre_proj: Conv1d | None = None
    extras: dict = field(default_factory=dict)

    def encode(self, wav: mx.array) -> mx.array:
        """Encode a waveform ``[B, 1, S]`` into a latent ``[B, 2*latent_dim, T]``.

        Returns the stacked (mean, log_std) pair, i.e. the ``do_sample=False`` output
        of ``extract_latents``. ``T == S // 1920``.
        """
        if self.encoder is None:
            raise RuntimeError(
                "encode path not loaded; call load_audiovae(..., with_encoder=True)"
            )
        x = wav.astype(mx.float32)
        # audio_encoder consumes NLC: [B, 1, S] -> [B, S, 1].
        x = x.transpose(0, 2, 1)
        x = self.encoder(x)  # [B, T, latent_dim]
        # enc_mi_layer (already NLC == batch_first layout); HP matmuls for fp32 parity.
        x = hp_matmul(x, self.enc_lin0_w.T) + self.enc_lin0_b
        x = self.enc_slstm(x)
        x = hp_matmul(x, self.enc_lin2_w.T) + self.enc_lin2_b
        # pre_proj is Conv1d k=1 (hp): a per-frame channel mix on NLC -> [B, T, 256].
        x = self.pre_proj(x)
        # Return [B, 2*latent_dim, T] (channel-first), matching the torch latent.
        return x.transpose(0, 2, 1)

    def decode(self, latent: mx.array) -> mx.array:
        """Decode a latent ``[B, latent_dim, T]`` into a waveform ``[B, 1, T*1920]``."""
        x = latent.astype(mx.float32)
        # post_proj is Conv1d k=1 over [B, C, T]; run it as a channel-mix on NLC.
        x = x.transpose(0, 2, 1)  # [B, T, C]
        x = self.post_proj(x)  # k=1 conv == per-frame linear
        # dec_mi_layer (already NLC == batch_first layout)
        x = x @ self.mi_lin0_w.T + self.mi_lin0_b
        x = self.slstm(x)
        x = x @ self.mi_lin2_w.T + self.mi_lin2_b
        # Decoder consumes NLC directly.
        out = self.decoder(x)  # [B, T*1920, 1]
        return out.transpose(0, 2, 1)  # [B, 1, T*1920]
