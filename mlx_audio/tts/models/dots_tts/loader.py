"""MLX runtime loader for dots.tts.

Sets a conservative MLX memory ceiling at import as a safety guard, then provides a
``DotsTts`` container + ``from_pretrained`` that loads the config,
latent stats, tokenizer dir, and validates the three converted safetensors.

Submodule wiring (instantiating the DiT / encoder / vocoder / speaker / LLM and
binding weights) is filled in by later tasks; for now ``load_weights`` only
validates presence and surfaces the loaded raw weight dicts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

# Memory-ceiling safety guard: set BEFORE any heavy allocation.
_DOTS_MEMORY_LIMIT_GB = int(os.environ.get("MLX_AUDIO_DOTS_MEMORY_LIMIT_GB", "20"))
if _DOTS_MEMORY_LIMIT_GB > 0:
    mx.set_memory_limit(int(_DOTS_MEMORY_LIMIT_GB * (1 << 30)))

from .audiovae import (  # noqa: E402  (must follow the memory guard)
    _SLSTM,
    AudioVAE,
    _AliasFreeResampler,
    _AMPBlock1,
    _Decoder,
    _Encoder,
    _ResStack,
)
from .config import ModelConfig, VocoderConfig  # noqa: E402
from .dit import DiT, DiTBlock, FinalLayer, TimestepEmbedder  # noqa: E402
from .layers import (  # noqa: E402
    Conv1d,
    ConvTranspose1d,
    Linear,
    Mlp,
    MultiHeadAttention,
    RMSNorm,
    gelu_tanh,
    silu,
)
from .semantic_encoder import (  # noqa: E402
    SuperviseEncoder,
    TransformerEncoderLayer,
    VAESemanticEncoder,
)
from .speaker import (  # noqa: E402
    _BN,
    FCM,
    CAMPPlus,
    _BasicResBlock,
    _CAMDenseTDNNBlock,
    _CAMDenseTDNNLayer,
    _CAMLayer,
    _Conv1d,
    _Conv2d,
    _TDNNLayer,
    _TransitLayer,
)

_SAFETENSORS = ("core.safetensors", "vocoder.safetensors", "speaker.safetensors")


@dataclass
class DotsTts:
    """Container for the loaded dots.tts model + assets.

    Attributes:
        config: the resolved ``ModelConfig`` (from the checkpoint JSON).
        dtype: the runtime cast dtype (bf16 default; parity tests pass fp32).
        path: the ``weights/dots_tts_mlx`` directory the assets were loaded from.
        latent_mean / latent_var: per-channel latent normalization stats, shape (128,).
        tokenizer_dir: path to the copied tokenizer files.
        weights: raw weight dicts keyed by submodule file stem
            (``core`` / ``vocoder`` / ``speaker``), populated by ``load_weights``.
        model: the fully-wired ``DotsTtsModel`` AR runtime (populated by
            ``from_pretrained``); ``None`` until wired.
    """

    config: ModelConfig
    dtype: mx.Dtype
    path: Path
    latent_mean: mx.array
    latent_var: mx.array
    tokenizer_dir: Path
    weights: dict[str, dict[str, mx.array]] = field(default_factory=dict)
    model: object | None = None

    def load_weights(self) -> "DotsTts":
        """Validate the three safetensors load and stash the raw weight dicts.

        Submodule construction + weight binding is deferred to later tasks. This
        only confirms the converted artifacts are present and parseable, casting
        each tensor to ``self.dtype``.
        """
        for name in _SAFETENSORS:
            fpath = self.path / name
            if not fpath.exists():
                raise FileNotFoundError(
                    f"missing converted weight file: {fpath} "
                    "(run `python -m dots_tts_mlx.convert` under the oracle venv first)"
                )
            raw = mx.load(str(fpath))
            if not raw:
                raise ValueError(f"{name} loaded empty")
            stem = name.split(".")[0]
            self.weights[stem] = {k: v.astype(self.dtype) for k, v in raw.items()}
        return self


def _conv1d_weight(w: mx.array, dtype: mx.Dtype) -> mx.array:
    """Conv1d ``[out, in, k]`` -> MLX ``[out, k, in]``."""
    return w.transpose(0, 2, 1).astype(dtype)


def _convtranspose1d_weight(w: mx.array, dtype: mx.Dtype) -> mx.array:
    """ConvTranspose1d ``[in, out, k]`` -> MLX ``[out, k, in]``."""
    return w.transpose(1, 2, 0).astype(dtype)


def _filter_weight(f: mx.array, channels: int) -> mx.array:
    """Alias-free filter -> depthwise MLX kernel ``[C, k, 1]``, kept fp32.

    Checkpoint filters are ``[1, 1, k]`` (fixed_filter=True) or ``[C, 1, k]``
    (fixed_filter=False, per-channel). Broadcast the fixed case to ``C`` channels,
    then transpose ``[C, 1, k]`` -> ``[C, k, 1]``.

    The taps are forced fp32 regardless of the runtime ``dtype`` (they are tiny and
    the alias-free resampler runs its conv in fp32 — see ``audiovae._AliasFreeResampler``).
    """
    if f.shape[0] == 1 and channels != 1:
        f = mx.broadcast_to(f, (channels, 1, f.shape[2]))
    return f.transpose(0, 2, 1).astype(mx.float32)


def load_audiovae(
    path_to_vocoder_safetensors: str | Path,
    dtype: mx.Dtype = mx.bfloat16,
    *,
    with_encoder: bool = False,
) -> AudioVAE:
    """Build the AudioVAE decode (and optionally encode) path + bind vocoder weights.

    Args:
        path_to_vocoder_safetensors: path to ``vocoder.safetensors`` (un-prefixed,
            weight_norm-folded, fp32 weights).
        dtype: runtime dtype. ``mx.float32`` for parity tests, ``mx.bfloat16``
            for inference. Snake math + alias-free filter taps are fp32 regardless.
        with_encoder: also build + bind the encode path (``audio_encoder`` /
            ``enc_mi_layer`` / ``pre_proj``), enabling ``AudioVAE.encode``.

    Returns:
        an ``AudioVAE`` whose ``decode(latent)`` (and, if ``with_encoder``,
        ``encode(wav)``) reproduces the oracle.
    """
    path = Path(path_to_vocoder_safetensors)
    if not path.exists():
        raise FileNotFoundError(f"vocoder weights not found: {path}")
    cfg_dir = path.parent
    config = (
        ModelConfig.from_checkpoint(cfg_dir).vocoder
        if ((cfg_dir / "config.json").exists())
        else VocoderConfig()
    )

    w = mx.load(str(path))

    # --- post_proj: Conv1d k=1 (128 -> 128) ---
    post_proj = Conv1d(
        _conv1d_weight(w["post_proj.weight"], dtype),
        w["post_proj.bias"].astype(dtype),
        causal=False,
    )

    # --- dec_mi_layer = Sequential(Linear, SLSTM, Linear) ---
    mi_num_layers = config.mi_num_layers
    slstm = _SLSTM(
        weight_ih=[
            w[f"dec_mi_layer.1.lstm.weight_ih_l{n}"].astype(dtype)
            for n in range(mi_num_layers)
        ],
        weight_hh=[
            w[f"dec_mi_layer.1.lstm.weight_hh_l{n}"].astype(dtype)
            for n in range(mi_num_layers)
        ],
        bias_ih=[
            w[f"dec_mi_layer.1.lstm.bias_ih_l{n}"].astype(dtype)
            for n in range(mi_num_layers)
        ],
        bias_hh=[
            w[f"dec_mi_layer.1.lstm.bias_hh_l{n}"].astype(dtype)
            for n in range(mi_num_layers)
        ],
        num_layers=mi_num_layers,
    )

    # --- Decoder ---
    causal = config.causal
    num_kernels = len(config.resblock_kernel_sizes)
    initial = config.upsample_initial_channel

    conv_pre = Conv1d(
        _conv1d_weight(w["decoder.conv_pre.weight"], dtype),
        w["decoder.conv_pre.bias"].astype(dtype),
        causal=False,  # upstream exception: conv_pre is non-causal even in causal decoder
    )

    ups = []
    for i, (u, _k) in enumerate(
        zip(config.upsample_rates, config.upsample_kernel_sizes)
    ):
        ups.append(
            ConvTranspose1d(
                _convtranspose1d_weight(w[f"decoder.ups.{i}.0.weight"], dtype),
                w[f"decoder.ups.{i}.0.bias"].astype(dtype),
                stride=u,
                causal=causal,
            )
        )

    def _resampler(prefix: str, channels: int) -> _AliasFreeResampler:
        return _AliasFreeResampler(
            up_filter=_filter_weight(w[f"{prefix}.upsample.filter"], channels),
            down_filter=_filter_weight(
                w[f"{prefix}.downsample.lowpass.filter"], channels
            ),
            alpha=w[f"{prefix}.act.alpha"].astype(dtype),
            beta=w[f"{prefix}.act.beta"].astype(dtype),
            ratio=2,
        )

    resblocks = []
    for i in range(len(ups)):
        ch = initial // (2 ** (i + 1))
        for j, (k_size, dils) in enumerate(
            zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
        ):
            rb_idx = i * num_kernels + j
            prefix = f"decoder.resblocks.{rb_idx}"
            convs1 = [
                Conv1d(
                    _conv1d_weight(w[f"{prefix}.convs1.{d}.weight"], dtype),
                    w[f"{prefix}.convs1.{d}.bias"].astype(dtype),
                    causal=causal,
                    dilation=dils[d],
                )
                for d in range(num_kernels)
            ]
            convs2 = [
                Conv1d(
                    _conv1d_weight(w[f"{prefix}.convs2.{d}.weight"], dtype),
                    w[f"{prefix}.convs2.{d}.bias"].astype(dtype),
                    causal=causal,
                    dilation=1,
                )
                for d in range(num_kernels)
            ]
            acts = [_resampler(f"{prefix}.activations.{a}", ch) for a in range(6)]
            resblocks.append(_AMPBlock1(convs1=convs1, convs2=convs2, acts=acts))

    final_ch = initial // (2 ** len(ups))
    act_post = _resampler("decoder.activation_post", final_ch)

    conv_post = Conv1d(
        _conv1d_weight(w["decoder.conv_post.weight"], dtype),
        None,  # use_bias_at_final=False
        causal=causal,
    )

    decoder = _Decoder(
        conv_pre=conv_pre,
        ups=ups,
        resblocks=resblocks,
        act_post=act_post,
        conv_post=conv_post,
        num_kernels=num_kernels,
    )

    encoder = None
    enc_lin0_w = enc_lin0_b = enc_lin2_w = enc_lin2_b = None
    enc_slstm = None
    pre_proj = None
    if with_encoder:
        (
            encoder,
            enc_lin0_w,
            enc_lin0_b,
            enc_slstm,
            enc_lin2_w,
            enc_lin2_b,
            pre_proj,
        ) = _build_encoder(w, config, dtype)

    return AudioVAE(
        config=config,
        post_proj=post_proj,
        mi_lin0_w=w["dec_mi_layer.0.weight"].astype(dtype),
        mi_lin0_b=w["dec_mi_layer.0.bias"].astype(dtype),
        slstm=slstm,
        mi_lin2_w=w["dec_mi_layer.2.weight"].astype(dtype),
        mi_lin2_b=w["dec_mi_layer.2.bias"].astype(dtype),
        decoder=decoder,
        encoder=encoder,
        enc_lin0_w=enc_lin0_w,
        enc_lin0_b=enc_lin0_b,
        enc_slstm=enc_slstm,
        enc_lin2_w=enc_lin2_w,
        enc_lin2_b=enc_lin2_b,
        pre_proj=pre_proj,
    )


def _build_encoder(w, config, dtype):
    """Build the AudioVAE encode submodules from ``vocoder.safetensors`` weights.

    Mirrors the upstream ``Encoder.generator`` Sequential index layout:
      generator.0   Conv1d_S(1 -> base, k=3, s=1, causal)   [pre]
      generator.1   LeakyReLU (no params)
      per stage i in 0..5 at base index ``2 + 3*i``:
        +0  Conv1d_S(in -> out, k=2*down, s=down, causal)    [downsample]
        +1  ResStack(out, k=3, base=2, nums=6, causal)
        +2  LeakyReLU
      generator.20  Conv1d_S(768 -> latent_dim, k=5, s=1, NON-causal)  [lookahead post]
    """
    causal = config.causal_encoder
    down_rates = config.downsample_rates  # [2, 2, 2, 4, 6, 10]
    g = "audio_encoder.generator"

    # All encode convs run in high-precision (hp=True) mode: the encoder is a deep,
    # large-magnitude stack gated by a tight max-abs tolerance, and MLX's reduced-
    # precision fp32 matmul/conv otherwise drifts ~1e0 (see layers.hp_matmul).

    # pre proj conv (causal, stride 1, k=3)
    pre_conv = Conv1d(
        _conv1d_weight(w[f"{g}.0.layer.weight"], dtype),
        w[f"{g}.0.layer.bias"].astype(dtype),
        causal=causal,
        hp=True,
    )

    down_convs = []
    res_stacks = []
    n_blocks = 6  # ResStack nums (Encoder stacks=6)
    dil_base = 2  # stack_dilation_base
    for i, down in enumerate(down_rates):
        base_idx = 2 + 3 * i
        # downsample Conv1d_S: kernel = 2*down, stride = down, causal
        down_convs.append(
            Conv1d(
                _conv1d_weight(w[f"{g}.{base_idx}.layer.weight"], dtype),
                w[f"{g}.{base_idx}.layer.bias"].astype(dtype),
                causal=causal,
                stride=down,
                hp=True,
            )
        )
        rs_idx = base_idx + 1
        convs1 = []  # dilated (k=3), block-index .2
        convs2 = []  # dilation=1 (k=3), block-index .5
        for b in range(n_blocks):
            dil = dil_base**b
            convs1.append(
                Conv1d(
                    _conv1d_weight(w[f"{g}.{rs_idx}.layers.{b}.2.weight"], dtype),
                    w[f"{g}.{rs_idx}.layers.{b}.2.bias"].astype(dtype),
                    causal=causal,
                    dilation=dil,
                    hp=True,
                )
            )
            convs2.append(
                Conv1d(
                    _conv1d_weight(w[f"{g}.{rs_idx}.layers.{b}.5.weight"], dtype),
                    w[f"{g}.{rs_idx}.layers.{b}.5.bias"].astype(dtype),
                    causal=causal,
                    dilation=1,
                    hp=True,
                )
            )
        res_stacks.append(_ResStack(convs1=convs1, convs2=convs2))

    # lookahead post conv: NON-causal (kernel = 2*lookahead + 1 = 5), stride 1
    post_idx = 2 + 3 * len(down_rates)
    post_conv = Conv1d(
        _conv1d_weight(w[f"{g}.{post_idx}.layer.weight"], dtype),
        w[f"{g}.{post_idx}.layer.bias"].astype(dtype),
        causal=False,
        hp=True,
    )

    encoder = _Encoder(
        pre_conv=pre_conv,
        down_convs=down_convs,
        res_stacks=res_stacks,
        post_conv=post_conv,
    )

    # enc_mi_layer = Sequential(Linear, SLSTM, Linear)
    mi_num_layers = config.mi_num_layers
    enc_slstm = _SLSTM(
        weight_ih=[
            w[f"enc_mi_layer.1.lstm.weight_ih_l{n}"].astype(dtype)
            for n in range(mi_num_layers)
        ],
        weight_hh=[
            w[f"enc_mi_layer.1.lstm.weight_hh_l{n}"].astype(dtype)
            for n in range(mi_num_layers)
        ],
        bias_ih=[
            w[f"enc_mi_layer.1.lstm.bias_ih_l{n}"].astype(dtype)
            for n in range(mi_num_layers)
        ],
        bias_hh=[
            w[f"enc_mi_layer.1.lstm.bias_hh_l{n}"].astype(dtype)
            for n in range(mi_num_layers)
        ],
        num_layers=mi_num_layers,
        hp=True,
    )

    # pre_proj: Conv1d k=1 (latent_dim -> 2*latent_dim)
    pre_proj = Conv1d(
        _conv1d_weight(w["pre_proj.weight"], dtype),
        w["pre_proj.bias"].astype(dtype),
        causal=False,
        hp=True,
    )

    return (
        encoder,
        w["enc_mi_layer.0.weight"].astype(dtype),
        w["enc_mi_layer.0.bias"].astype(dtype),
        enc_slstm,
        w["enc_mi_layer.2.weight"].astype(dtype),
        w["enc_mi_layer.2.bias"].astype(dtype),
        pre_proj,
    )


def _conv2d_weight(w: mx.array, dtype: mx.Dtype) -> mx.array:
    """Conv2d ``[out, in, kh, kw]`` -> MLX ``[out, kh, kw, in]``."""
    return w.transpose(0, 2, 3, 1).astype(dtype)


def _fold_bn(
    w: dict[str, mx.array], prefix: str, dtype: mx.Dtype, *, affine: bool = True
) -> _BN:
    """Fold a frozen BatchNorm to an affine ``(scale, shift)`` over the channel axis.

    ``y = (x - rm)/sqrt(rv + eps)*gamma + beta`` (eps 1e-5). For affine-free BN
    (config ``batchnorm_``, the final dense), gamma=1 / beta=0 so it reduces to
    ``(x - rm)/sqrt(rv + eps)``. Keys live at ``<prefix>.{running_mean,running_var,
    weight,bias}``; the result broadcasts against channels-last activations.
    """
    eps = 1e-5
    rm = w[f"{prefix}.running_mean"].astype(mx.float32)
    rv = w[f"{prefix}.running_var"].astype(mx.float32)
    inv = mx.rsqrt(rv + eps)
    if affine:
        gamma = w[f"{prefix}.weight"].astype(mx.float32)
        beta = w[f"{prefix}.bias"].astype(mx.float32)
    else:
        gamma = mx.ones_like(rm)
        beta = mx.zeros_like(rm)
    scale = (gamma * inv).astype(dtype)
    shift = (beta - rm * gamma * inv).astype(dtype)
    return _BN(scale, shift)


def load_speaker(
    path_to_speaker_safetensors: str | Path, dtype: mx.Dtype = mx.float32
) -> CAMPPlus:
    """Build the pure-MLX CAM++ x-vector encoder + bind ``speaker.safetensors`` weights.

    The converted ``speaker.safetensors`` preserves the upstream keys verbatim:
    a ``model.`` prefix (the wrapping ``SpeakerXVectorFeatures`` held the CAMPPlus as
    ``self.model``), the expected layout conv weights, raw BatchNorm running stats, and a
    wrapper-only ``resample.kernel`` buffer. This loader strips the prefix, transposes
    conv weights to MLX layout, and folds every BN to a frozen affine.

    Args:
        path_to_speaker_safetensors: path to ``speaker.safetensors``.
        dtype: runtime dtype (``mx.float32`` for the parity gate).

    Returns:
        a ``CAMPPlus`` whose ``model(fbank[B, T, 80]) -> [B, 512]`` reproduces the
        oracle (cosine >= 0.9995).
    """
    path = Path(path_to_speaker_safetensors)
    if not path.exists():
        raise FileNotFoundError(f"speaker weights not found: {path}")
    raw = mx.load(str(path))
    # Strip the wrapper ``model.`` prefix; drop the wrapper-only resample buffer.
    w = {k[len("model.") :]: v for k, v in raw.items() if k.startswith("model.")}

    def conv2d(name, stride, padding):
        return _Conv2d(
            _conv2d_weight(w[f"{name}.weight"], dtype),
            None,  # all FCM convs are bias=False
            stride=stride,
            padding=padding,
        )

    def res_block(name, stride):
        # conv1: k3 stride (stride,1) p1 ; conv2: k3 s1 p1 ; shortcut k1 (stride,1) p0
        shortcut_conv = shortcut_bn = None
        if f"{name}.shortcut.0.weight" in w:
            shortcut_conv = _Conv2d(
                _conv2d_weight(w[f"{name}.shortcut.0.weight"], dtype),
                None,
                stride=(stride, 1),
                padding=(0, 0),
            )
            shortcut_bn = _fold_bn(w, f"{name}.shortcut.1", dtype)
        return _BasicResBlock(
            conv1=conv2d(f"{name}.conv1", (stride, 1), (1, 1)),
            bn1=_fold_bn(w, f"{name}.bn1", dtype),
            conv2=conv2d(f"{name}.conv2", (1, 1), (1, 1)),
            bn2=_fold_bn(w, f"{name}.bn2", dtype),
            shortcut_conv=shortcut_conv,
            shortcut_bn=shortcut_bn,
        )

    head = FCM(
        conv1=conv2d("head.conv1", (1, 1), (1, 1)),
        bn1=_fold_bn(w, "head.bn1", dtype),
        layer1=[res_block("head.layer1.0", 2), res_block("head.layer1.1", 1)],
        layer2=[res_block("head.layer2.0", 2), res_block("head.layer2.1", 1)],
        conv2=conv2d("head.conv2", (2, 1), (1, 1)),
        bn2=_fold_bn(w, "head.bn2", dtype),
        out_channels=320,
    )

    # --- xvector.tdnn: Conv1d 320->128 k5 s2 dil1 pad2, BN-ReLU
    tdnn = _TDNNLayer(
        conv=_Conv1d(
            _conv1d_weight(w["xvector.tdnn.linear.weight"], dtype),
            None,
            stride=2,
            padding=2,
            dilation=1,
        ),
        bn=_fold_bn(w, "xvector.tdnn.nonlinear.batchnorm", dtype),
    )

    def dense_block(block_name, num_layers, dilation):
        layers = []
        for i in range(num_layers):
            ln = f"xvector.{block_name}.tdnnd{i + 1}"
            cam = _CAMLayer(
                linear_local=_Conv1d(
                    _conv1d_weight(w[f"{ln}.cam_layer.linear_local.weight"], dtype),
                    None,  # cam_layer.linear_local bias=False
                    stride=1,
                    padding=dilation,  # (k-1)//2 * dilation, k=3
                    dilation=dilation,
                ),
                linear1=_Conv1d(
                    _conv1d_weight(w[f"{ln}.cam_layer.linear1.weight"], dtype),
                    w[f"{ln}.cam_layer.linear1.bias"].astype(dtype),
                ),
                linear2=_Conv1d(
                    _conv1d_weight(w[f"{ln}.cam_layer.linear2.weight"], dtype),
                    w[f"{ln}.cam_layer.linear2.bias"].astype(dtype),
                ),
            )
            layers.append(
                _CAMDenseTDNNLayer(
                    bn1=_fold_bn(w, f"{ln}.nonlinear1.batchnorm", dtype),
                    relu1=None,
                    linear1=_Conv1d(
                        _conv1d_weight(w[f"{ln}.linear1.weight"], dtype),
                        None,  # linear1 bias=False
                    ),
                    bn2=_fold_bn(w, f"{ln}.nonlinear2.batchnorm", dtype),
                    relu2=None,
                    cam_layer=cam,
                )
            )
        return _CAMDenseTDNNBlock(layers)

    def transit(name):
        return _TransitLayer(
            bn=_fold_bn(w, f"xvector.{name}.nonlinear.batchnorm", dtype),
            linear=_Conv1d(
                _conv1d_weight(w[f"xvector.{name}.linear.weight"], dtype),
                None,  # transit bias=False
            ),
        )

    block1 = dense_block("block1", 12, 1)
    block2 = dense_block("block2", 24, 2)
    block3 = dense_block("block3", 16, 2)

    out_nonlinear = _fold_bn(w, "xvector.out_nonlinear.batchnorm", dtype)
    dense_linear = _Conv1d(
        _conv1d_weight(w["xvector.dense.linear.weight"], dtype),
        None,  # DenseLayer bias=False
    )
    dense_bn = _fold_bn(w, "xvector.dense.nonlinear.batchnorm", dtype, affine=False)

    return CAMPPlus(
        head=head,
        tdnn=tdnn,
        block1=block1,
        transit1=transit("transit1"),
        block2=block2,
        transit2=transit("transit2"),
        block3=block3,
        transit3=transit("transit3"),
        out_nonlinear=out_nonlinear,
        dense_linear=dense_linear,
        dense_bn=dense_bn,
    )


def _linear(w: dict, prefix: str, dtype: mx.Dtype, *, bias: bool = True) -> Linear:
    """Build a ``Linear`` from the expected layout ``<prefix>.weight`` (+ optional bias)."""
    weight = w[f"{prefix}.weight"].astype(dtype)
    b = w[f"{prefix}.bias"].astype(dtype) if bias else None
    return Linear(weight, b)


def load_dit(path_to_core_safetensors: str | Path, dtype: mx.Dtype = mx.float32) -> DiT:
    """Build the pure-MLX flow-matching DiT + bind ``velocity_field_predictor.*`` weights.

    The converted ``core.safetensors`` keeps the upstream keys verbatim (un-prefixed
    the expected layout fp32). This loader maps the 244 ``velocity_field_predictor.*`` keys
    onto the MLX ``DiT``:

      * Linear weights stay the expected layout ``[out, in]`` (``layers.Linear`` applies
        ``x @ W.T``) — no transpose.
      * ``attn.{q,k,v}_proj`` (no bias) / ``attn.o_proj`` (has bias) / ``attn.{q,k}_norm``
        (RMSNorm weight) are bound via ``MultiHeadAttention.load_weights`` (the
        npz-key-safe ``w_<dotted>`` scheme).
      * ``norm1`` / ``norm2`` / ``output_layer.norm`` are affine-free (no stored
        weights) — skipped.
      * attn rotary has no stored params (recomputed in ``layers``).

    The block attention runs on the fast (hp=False) path; see ``dit`` for numerics.

    In meanflow mode (``config.json`` carries an enabled ``meanflow`` block) the
    checkpoint additionally carries 4 ``velocity_field_predictor.duration_embedder.mlp.{0,2}.{weight,bias}``
    keys; this loader binds them onto ``DiT.duration_embedder`` (a ``TimestepEmbedder``)
    using the same ``_linear`` machinery as ``time_embedder``. In flow-matching mode
    that submodule is ``None`` and those keys are absent.

    Args:
        path_to_core_safetensors: path to ``core.safetensors``.
        dtype: runtime dtype (``mx.float32`` for the parity gate).

    Returns:
        a ``DiT`` whose ``__call__``
    """
    path = Path(path_to_core_safetensors)
    if not path.exists():
        raise FileNotFoundError(f"core weights not found: {path}")
    raw = mx.load(str(path))
    p = "velocity_field_predictor."
    w = {k[len(p) :]: v for k, v in raw.items() if k.startswith(p)}

    cfg_dir = path.parent
    config = ModelConfig.from_checkpoint(cfg_dir)
    dit_cfg = config.dit
    hidden = dit_cfg.hidden_size
    num_layers = dit_cfg.num_layers

    input_layer = _linear(w, "input_layer", dtype)

    time_embedder = TimestepEmbedder(
        _linear(w, "time_embedder.mlp.0", dtype),
        _linear(w, "time_embedder.mlp.2", dtype),
    )

    duration_embedder = None
    if config.mode == "meanflow":
        if "duration_embedder.mlp.0.weight" not in w:
            raise ValueError(
                "config declares meanflow mode but core.safetensors has no "
                "duration_embedder.* keys (wrong checkpoint?)."
            )
        duration_embedder = TimestepEmbedder(
            _linear(w, "duration_embedder.mlp.0", dtype),
            _linear(w, "duration_embedder.mlp.2", dtype),
        )

    blocks: list[DiTBlock] = []
    for i in range(num_layers):
        bp = f"blocks.{i}"
        attn = MultiHeadAttention(
            hidden,
            dit_cfg.num_heads,
            qkv_bias=dit_cfg.qkv_bias,
            qk_norm=dit_cfg.qk_norm,
            norm_layer=dit_cfg.norm_layer,
            rotary_bias=dit_cfg.rotary_bias,
            rotary_theta=dit_cfg.rotary_theta,
        )
        attn.load_weights(
            {
                "w_q_proj_weight": w[f"{bp}.attn.q_proj.weight"].astype(dtype),
                "w_k_proj_weight": w[f"{bp}.attn.k_proj.weight"].astype(dtype),
                "w_v_proj_weight": w[f"{bp}.attn.v_proj.weight"].astype(dtype),
                "w_o_proj_weight": w[f"{bp}.attn.o_proj.weight"].astype(dtype),
                "w_o_proj_bias": w[f"{bp}.attn.o_proj.bias"].astype(dtype),
                "w_q_norm_weight": w[f"{bp}.attn.q_norm.weight"].astype(dtype),
                "w_k_norm_weight": w[f"{bp}.attn.k_norm.weight"].astype(dtype),
            }
        )
        ffn = Mlp(
            _linear(w, f"{bp}.ffn.fc1", dtype),
            _linear(w, f"{bp}.ffn.fc2", dtype),
            gelu_tanh,
        )
        adaLN = _linear(w, f"{bp}.adaLN_modulation.1", dtype)
        blocks.append(DiTBlock(attn, ffn, adaLN, hidden_size=hidden))

    output_layer = FinalLayer(
        _linear(w, "output_layer.adaLN_modulation.1", dtype),
        _linear(w, "output_layer.linear", dtype),
        hidden_size=hidden,
    )

    return DiT(
        input_layer,
        time_embedder,
        blocks,
        output_layer,
        duration_embedder=duration_embedder,
    )


def load_coordinate_proj(
    path_to_core_safetensors: str | Path, dtype: mx.Dtype = mx.float32
) -> Linear:
    """Build the flow-matching ``coordinate_proj`` ``Linear(latent_dim, hidden_size)``.

    The noisy-latent projection used by the euler/CFG solver lives at
    ``coordinate_proj.{weight,bias}`` in ``core.safetensors`` (un-prefixed,
    the expected layout ``[hidden_size, latent_dim]`` = ``[1024, 128]``). It is NOT under the
    ``velocity_field_predictor.`` prefix, so ``load_dit`` does not pick it up.

    Args:
        path_to_core_safetensors: path to ``core.safetensors``.
        dtype: runtime dtype (``mx.float32`` for the parity gate).

    Returns:
        a ``layers.Linear`` mapping ``[..., latent_dim] -> [..., hidden_size]``.
    """
    path = Path(path_to_core_safetensors)
    if not path.exists():
        raise FileNotFoundError(f"core weights not found: {path}")
    raw = mx.load(str(path))
    if "coordinate_proj.weight" not in raw:
        raise ValueError(f"no coordinate_proj.* keys in {path}")
    return Linear(
        raw["coordinate_proj.weight"].astype(dtype),
        raw["coordinate_proj.bias"].astype(dtype),
    )


def load_semantic_encoder(
    path_to_core_safetensors: str | Path, dtype: mx.Dtype = mx.float32
) -> VAESemanticEncoder:
    """Build the pure-MLX VAE semantic encoder + bind ``patch_encoder.*`` weights.

    Maps the un-prefixed ``patch_encoder.*`` keys from ``core.safetensors`` onto the
    MLX ``VAESemanticEncoder`` (in_dim=128, out_dim=1536, patch_size=4):

      * ``ds_proj`` — causal stride-2 ``Conv1d(128, 128, k=2)`` (``[out, in, k]``
        -> MLX ``[out, k, in]``); ``causal=True`` left-pads ``dilation*(k-1)=1``.
      * ``in_proj`` — ``Linear(128 -> 1024)``; ``out_proj`` — ``Linear(2048 -> 1536)``.
      * 24 ``TransformerEncoderLayer``: ``attn_norm`` / ``ffn_norm`` are affine
        ``RMSNorm(1024)``; ``attn`` is ``MultiHeadAttention(1024, 16)`` with
        **NO qk_norm / NO rotary / NO qkv_bias** (config says otherwise but the
        upstream ``SuperviseEncoder`` does not forward those — see module docstring);
        ``ffn`` is a SiLU ``Mlp(1024 -> 4096 -> 1024)``.

    CONFIG-VS-CODE GUARD: asserts the checkpoint has NO ``patch_encoder.encoder.
    layers.*.attn.q_norm`` / ``k_norm`` / ``rotary`` keys before wiring — this proves
    the encoder runs without qk-norm / rotary regardless of the config block.

    Args:
        path_to_core_safetensors: path to ``core.safetensors``.
        dtype: runtime dtype (``mx.float32`` for the parity gate).

    Returns:
        a ``VAESemanticEncoder`` whose ``__call__([B, T, 128]) -> [B, T/4, 1536]``

    """
    path = Path(path_to_core_safetensors)
    if not path.exists():
        raise FileNotFoundError(f"core weights not found: {path}")
    raw = mx.load(str(path))
    p = "patch_encoder."
    w = {k[len(p) :]: v for k, v in raw.items() if k.startswith(p)}
    if not w:
        raise ValueError(f"no patch_encoder.* keys in {path}")

    # CONFIG-VS-CODE GUARD: the trained encoder has no qk-norm / rotary params,
    # despite config.PatchEncoder.{qk_norm,rotary_bias}=True (SuperviseEncoder does
    # not forward those flags into the layers). Fail loudly if that ever changes.
    forbidden = [
        k
        for k in w
        if (".q_norm" in k or ".k_norm" in k or "rotary" in k or "inv_freq" in k)
    ]
    assert not forbidden, (
        "patch_encoder unexpectedly carries qk-norm / rotary weights "
        f"(config-vs-code resolution broken): {forbidden}"
    )

    cfg_dir = path.parent
    config = ModelConfig.from_checkpoint(cfg_dir)
    enc_cfg = config.encoder
    hidden = enc_cfg.hidden_size  # 1024
    num_layers = enc_cfg.num_layers  # 24
    num_heads = enc_cfg.num_heads  # 16
    # The MLX EncoderConfig already forces qk_norm / rotary OFF (Task 0 finding);
    # assert here so a config-schema regression can't silently re-enable them.
    assert (
        not enc_cfg.qk_norm and not enc_cfg.rotary_bias
    ), "EncoderConfig must force qk_norm / rotary OFF for the patch encoder"

    # ds_proj: causal stride-2 Conv1d(128, 128, k=2). [out, in, k] -> MLX [out, k, in].
    ds_proj = Conv1d(
        _conv1d_weight(w["ds_proj.weight"], dtype),
        w["ds_proj.bias"].astype(dtype),
        causal=True,
        stride=2,
    )

    in_proj = _linear(w, "in_proj", dtype)
    out_proj = _linear(w, "out_proj", dtype)

    layers: list[TransformerEncoderLayer] = []
    for i in range(num_layers):
        lp = f"encoder.layers.{i}"
        attn_norm = RMSNorm(hidden, weight=w[f"{lp}.attn_norm.weight"].astype(dtype))
        ffn_norm = RMSNorm(hidden, weight=w[f"{lp}.ffn_norm.weight"].astype(dtype))
        attn = MultiHeadAttention(
            hidden,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            rotary_bias=False,
        )
        attn.load_weights(
            {
                "w_q_proj_weight": w[f"{lp}.attn.q_proj.weight"].astype(dtype),
                "w_k_proj_weight": w[f"{lp}.attn.k_proj.weight"].astype(dtype),
                "w_v_proj_weight": w[f"{lp}.attn.v_proj.weight"].astype(dtype),
                "w_o_proj_weight": w[f"{lp}.attn.o_proj.weight"].astype(dtype),
                "w_o_proj_bias": w[f"{lp}.attn.o_proj.bias"].astype(dtype),
            }
        )
        ffn = Mlp(
            _linear(w, f"{lp}.ffn.fc1", dtype),
            _linear(w, f"{lp}.ffn.fc2", dtype),
            silu,
        )
        layers.append(TransformerEncoderLayer(attn_norm, attn, ffn_norm, ffn))

    encoder = SuperviseEncoder(layers, causal=enc_cfg.causal)

    return VAESemanticEncoder(
        ds_proj=ds_proj,
        in_proj=in_proj,
        encoder=encoder,
        out_proj=out_proj,
        out_ds_rate=config.patch_size // 2,
        patch_size=config.patch_size,
    )


def _load_container(path: Path, dtype: mx.Dtype) -> DotsTts:
    """Validate the converted assets + build the bare ``DotsTts`` container.

    Confirms ``config.json`` / ``llm_config.json`` / ``latent_stats.npz`` /
    ``tokenizer/`` / the three safetensors are present and parseable, and stashes the
    raw weight dicts. Does NOT wire the AR runtime (that is ``from_pretrained``).
    """
    if not path.exists():
        raise FileNotFoundError(f"model directory not found: {path}")

    config = ModelConfig.from_checkpoint(path)

    stats_path = path / "latent_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"missing latent_stats.npz: {stats_path}")
    stats = mx.load(str(stats_path))
    if "mean" not in stats or "var" not in stats:
        raise ValueError(f"latent_stats.npz missing mean/var keys: {sorted(stats)}")

    tokenizer_dir = path / "tokenizer"
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"missing tokenizer dir: {tokenizer_dir}")

    container = DotsTts(
        config=config,
        dtype=dtype,
        path=path,
        latent_mean=stats["mean"].astype(mx.float32),
        latent_var=stats["var"].astype(mx.float32),
        tokenizer_dir=tokenizer_dir,
    )
    return container.load_weights()


def from_pretrained(path: str | Path, dtype: mx.Dtype = mx.bfloat16) -> DotsTts:
    """Load + FULLY WIRE a converted dots.tts model from ``path``.

    Builds the bare ``DotsTts`` container (validating + loading assets), then wires
    the complete AR runtime — tokenizer, Qwen2.5 LLM + eos head, AudioVAE
    encode+decode, CAM++ x-vector, flow-matching ``FlowSolver`` (DiT +
    coordinate_proj), the semantic patch encoder, and the hidden / latent / xvec
    projections — attaching the resulting ``DotsTtsModel`` as ``container.model``.

    Args:
        path: directory containing ``{core,vocoder,speaker}.safetensors``,
            ``latent_stats.npz``, ``config.json`` + ``llm_config.json``, and ``tokenizer/``.
        dtype: runtime cast dtype. ``mx.bfloat16`` for inference, ``mx.float32`` for
            parity / behavioral tests.

    Returns:
        a ``DotsTts`` with assets loaded, safetensors validated, and ``.model`` set to
        the fully-wired ``DotsTtsModel`` (call ``container.model.generate(...)``).
    """
    path = Path(path)
    container = _load_container(path, dtype)
    # Lazy import to avoid a circular import (model.py imports loader.py).
    from .model import DotsTtsModel

    container.model = DotsTtsModel.from_pretrained(path, dtype=dtype)
    return container
