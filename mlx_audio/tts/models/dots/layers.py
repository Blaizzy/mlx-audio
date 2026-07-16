"""Pure-MLX 1-D conv primitives for the dots.tts AudioVAE encode + decode paths.

Imports ONLY mlx — never torch, never numpy. These mirror the upstream causal
``Conv1d`` / ``ConvTranspose1d`` from ``dots_tts.modules.backbone.layers`` (and the
strided causal ``Conv1d_S`` used by the encoder) plus a replicate (edge-clamp) pad
helper used by the alias-free resamplers.

Tensor convention: tensors flow as NLC ``[B, T, C]`` (MLX-native channels-last).
Weight tensors are stored in torch layout and transposed to MLX layout at load
time by ``loader.py`` (Conv1d: ``[out, in, k]`` -> ``[out, k, in]``;
ConvTranspose1d: ``[in, out, k]`` -> ``[out, k, in]``).
"""

from __future__ import annotations

import mlx.core as mx


def hp_matmul(x: mx.array, w_t: mx.array) -> mx.array:
    """High-precision ``x @ w_t`` accumulated in true fp32 (contraction over last axis).

    MLX's ``mx.matmul`` / ``mx.conv1d`` on Apple Silicon round each fp32 operand to a
    reduced-precision (~tf32) mantissa before the multiply-accumulate, which costs up
    to ~1e-1 absolute on large-magnitude inputs. Computing the contraction as an
    explicit elementwise multiply + ``mx.sum`` keeps the reduction in genuine fp32
    (matches the torch oracle to ~1e-6). Used on the *encode* path, whose deep stack
    is gated by a tight max-abs tolerance; the decode path keeps the faster
    ``mx.matmul`` (it is PSNR-gated, not max-abs).

    CAVEAT (I1): ``w_t`` MUST be a 2-D right operand ``[K, N]`` and the call
    materializes the broadcast product ``[..., K, N]`` before reducing — i.e.
    ``O(M·N·K)`` peak memory for an ``[M, K]`` left operand. It is therefore
    suitable only for the activation@2-D-weight projections (q/k/v/o, fc1/fc2),
    NOT for the batched attention bmm (``QKᵀ`` / ``attn·V``, whose right operand
    is itself batched) and NOT for large-``N`` projections without chunking. The
    attention parity path computes its (tiny-``L``) bmm with a separate explicit
    fp32 reduction; the runtime path uses fast SDPA.

    Args:
        x: ``[..., K]``.
        w_t: ``[K, N]`` (already transposed; i.e. for a torch ``Linear`` weight
            ``W`` of shape ``[N, K]`` pass ``W.T``).

    Returns:
        ``[..., N]`` in fp32.
    """
    x = x.astype(mx.float32)
    w_t = w_t.astype(mx.float32)
    return mx.sum(x[..., :, None] * w_t, axis=-2)


def replicate_pad(x: mx.array, left: int, right: int) -> mx.array:
    """Edge-clamp pad the time axis of an NLC tensor (mirrors ``F.pad(mode="replicate")``).

    ``mx.pad`` only supports constant padding, so the edge frames are repeated and
    concatenated manually. ``x`` is ``[B, T, C]``; padding is applied along T.
    """
    if left == 0 and right == 0:
        return x
    pieces = []
    if left > 0:
        pieces.append(mx.broadcast_to(x[:, :1, :], (x.shape[0], left, x.shape[2])))
    pieces.append(x)
    if right > 0:
        pieces.append(mx.broadcast_to(x[:, -1:, :], (x.shape[0], right, x.shape[2])))
    return mx.concatenate(pieces, axis=1)


class Conv1d:
    """Causal / "same"-padded (optionally strided) 1-D convolution over NLC tensors.

    Mirrors both ``dots_tts.modules.backbone.layers.Conv1d`` (decode path, stride=1)
    and the encoder's ``Conv1d_S`` (which adds a stride for downsampling):
      * ``causal=True``  -> left-pad ``dilation * (kernel - 1)`` zeros, conv with
        padding=0. For ``Conv1d_S`` the dilation is always 1, so the left pad is
        ``kernel - 1`` (== its ``causal_pad``).
      * ``causal=False`` -> symmetric "same" pad ``(kernel*dilation - dilation)//2``
        (handled by ``mx.conv1d``'s ``padding`` arg). ``Conv1d_S``'s post-projection
        (lookahead) conv uses this non-causal path.

    ``weight`` is the MLX-layout kernel ``[out, k, in]``; ``bias`` is ``[out]`` or None.

    ``hp=True`` runs the conv as a tap-loop of fp32 elementwise-multiply + ``mx.sum``
    reductions (see ``hp_matmul``) instead of ``mx.conv1d``, matching the torch oracle
    to ~1e-6. The encode path sets this (it is max-abs gated and its deep, large-
    magnitude conv stack otherwise drifts ~1e0); ``groups != 1`` is not supported in
    the HP path (only the channel-mixing encode convs use it, all groups=1).
    """

    def __init__(
        self,
        weight: mx.array,
        bias: mx.array | None,
        *,
        causal: bool,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        hp: bool = False,
    ):
        self.weight = weight
        self.bias = bias
        self.causal = causal
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.hp = hp
        self.kernel_size = weight.shape[1]
        if causal:
            self.left_padding = dilation * (self.kernel_size - 1)
            self.padding = 0
        else:
            self.left_padding = 0
            self.padding = (self.kernel_size * dilation - dilation) // 2

    def __call__(self, x: mx.array) -> mx.array:
        if self.hp:
            return self._call_hp(x)
        if self.causal and self.left_padding > 0:
            x = mx.pad(x, [(0, 0), (self.left_padding, 0), (0, 0)])
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y

    def _call_hp(self, x: mx.array) -> mx.array:
        """fp32 tap-loop conv: out[t,o] = sum_j sum_i x_padded[t*s + j*d, i] * W[o,j,i]."""
        x = x.astype(mx.float32)
        if self.causal:
            x = mx.pad(x, [(0, 0), (self.left_padding, 0), (0, 0)])
        elif self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])
        w = self.weight.astype(mx.float32)  # [Cout, k, Cin]
        cout, k, _ = w.shape
        s, d = self.stride, self.dilation
        t_pad = x.shape[1]
        t_out = (t_pad - d * (k - 1) - 1) // s + 1
        acc = None
        for j in range(k):
            # gather the j-th tap at the strided output positions: [B, t_out, Cin]
            seg = x[:, j * d : j * d + s * t_out : s, :]
            # contract over Cin in fp32: [B, t_out, 1, Cin] * [Cout, Cin] -> [B, t_out, Cout]
            contrib = mx.sum(seg[:, :, None, :] * w[None, None, :, j, :], axis=3)
            acc = contrib if acc is None else acc + contrib
        if self.bias is not None:
            acc = acc + self.bias.astype(mx.float32)
        return acc


class ConvTranspose1d:
    """Causal / non-causal transposed 1-D convolution over NLC tensors.

    Mirrors ``dots_tts.modules.backbone.layers.ConvTranspose1d``: run the standard
    transposed conv, then if ``causal`` trim the last ``stride`` time samples.
    For the decoder upsamplers ``padding=0`` (causal) always.

    ``weight`` is the MLX-layout kernel ``[out, k, in]``; ``bias`` is ``[out]`` or None.
    """

    def __init__(
        self,
        weight: mx.array,
        bias: mx.array | None,
        *,
        stride: int,
        causal: bool,
    ):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.causal = causal

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv_transpose1d(x, self.weight, stride=self.stride, padding=0)
        if self.bias is not None:
            y = y + self.bias
        if self.causal:
            y = y[:, : -self.stride, :]
        return y


# --------------------------------------------------------------------------- #
# Shared transformer primitives (reused by the DiT (T6) and semantic encoder (T7)).
#
# Convention (M1): plain classes holding ``mx.array`` weights — NOT ``mlx.nn.Module``.
# Weights are stored in torch layout (``Linear.weight`` is ``[out, in]``) and applied
# as ``x @ W.T``. The ``hp`` flag routes the activation@2-D-weight matmuls through the
# fp32 ``hp_matmul`` reduction (tight max-abs parity vs the torch oracle); ``hp=False``
# uses MLX's fast (~tf32) matmul / SDPA for the runtime path.
# --------------------------------------------------------------------------- #


class Linear:
    """Affine map ``x @ W.T (+ b)`` with a torch-layout weight ``[out, in]``.

    ``hp=True`` contracts through ``hp_matmul`` (true fp32); ``hp=False`` uses the
    fast ``mx.matmul`` (rounds fp32 -> ~tf32 on Apple Silicon).
    """

    def __init__(self, weight: mx.array, bias: mx.array | None = None):
        self.weight = weight  # [out, in] (torch layout)
        self.bias = bias  # [out] or None

    def __call__(self, x: mx.array, *, hp: bool = False) -> mx.array:
        if hp:
            y = hp_matmul(x, self.weight.T)
            if self.bias is not None:
                y = y + self.bias.astype(mx.float32)
            return y
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class RMSNorm:
    """Root-mean-square layer norm with a learned per-feature ``weight`` (gamma).

    Mirrors ``torch.nn.RMSNorm`` (eps=None default -> ``finfo(fp32).eps``): the
    rms reduction is computed in fp32, then scaled by ``weight``. Used for the
    attention qk-norm (``RMSNorm(head_dim)``) and the semantic-encoder norms.
    """

    # torch's nn.RMSNorm(dim) is constructed with eps=None, which it resolves to
    # ``torch.finfo(x.dtype).eps`` (fp32 -> 1.1920928955078125e-07). Confirmed
    # against the oracle: using 1e-5 here drifts ~1.3e-5 max-abs.
    DEFAULT_EPS = 1.1920928955078125e-07

    def __init__(
        self, dim: int, eps: float | None = None, weight: mx.array | None = None
    ):
        self.dim = dim
        self.eps = self.DEFAULT_EPS if eps is None else eps
        # weight defaults to ones (the torch init) until load_weights overrides it.
        self.weight = weight if weight is not None else mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        in_dtype = x.dtype
        xf = x.astype(mx.float32)
        var = mx.mean(xf * xf, axis=-1, keepdims=True)
        xf = xf * mx.rsqrt(var + self.eps)
        out = xf * self.weight.astype(mx.float32)
        return out.astype(in_dtype)


class LayerNorm:
    """Affine-free layer norm (``elementwise_affine=False``, eps 1e-5).

    Mirrors ``nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)`` used by the
    DiT adaLN path (T6): normalize over the last axis, no learned scale/shift.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        in_dtype = x.dtype
        xf = x.astype(mx.float32)
        mean = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.mean((xf - mean) ** 2, axis=-1, keepdims=True)
        out = (xf - mean) * mx.rsqrt(var + self.eps)
        return out.astype(in_dtype)


class _Identity:
    """No-op stand-in for qk-norm=False (mirrors ``nn.Identity``)."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


def rotate_half(x: mx.array) -> mx.array:
    """GPT-NeoX half-half rotation: ``cat([-x[..., d/2:], x[..., :d/2]], -1)``."""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(emb: mx.array, t: mx.array) -> mx.array:
    """Apply rotary embedding ``emb`` (the ``[B, L, dim]`` freqs) to heads ``t``.

    ``emb`` is the duplicated freqs ``cat([freqs, freqs], -1)`` from
    ``RotaryEmbedding.forward``; here we take its cos/sin directly (NOT of a
    further-transformed tensor). A 3-D ``emb`` is unsqueezed to ``[B, 1, L, dim]``
    to broadcast over the head axis of ``t`` (``[B, H, L, dim]``). Computed in fp32.
    """
    emb = emb.astype(mx.float32)
    if emb.ndim == 3:
        emb = emb[:, None, :, :]
    tf = t.astype(mx.float32)
    out = tf * mx.cos(emb) + rotate_half(tf) * mx.sin(emb)
    return out.astype(t.dtype)


class RotaryEmbedding:
    """GPT-NeoX rotary position embedding (half-half, fp32).

    ``inv_freq[j] = theta^(-2j/dim)`` for ``j = 0..dim/2-1``. ``forward(pos)`` takes
    float positions ``[B, L]`` (or ``[L]``), computes ``freqs = pos ⊗ inv_freq``
    ``[B, L, dim/2]``, and returns ``cat([freqs, freqs], -1)`` ``[B, L, dim]``. All
    arithmetic is forced to fp32 (upstream disables autocast + keeps inv_freq fp32).
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = float(theta)
        j = mx.arange(0, dim, 2).astype(mx.float32)  # [dim/2]
        self.inv_freq = 1.0 / (self.theta ** (j / dim))  # [dim/2], fp32

    def __call__(self, pos: mx.array) -> mx.array:
        pos = pos.astype(mx.float32)
        if pos.ndim == 1:
            # [L] ⊗ [dim/2] -> [L, dim/2]
            freqs = pos[:, None] * self.inv_freq[None, :]
        else:
            # [B, L] ⊗ [dim/2] -> [B, L, dim/2]
            freqs = pos[..., None] * self.inv_freq
        return mx.concatenate([freqs, freqs], axis=-1)


class Mlp:
    """Two-layer FFN: ``fc2(act(fc1(x)))`` with bias on both projections.

    The activation is supplied by the caller: DiT uses tanh-approx GELU, the
    semantic encoder (T7) uses SiLU. ``hp`` routes both Linear layers through the
    fp32 ``hp_matmul`` path; the activation runs in fp32 under ``hp`` too.
    """

    def __init__(self, fc1: Linear, fc2: Linear, act):
        self.fc1 = fc1
        self.fc2 = fc2
        self.act = act

    def __call__(self, x: mx.array, *, hp: bool = False) -> mx.array:
        x = self.fc1(x, hp=hp)
        x = self.act(x)
        return self.fc2(x, hp=hp)


def gelu_tanh(x: mx.array) -> mx.array:
    """Tanh-approx GELU (matches ``nn.GELU(approximate="tanh")``), used by the DiT FFN."""
    xf = x.astype(mx.float32)
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)  # sqrt(2/pi)
    out = 0.5 * xf * (1.0 + mx.tanh(inner))
    return out.astype(x.dtype)


def silu(x: mx.array) -> mx.array:
    """SiLU / swish (``x * sigmoid(x)``), used by the semantic-encoder FFN."""
    return x * mx.sigmoid(x)


def _sdpa_manual_fp32(q, k, v, attn_bias, scale):
    """Explicit fp32 scaled-dot-product-attention for the tight parity path.

    The batched ``QKᵀ`` / ``attn·V`` bmms are computed with ``mx.matmul`` but with
    fp32 operands and an fp32 softmax; at the tiny fixture ``L`` the cost is
    negligible. ``hp_matmul`` is deliberately NOT used here (its right operand is
    batched, violating the 2-D-right-operand contract — see ``hp_matmul`` caveat).

    NOTE: ``mx.matmul`` still rounds fp32 -> ~tf32 on Apple Silicon, so the bmm
    contributes ~1e-3 max-abs even here (the projections/qk-norm/rotary are exact
    to ~1e-6). A true-fp32 sum-reduction bmm closes that to ~1.7e-6 but costs
    ``O(B·H·L·S·D)`` peak memory — fine at fixture ``L`` but NOT at DiT seq lengths,
    so the runtime/T6 path should rely on the structural-equivalence + ~tf32
    cosine gate rather than a tight hp max-abs at large ``L``.
    """
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) * scale  # [B, H, L, S]
    if attn_bias is not None:
        scores = scores + attn_bias.astype(mx.float32)
    weights = mx.softmax(scores, axis=-1)
    return mx.matmul(weights, v)  # [B, H, L, D]


class MultiHeadAttention:
    """Multi-head attention with optional qk-norm + GPT-NeoX rotary.

    Mirrors ``dots_tts.modules.backbone.layers.MultiHeadAttention``. Separate
    q/k/v projections (bias=``qkv_bias``); ``o_proj`` ALWAYS has bias (upstream
    asymmetry). ``qk_norm`` applies ``norm_layer(head_dim)`` per head AFTER the
    head split and BEFORE rotary. Rotary (when ``rotary_bias``) uses ``pos_ids``.
    The bool mask (True=attend) becomes an additive bias (0 / -inf). SDPA scale is
    ``head_dim**-0.5`` applied once.

    ``hp=True`` -> fp32 projections (``hp_matmul``) + explicit fp32 SDPA (tight
    max-abs parity). ``hp=False`` -> fast projections + ``mx.fast.scaled_dot_product_
    attention`` (the runtime path; structurally identical, ~tf32 numerics).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        norm_layer: str = "LayerNorm",
        rotary_bias: bool = False,
        rotary_theta: float = 10000.0,
    ):
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.norm_layer = norm_layer
        self.rotary_bias = rotary_bias

        # Projections start as zero-weight placeholders; load_weights fills them.
        self.q_proj = Linear(mx.zeros((hidden_size, hidden_size)))
        self.k_proj = Linear(mx.zeros((hidden_size, hidden_size)))
        self.v_proj = Linear(mx.zeros((hidden_size, hidden_size)))
        self.o_proj = Linear(
            mx.zeros((hidden_size, hidden_size)), mx.zeros((hidden_size,))
        )

        if qk_norm:
            self.q_norm = self._make_norm(norm_layer, self.head_dim)
            self.k_norm = self._make_norm(norm_layer, self.head_dim)
        else:
            self.q_norm = _Identity()
            self.k_norm = _Identity()

        self.rotary = (
            RotaryEmbedding(self.head_dim, theta=rotary_theta) if rotary_bias else None
        )

    @staticmethod
    def _make_norm(norm_layer: str, dim: int):
        if norm_layer == "RMSNorm":
            return RMSNorm(dim)
        if norm_layer == "LayerNorm":
            # qk-norm LayerNorm in upstream IS affine (has weight+bias); not used by
            # the checkpoint (it's RMSNorm), so the affine-free LayerNorm suffices for
            # parity only if weights happen to be ones — kept explicit for clarity.
            return LayerNorm(dim)
        raise ValueError(f"unsupported norm_layer {norm_layer!r}")

    def load_weights(self, weights: dict) -> None:
        """Load the upstream state_dict (npz-key-safe ``w_<dotted>`` scheme).

        Keys: ``w_q_proj_weight``, ``w_k_proj_weight``, ``w_v_proj_weight``,
        ``w_o_proj_weight``, ``w_o_proj_bias``, and (if qk_norm) ``w_q_norm_weight``,
        ``w_k_norm_weight``. Weights are stored in torch ``[out, in]`` layout.
        """

        def g(name):
            return weights["w_" + name]

        self.q_proj.weight = g("q_proj_weight")
        self.k_proj.weight = g("k_proj_weight")
        self.v_proj.weight = g("v_proj_weight")
        self.o_proj.weight = g("o_proj_weight")
        self.o_proj.bias = g("o_proj_bias")
        if self.qkv_bias:
            self.q_proj.bias = g("q_proj_bias")
            self.k_proj.bias = g("k_proj_bias")
            self.v_proj.bias = g("v_proj_bias")
        if self.qk_norm:
            self.q_norm.weight = g("q_norm_weight")
            self.k_norm.weight = g("k_norm_weight")

    def _split_heads(self, x: mx.array) -> mx.array:
        """``[B, N, H*D] -> [B, H, N, D]``."""
        b, n, _ = x.shape
        x = x.reshape(b, n, self.num_heads, self.head_dim)
        return mx.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x: mx.array) -> mx.array:
        """``[B, H, N, D] -> [B, N, H*D]``."""
        b, h, n, d = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(b, n, h * d)

    def __call__(
        self,
        q: mx.array,
        k: mx.array | None = None,
        v: mx.array | None = None,
        *,
        mask: mx.array | None = None,
        pos_ids: mx.array | None = None,
        hp: bool = False,
    ) -> mx.array:
        if k is None:
            k = q
        if v is None:
            v = q

        qp = self.q_proj(q, hp=hp)
        kp = self.k_proj(k, hp=hp)
        vp = self.v_proj(v, hp=hp)

        qh = self._split_heads(qp)  # [B, H, N, D]
        kh = self._split_heads(kp)
        vh = self._split_heads(vp)

        qh = self.q_norm(qh)
        kh = self.k_norm(kh)

        if self.rotary is not None:
            b, _, length, _ = qh.shape
            _, _, slen, _ = kh.shape
            if pos_ids is None:
                pos_ids = mx.arange(length).astype(mx.float32)[None]
            if length == slen:
                emb = self.rotary(pos_ids)
                qh = apply_rotary_pos_emb(emb, qh)
                kh = apply_rotary_pos_emb(emb, kh)
            else:
                q_emb = self.rotary(mx.arange(length).astype(mx.float32))
                k_emb = self.rotary(mx.arange(slen).astype(mx.float32))
                qh = apply_rotary_pos_emb(q_emb, qh)
                kh = apply_rotary_pos_emb(k_emb, kh)

        # Bool mask (True=attend) -> additive bias [B, 1, L, S] broadcasting over heads.
        attn_bias = None
        if mask is not None:
            if mask.ndim == 2:  # [B, S] -> [B, 1, 1, S]
                m = mask[:, None, None, :]
            elif mask.ndim == 3:  # [B, L, S] -> [B, 1, L, S]
                m = mask[:, None, :, :]
            else:
                m = mask  # already [B, H, L, S]
            attn_bias = mx.where(
                m, mx.array(0.0, dtype=mx.float32), mx.array(-mx.inf, dtype=mx.float32)
            )

        if hp:
            out = _sdpa_manual_fp32(qh, kh, vh, attn_bias, self.scale)
        else:
            # mx.fast.sdpa requires the additive mask to promote to the output
            # (query) dtype; a fp32 bias does NOT promote to bf16, so cast it down.
            if attn_bias is not None and attn_bias.dtype != qh.dtype:
                attn_bias = attn_bias.astype(qh.dtype)
            out = mx.fast.scaled_dot_product_attention(
                qh, kh, vh, scale=self.scale, mask=attn_bias
            )

        out = self._merge_heads(out)
        return self.o_proj(out, hp=hp)

    def step(self, x_new, kv_cache, mask, *, hp: bool = False):
        """Streaming self-attention: new tokens attend to a growing KV cache.

        Projects ``x_new`` [1, n, hidden], appends its K/V to ``kv_cache``
        (mlx_lm KVCache; returns the full [1, H, T, D] tensors), and runs SDPA under the
        bool ``mask`` [n, T] (True=attend). No rotary / qk-norm (the patch encoder uses
        neither) — position enters only through ``mask``, so no positions are tracked.
        """
        assert self.rotary is None, "streaming step() does not support rotary"
        qh = self._split_heads(self.q_proj(x_new, hp=hp))  # [1, H, n, D]
        kh = self._split_heads(self.k_proj(x_new, hp=hp))
        vh = self._split_heads(self.v_proj(x_new, hp=hp))
        qh = self.q_norm(qh)  # _Identity for the encoder
        kh = self.k_norm(kh)
        k_full, v_full = kv_cache.update_and_fetch(kh, vh)  # [1, H, T, D]

        attn_bias = None
        if mask is not None:
            m = mask[None, None, :, :]  # [1, 1, n, T]
            attn_bias = mx.where(
                m, mx.array(0.0, dtype=mx.float32), mx.array(-mx.inf, dtype=mx.float32)
            )
        if hp:
            out = _sdpa_manual_fp32(qh, k_full, v_full, attn_bias, self.scale)
        else:
            if attn_bias is not None and attn_bias.dtype != qh.dtype:
                attn_bias = attn_bias.astype(qh.dtype)
            out = mx.fast.scaled_dot_product_attention(
                qh, k_full, v_full, scale=self.scale, mask=attn_bias
            )
        return self.o_proj(self._merge_heads(out), hp=hp)
