"""Pure-MLX flow-matching DiT acoustic head for dots.tts.

Imports ONLY mlx (+ the package's ``layers.py``) — never torch. Mirrors
``dots_tts.modules.backbone.dit`` for the shipped checkpoint (``mode="flow_matching"``,
with an optional ``duration_embedder`` for the meanflow (``mf``) checkpoint): an 18-layer adaLN-zero transformer that predicts a
denoising velocity for one VAE latent patch.

Convention (M1, matching ``layers.py``): plain classes holding ``mx.array`` weights
and reused ``layers.py`` primitives — NOT ``mlx.nn.Module``. Tensors flow as
``[B, L, C]``. The single forward (``DiT.__call__``) is the T6 gate; the euler-ODE
loop + CFG live in a later task. Weights are bound by ``loader.load_dit``.

Numerics: the per-block attention runs on the FAST (``hp=False``) path — fp32-stored
weights with MLX's reduced-precision (~tf32) matmul/SDPA, which is viable at DiT
sequence lengths (the fp32 ``hp`` path OOMs there). The adaLN modulation, layer
norms, time-embedding sinusoid, and FFN GELU are all computed in fp32. The gate is
cosine >= 0.9999 (tf32-robust over 18 stacked attention bmms) AND max-abs <= 2e-2.
"""

from __future__ import annotations

import math

import mlx.core as mx

from .layers import LayerNorm, Linear, Mlp, MultiHeadAttention, silu


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """adaLN modulation ``x * (1 + scale[:, None, :]) + shift[:, None, :]``.

    ``x`` is ``[B, L, C]``; ``shift`` / ``scale`` are ``[B, C]`` (broadcast over L).
    Matches the upstream ``modulate`` (``scale.unsqueeze(1)`` / ``shift.unsqueeze(1)``).
    """
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class TimestepEmbedder:
    """Sinusoidal timestep embedding -> 2-layer MLP (Linear -> SiLU -> Linear).

    ``timestep_embedding(t, 256)`` builds a 256-dim sinusoid with ``half=128``,
    ``freqs = exp(-ln(10000) * arange(128) / 128)``, ``args = t[:, None] * freqs[None]``,
    ``emb = cat([cos(args), sin(args)], -1)`` (**cos FIRST**). ``t`` is the raw
    timestep in [0, 1] (no 1000x scaling). The MLP maps ``256 -> 1024 -> 1024``.
    """

    def __init__(
        self,
        mlp_fc0: Linear,
        mlp_fc2: Linear,
        *,
        frequency_embedding_size: int = 256,
        max_period: float = 10000.0,
    ):
        self.mlp_fc0 = mlp_fc0
        self.mlp_fc2 = mlp_fc2
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    def _timestep_embedding(self, t: mx.array) -> mx.array:
        dim = self.frequency_embedding_size
        half = dim // 2
        tf = t.astype(mx.float32)
        freqs = mx.exp(
            -math.log(self.max_period) * mx.arange(0, half).astype(mx.float32) / half
        )  # [half]
        args = tf[:, None] * freqs[None, :]  # [B, half]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)  # [B, dim]
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        emb = self._timestep_embedding(t)  # [B, 256], fp32
        x = self.mlp_fc0(emb)
        x = silu(x)
        return self.mlp_fc2(x)  # [B, 1024]


class FinalLayer:
    """adaLN-zero output head: ``linear(modulate(norm(x), shift, scale))``.

    ``adaLN_modulation = SiLU -> Linear(1024 -> 2*1024)``; ``shift, scale`` are the
    2-way chunk. ``norm`` is an affine-free ``LayerNorm(1024, eps=1e-5)`` (no
    weights). ``linear`` maps ``1024 -> out_dim`` (128).
    """

    def __init__(self, adaLN_linear: Linear, linear: Linear, hidden_size: int = 1024):
        self.adaLN_linear = adaLN_linear
        self.norm = LayerNorm(hidden_size, eps=1e-5)
        self.linear = linear

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mod = self.adaLN_linear(silu(c))  # [B, 2*1024]
        shift, scale = mx.split(mod, 2, axis=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiTBlock:
    """adaLN-zero transformer block (modulation=True): gated attn + gated FFN.

    ``adaLN_modulation = SiLU -> Linear(1024 -> 6*1024)`` produces six ``[B, 1024]``
    chunks ``shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn``.
    ``norm1`` / ``norm2`` are affine-free ``LayerNorm(1024, eps=1e-5)`` (no weights):

        x = x + gate_attn[:, None, :] * attn(modulate(norm1(x), shift_attn, scale_attn))
        x = x + gate_ffn[:,  None, :] * ffn(modulate(norm2(x),  shift_ffn,  scale_ffn))

    The attention is the shared ``layers.MultiHeadAttention`` (qkv_bias=False,
    qk_norm=True/RMSNorm, rotary_bias=True, theta=10000), run on the fast (hp=False)
    path. The FFN is ``layers.Mlp`` with tanh-approx GELU.
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        ffn: Mlp,
        adaLN_linear: Linear,
        hidden_size: int = 1024,
    ):
        self.attn = attn
        self.ffn = ffn
        self.adaLN_linear = adaLN_linear
        self.norm1 = LayerNorm(hidden_size, eps=1e-5)
        self.norm2 = LayerNorm(hidden_size, eps=1e-5)

    def __call__(
        self,
        x: mx.array,
        c: mx.array,
        *,
        mask: mx.array | None = None,
        pos_ids: mx.array | None = None,
    ) -> mx.array:
        mod = self.adaLN_linear(silu(c))  # [B, 6*1024]
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = mx.split(
            mod, 6, axis=-1
        )
        x = x + gate_attn[:, None, :] * self.attn(
            modulate(self.norm1(x), shift_attn, scale_attn),
            mask=mask,
            pos_ids=pos_ids,
        )
        x = x + gate_ffn[:, None, :] * self.ffn(
            modulate(self.norm2(x), shift_ffn, scale_ffn)
        )
        return x


class DiT:
    """Flow-matching DiT: predicts a velocity for one VAE latent patch.

    ``__call__(x, timesteps, attn_mask, pos_ids, g_cond, duration)``:

        c = time_embedder(timesteps)        # [B, 1024]
        c = c + duration_embedder(duration) # optional: c = c + duration_embedder(duration)
                                            # (meanflow mode only; omitted for flow_matching)
        c = c + g_cond                      # global conditioning, [B, 1024]
        x = input_layer(x)                  # Linear(in_dim -> 1024)
        for block: x = block(x, c, mask=attn_mask, pos_ids=pos_ids)
        return output_layer(x, c)           # [B, L, out_dim]
    """

    def __init__(
        self,
        input_layer: Linear,
        time_embedder: TimestepEmbedder,
        blocks: list[DiTBlock],
        output_layer: FinalLayer,
        duration_embedder: TimestepEmbedder | None = None,
    ):
        self.input_layer = input_layer
        self.time_embedder = time_embedder
        self.duration_embedder = duration_embedder
        self.blocks = blocks
        self.output_layer = output_layer

    def __call__(
        self,
        x: mx.array,
        timesteps: mx.array,
        *,
        attn_mask: mx.array | None = None,
        pos_ids: mx.array | None = None,
        g_cond: mx.array | None = None,
        duration: mx.array | None = None,
    ) -> mx.array:
        c = self.time_embedder(timesteps)  # [B, 1024]
        if self.duration_embedder is not None and duration is not None:
            c = c + self.duration_embedder(duration)  # MeanFlow average-velocity signal
        if g_cond is not None:
            c = c + g_cond
        x = self.input_layer(x)  # [B, L, 1024]
        for block in self.blocks:
            x = block(x, c, mask=attn_mask, pos_ids=pos_ids)
        return self.output_layer(x, c)


def fm_solver_step(
    dit: DiT,
    coordinate_proj: Linear,
    t: mx.array,
    z: mx.array,
    *,
    input_sequence: mx.array,
    cfg_sequence: mx.array,
    attn_mask: mx.array | None,
    pos_ids: mx.array | None,
    g_cond: mx.array,
    guidance_scale: float,
) -> mx.array:
    """One flow-matching RHS eval ``v(t, z)`` with classifier-free guidance.

    Mirrors ``DotsTtsCore.fm_solver_step`` (upstream ``core.py``). Projects the noisy
    latent coordinate, splices it into both the conditional (real ``g_cond``) and the
    null (``cfg_sequence`` + zeroed ``g_cond``) branches, runs them batched through the
    DiT in one call, then extrapolates: ``v = vt_c + guidance_scale * (vt_c - vt_u)``.

    Args:
        dit: the flow-matching ``DiT`` velocity predictor.
        coordinate_proj: ``Linear(latent_dim, hidden_size)`` projecting ``z`` (the
            ``coordinate_proj.*`` weight from ``core.safetensors``).
        t: scalar timestep ``[1]`` (or any 1-D), broadcast to the 2-way batch.
        z: the noisy latent coordinate ``[1, patch_size, latent_dim]``.
        input_sequence: the conditional sequence ``[1, L, hidden_size]``; its last
            ``patch_size`` slots are overwritten by the projected ``z``.
        cfg_sequence: the null/history-only sequence ``[1, L, hidden_size]`` (same
            splice, but paired with a zeroed ``g_cond``).
        attn_mask / pos_ids: the ``[1, L, L]`` mask / ``[1, L]`` positions (broadcast
            across the batch by the DiT).
        g_cond: the (already-scaled) global conditioning ``[1, hidden_size]``.
        guidance_scale: CFG scale (the extrapolation coefficient).

    Returns:
        the velocity ``v`` for the latent block, ``[1, patch_size, latent_dim]``.
    """
    batch_size = input_sequence.shape[0]
    patch_size = z.shape[1]
    latent_start = input_sequence.shape[1] - patch_size

    # Project the noisy coordinate [1, patch_size, latent_dim] -> [1, patch_size, H].
    z_proj = coordinate_proj(z)
    rdtype = input_sequence.dtype
    z_proj = z_proj.astype(rdtype)

    # COND branch: real g_cond, projected z spliced into the latent slots.
    z_c = mx.concatenate([input_sequence[:, :latent_start], z_proj], axis=1)
    # UNCOND branch: null sequence + zeroed g_cond, SAME projected z.
    z_u = mx.concatenate([cfg_sequence[:, :latent_start], z_proj], axis=1)

    z_z = mx.concatenate([z_c, z_u], axis=0)  # [2, L, H]
    t_t = mx.broadcast_to(t.reshape(1), (2,))
    g = g_cond.astype(rdtype)
    g_cond_t = mx.concatenate([g, mx.zeros_like(g)], axis=0)  # [2, H]

    vt = dit(z_z, t_t, attn_mask=attn_mask, pos_ids=pos_ids, g_cond=g_cond_t)
    vt = vt[:, latent_start:]  # [2, patch_size, latent_dim]
    vt_c = vt[:batch_size]
    vt_u = vt[batch_size:]
    return vt_c + guidance_scale * (vt_c - vt_u)


class FlowSolver:
    """Flow-matching euler-ODE solver (replacing torchdiffeq) + CFG.

    Convention (sigma=0): ``t=0`` is pure noise ``x0``, ``t=1`` is data ``x1``;
    the DiT predicts the velocity ``u = x1 - x0``. The euler loop integrates
    ``t: 0 -> 1`` in ``num_steps`` fixed left-endpoint steps:

        dt = 1 / num_steps
        z = noise
        for k in 0 .. num_steps-1:  z = z + dt * v(k*dt, z)

    matching the upstream ``odeint(method="euler", t=[0, 1],
    options={"step_size": 1/num_steps})``. Each ``v`` is one CFG-guided
    ``fm_solver_step`` (a batched cond+uncond DiT call). The final ``z`` (at t=1)
    is the denoised normalized latent patch.

    Numerics (T9, measured): the euler ODE AMPLIFIES the per-step DiT bmm precision
    error. With INJECTED identical noise, the shippable tf32-bmm path drives the
    denoised patch to ~0.033 max-abs vs the torch oracle, while a true-fp32 reduction
    collapses it to ~1.4e-4 — i.e. the math is structurally correct and the residual
    is a tf32 numerical floor, NOT a bug. Direction (cosine) stays >= 0.999 and the
    downstream VAE decode is robust to this magnitude drift, so end-to-end parity is
    gated on BEHAVIOR (intelligible clone), not sample-exact PSNR.
    """

    def __init__(self, dit: DiT, coordinate_proj: Linear, *, latent_dim: int = 128):
        self.dit = dit
        self.coordinate_proj = coordinate_proj
        self.latent_dim = latent_dim

    def denoise(
        self,
        *,
        input_sequence: mx.array,
        cfg_sequence: mx.array,
        attn_mask: mx.array | None,
        pos_ids: mx.array | None,
        g_cond: mx.array,
        guidance_scale: float = 3.0,
        num_steps: int = 10,
        patch_size: int = 4,
        noise: mx.array | None = None,
    ) -> mx.array:
        """Denoise one latent patch from ``noise`` (or a fresh draw) via euler + CFG.

        Args:
            input_sequence / cfg_sequence: cond / null sequences ``[1, L, hidden]``.
            attn_mask / pos_ids: the ``[1, L, L]`` mask / ``[1, L]`` positions.
            g_cond: global conditioning ``[1, hidden]`` (already scaled upstream).
            guidance_scale: CFG scale.
            num_steps: euler step count (NFE).
            patch_size: number of latent slots to denoise.
            noise: the injected initial coordinate ``[1, patch_size, latent_dim]``;
                ``None`` draws a fresh ``mx.random.normal``.

        Returns:
            the denoised latent patch ``[1, patch_size, latent_dim]`` (at t=1).
        """
        rdtype = input_sequence.dtype
        if noise is None:
            z = mx.random.normal((1, patch_size, self.latent_dim)).astype(rdtype)
        else:
            z = noise.astype(rdtype)

        dt = 1.0 / num_steps
        for k in range(num_steps):
            t = mx.array([k * dt], dtype=rdtype)
            v = fm_solver_step(
                self.dit,
                self.coordinate_proj,
                t,
                z,
                input_sequence=input_sequence,
                cfg_sequence=cfg_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                g_cond=g_cond,
                guidance_scale=guidance_scale,
            )
            z = z + dt * v
        return z

    def meanflow_sample(
        self,
        *,
        input_sequence: mx.array,
        attn_mask: mx.array | None,
        pos_ids: mx.array | None,
        g_cond: mx.array,
        num_steps: int = 4,
        patch_size: int = 4,
        noise: mx.array,
    ) -> mx.array:
        """Distilled MeanFlow few-step sampler (NFE = ``num_steps``, NO CFG).

        Mirrors upstream ``_meanflow_step_fm`` / ``meanflow_solver_step``: the DiT
        predicts the AVERAGE velocity over the interval ``[t, t+dt]`` (it receives both
        ``t`` and ``dt`` as ``duration``), so a single forward per step advances the full
        ``dt``. No classifier-free guidance — one conditional branch with the real
        ``g_cond`` (guidance is fused into the distilled student). Uniform schedule on
        ``[0, 1]``: ``t_k = k/nfe``, ``dt = 1/nfe``.

        Args:
            input_sequence: the conditional sequence ``[1, L, hidden]``; its last
                ``patch_size`` slots are overwritten by the projected ``z`` each step.
            attn_mask / pos_ids: the ``[1, L, L]`` mask / ``[1, L]`` positions.
            g_cond: global conditioning ``[1, hidden]`` (already scaled upstream).
            num_steps: NFE (number of DiT forwards). Distilled to work at 2-4.
            patch_size: number of latent slots to denoise.
            noise: the injected initial coordinate ``[1, patch_size, latent_dim]``.

        Returns:
            the denoised latent patch ``[1, patch_size, latent_dim]`` (at t=1).
        """
        rdtype = input_sequence.dtype
        z = noise.astype(rdtype)
        latent_start = input_sequence.shape[1] - patch_size
        g = g_cond.astype(rdtype)
        inv = 1.0 / num_steps
        for k in range(num_steps):
            t = mx.array([k * inv], dtype=rdtype)
            dt = mx.array([inv], dtype=rdtype)
            z_proj = self.coordinate_proj(z).astype(rdtype)
            seq = mx.concatenate([input_sequence[:, :latent_start], z_proj], axis=1)
            v = self.dit(
                seq, t, attn_mask=attn_mask, pos_ids=pos_ids, g_cond=g, duration=dt
            )
            v = v[:, latent_start:]  # [1, patch_size, latent_dim]
            z = z + v * inv
        return z
