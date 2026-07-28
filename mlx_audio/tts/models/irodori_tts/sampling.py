from __future__ import annotations

from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .model import IrodoriDiT

KVCache = List[Tuple[mx.array, mx.array]]


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------


def _concat_kv_caches(*caches: KVCache) -> KVCache:
    """Concatenate KV caches from multiple conditions along the batch axis."""
    result: KVCache = []
    for i in range(len(caches[0])):
        k = mx.concatenate([c[i][0] for c in caches], axis=0)
        v = mx.concatenate([c[i][1] for c in caches], axis=0)
        result.append((k, v))
    return result


def _scale_kv_cache(
    cache: KVCache,
    scale: float,
    speaker_only: bool = True,
    max_layers: Optional[int] = None,
) -> KVCache:
    """Return a new KV cache with speaker KVs scaled (immutable, MLX-friendly)."""
    n = len(cache) if max_layers is None else min(max_layers, len(cache))
    result: KVCache = []
    for i, (k, v) in enumerate(cache):
        if i < n:
            result.append((k * scale, v * scale))
        else:
            result.append((k, v))
    return result


# ---------------------------------------------------------------------------
# Score rescaling (optional post-processing of velocity prediction)
# ---------------------------------------------------------------------------


def _temporal_score_rescale(
    v_pred: mx.array,
    x_t: mx.array,
    t: float,
    rescale_k: float,
    rescale_sigma: float,
) -> mx.array:
    """Temporal score rescaling from https://arxiv.org/pdf/2510.01184."""
    if t >= 1.0:
        return v_pred
    one_minus_t = 1.0 - t
    snr = (one_minus_t**2) / (t**2)
    sigma_sq = rescale_sigma**2
    ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / rescale_k + 1.0)
    return (ratio * (one_minus_t * v_pred + x_t) - x_t) / one_minus_t


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------


def sample_euler_cfg(
    model: IrodoriDiT,
    text_input_ids: mx.array,
    text_mask: mx.array,
    ref_latent: Optional[mx.array],
    ref_mask: Optional[mx.array],
    latent_dim: int,
    rng_seed: int = 0,
    sequence_length: int = 750,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_scale_caption: float = 3.0,
    cfg_guidance_mode: str = "independent",
    cfg_scale: Optional[float] = None,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    truncation_factor: Optional[float] = None,
    rescale_k: Optional[float] = None,
    rescale_sigma: Optional[float] = None,
    context_kv_cache: bool = True,
    speaker_kv_scale: Optional[float] = None,
    speaker_kv_min_t: Optional[float] = None,
    speaker_kv_max_layers: Optional[int] = None,
    caption_input_ids: Optional[mx.array] = None,
    caption_mask: Optional[mx.array] = None,
    t_schedule_mode: str = "linear",
    sway_coeff: float = -1.0,
    **_ignored,
) -> mx.array:
    """
    Euler sampler for Rectified Flow ODE with Classifier-Free Guidance.

    Supports three CFG modes:
      independent : text and context guidance computed in a single 3x-batch forward pass.
      joint       : single combined unconditional (both text and context zeroed).
      alternating : text-uncond and context-uncond alternate each step.

    Returns latent of shape (batch, sequence_length, latent_dim).
    """
    # Backward-compat: single cfg_scale overrides both
    if cfg_scale is not None:
        cfg_scale_text = float(cfg_scale)
        cfg_scale_speaker = float(cfg_scale)
        cfg_scale_caption = float(cfg_scale)

    use_spk = model.cfg.use_speaker_condition_resolved
    use_cap = model.cfg.use_caption_condition
    is_dual = use_spk and use_cap

    # Resolve context CFG scale for single-context models
    if not is_dual:
        cfg_scale_context = cfg_scale_caption if use_cap else cfg_scale_speaker
    else:
        cfg_scale_context = cfg_scale_speaker  # unused in dual, kept for compat

    cfg_guidance_mode = cfg_guidance_mode.strip().lower()
    if cfg_guidance_mode not in {"independent", "joint", "alternating"}:
        raise ValueError(
            f"Unknown cfg_guidance_mode={cfg_guidance_mode!r}. "
            "Expected: independent | joint | alternating"
        )

    batch_size = text_input_ids.shape[0]
    has_text_cfg = cfg_scale_text > 0
    has_speaker_cfg = cfg_scale_speaker > 0 and use_spk
    has_caption_cfg = cfg_scale_caption > 0 and use_cap
    # For single-context backward compat
    if not is_dual:
        has_context_cfg = cfg_scale_context > 0
    else:
        has_context_cfg = False  # not used in dual path

    # ---- encode all conditions ----
    (
        text_state_cond,
        text_mask_cond,
        spk_state_full,
        spk_mask_full,
        cap_state_full,
        cap_mask_full,
    ) = model.encode_conditions_full(
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        ref_latent=ref_latent,
        ref_mask=ref_mask,
        caption_input_ids=caption_input_ids,
        caption_mask=caption_mask,
    )

    # For single-context models, alias context into "speaker" slot (backward compat)
    if not is_dual and use_cap:
        speaker_state_cond = cap_state_full
        speaker_mask_cond = cap_mask_full
        caption_state_cond = None
        caption_mask_cond = None
    else:
        speaker_state_cond = spk_state_full
        speaker_mask_cond = spk_mask_full
        caption_state_cond = cap_state_full
        caption_mask_cond = cap_mask_full

    mx.eval(text_state_cond)
    if speaker_state_cond is not None:
        mx.eval(speaker_state_cond)
    if caption_state_cond is not None:
        mx.eval(caption_state_cond)

    # unconditioned states
    text_state_uncond = mx.zeros_like(text_state_cond)
    text_mask_uncond = mx.zeros_like(text_mask_cond)
    speaker_state_uncond = (
        mx.zeros_like(speaker_state_cond) if speaker_state_cond is not None else None
    )
    speaker_mask_uncond = (
        mx.zeros_like(speaker_mask_cond) if speaker_mask_cond is not None else None
    )
    caption_state_uncond = (
        mx.zeros_like(caption_state_cond) if caption_state_cond is not None else None
    )
    caption_mask_uncond = (
        mx.zeros_like(caption_mask_cond) if caption_mask_cond is not None else None
    )

    # ---- build KV caches ----
    use_kv_cache = context_kv_cache or (speaker_kv_scale is not None)

    kv_text_cond: Optional[KVCache] = None
    kv_speaker_cond: Optional[KVCache] = None
    kv_caption_cond: Optional[KVCache] = None
    kv_text_cfg: Optional[KVCache] = None
    kv_speaker_cfg: Optional[KVCache] = None
    kv_caption_cfg: Optional[KVCache] = None
    # extra caches for joint/alternating
    kv_text_uncond_joint: Optional[KVCache] = None
    kv_speaker_uncond_joint: Optional[KVCache] = None
    kv_caption_uncond_joint: Optional[KVCache] = None
    kv_text_uncond_alt: Optional[KVCache] = None
    kv_speaker_uncond_alt: Optional[KVCache] = None
    kv_caption_uncond_alt: Optional[KVCache] = None

    if use_kv_cache:
        kv_text_cond, kv_speaker_cond, kv_caption_cond = model.build_kv_cache(
            text_state_cond, speaker_state_cond, caption_state_cond
        )
        if speaker_kv_scale is not None and kv_speaker_cond is not None:
            kv_speaker_cond = _scale_kv_cache(
                kv_speaker_cond, speaker_kv_scale, max_layers=speaker_kv_max_layers
            )

        if cfg_guidance_mode == "independent":
            if is_dual:
                # Dual mode: up to 4x batch [cond, text-uncond, spk-uncond, cap-uncond]
                active = [True]  # always include cond
                if has_text_cfg:
                    active.append("text")
                if has_speaker_cfg:
                    active.append("speaker")
                if has_caption_cfg:
                    active.append("caption")
                # Build concatenated caches with per-slot uncond entries
                n_bundles = 1 + sum([has_text_cfg, has_speaker_cfg, has_caption_cfg])
                kv_text_parts = [kv_text_cond] * n_bundles
                kv_spk_parts = [kv_speaker_cond] * n_bundles
                kv_cap_parts = [kv_caption_cond] * n_bundles
                if n_bundles > 1:
                    kv_text_cfg = _concat_kv_caches(*kv_text_parts)
                    kv_speaker_cfg = _concat_kv_caches(*kv_spk_parts)
                    kv_caption_cfg = _concat_kv_caches(*kv_cap_parts)
            else:
                # Single-context backward compat
                if has_text_cfg and has_context_cfg:
                    kv_text_cfg = _concat_kv_caches(
                        kv_text_cond, kv_text_cond, kv_text_cond
                    )
                    if kv_speaker_cond is not None:
                        kv_speaker_cfg = _concat_kv_caches(
                            kv_speaker_cond, kv_speaker_cond, kv_speaker_cond
                        )
                elif has_text_cfg:
                    kv_text_cfg = _concat_kv_caches(kv_text_cond, kv_text_cond)
                    if kv_speaker_cond is not None:
                        kv_speaker_cfg = _concat_kv_caches(
                            kv_speaker_cond, kv_speaker_cond
                        )
                elif has_context_cfg:
                    kv_text_cfg = _concat_kv_caches(kv_text_cond, kv_text_cond)
                    if kv_speaker_cond is not None:
                        kv_speaker_cfg = _concat_kv_caches(
                            kv_speaker_cond, kv_speaker_cond
                        )

        elif cfg_guidance_mode == "joint":
            if is_dual:
                if has_text_cfg or has_speaker_cfg or has_caption_cfg:
                    (
                        kv_text_uncond_joint,
                        kv_speaker_uncond_joint,
                        kv_caption_uncond_joint,
                    ) = model.build_kv_cache(
                        text_state_uncond, speaker_state_uncond, caption_state_uncond
                    )
            else:
                if has_text_cfg or has_context_cfg:
                    kv_text_uncond_joint, kv_speaker_uncond_joint, _ = (
                        model.build_kv_cache(text_state_uncond, speaker_state_uncond)
                    )

        elif cfg_guidance_mode == "alternating":
            if not is_dual:
                if has_text_cfg:
                    kv_text_uncond_alt, _, _ = model.build_kv_cache(
                        text_state_uncond, speaker_state_cond
                    )
                if has_context_cfg:
                    _, kv_speaker_uncond_alt, _ = model.build_kv_cache(
                        text_state_cond, speaker_state_uncond
                    )
                    if (
                        speaker_kv_scale is not None
                        and kv_speaker_uncond_alt is not None
                    ):
                        kv_speaker_uncond_alt = _scale_kv_cache(
                            kv_speaker_uncond_alt,
                            speaker_kv_scale,
                            max_layers=speaker_kv_max_layers,
                        )

        mx.eval(kv_text_cond)
        if kv_speaker_cond is not None:
            mx.eval(kv_speaker_cond)
        if kv_caption_cond is not None:
            mx.eval(kv_caption_cond)

    # ---- initial noise ----
    mx.random.seed(rng_seed)
    init_scale = 0.999
    x_t = mx.random.normal((batch_size, sequence_length, latent_dim))
    if truncation_factor is not None:
        x_t = x_t * float(truncation_factor)

    t_schedule_np = np.linspace(1.0 * init_scale, 0.0, num_steps + 1, dtype=np.float32)

    # Sway Sampling (v3)
    t_schedule_mode_norm = str(t_schedule_mode).strip().lower()
    if t_schedule_mode_norm == "sway":
        sway_coeff_value = float(sway_coeff)
        u = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)
        u = u + sway_coeff_value * (np.cos(0.5 * np.pi * u) + u - 1.0)
        u = np.clip(u, 0.0, 1.0)
        t_schedule_np = (1.0 - u) * init_scale

    t_schedule = t_schedule_np

    speaker_kv_active = speaker_kv_scale is not None

    # ---- Euler steps ----
    for i in range(num_steps):
        t = float(t_schedule[i])
        t_next = float(t_schedule[i + 1])
        t_arr = mx.full((batch_size,), t, dtype=mx.float32)
        use_cfg = (has_text_cfg or has_speaker_cfg) and (cfg_min_t <= t <= cfg_max_t)

        if use_cfg:
            if cfg_guidance_mode == "independent":
                if is_dual:
                    # Dual mode: build bundle list [cond, text-uncond?, spk-uncond?, cap-uncond?]
                    bundles_x = [x_t]
                    bundles_t = [text_state_cond]
                    bundles_tm = [text_mask_cond]
                    bundles_s = [speaker_state_cond]
                    bundles_sm = [speaker_mask_cond]
                    bundles_c = [caption_state_cond]
                    bundles_cm = [caption_mask_cond]
                    if has_text_cfg:
                        bundles_x.append(x_t)
                        bundles_t.append(text_state_uncond)
                        bundles_tm.append(text_mask_uncond)
                        bundles_s.append(speaker_state_cond)
                        bundles_sm.append(speaker_mask_cond)
                        bundles_c.append(caption_state_cond)
                        bundles_cm.append(caption_mask_cond)
                    if has_speaker_cfg:
                        bundles_x.append(x_t)
                        bundles_t.append(text_state_cond)
                        bundles_tm.append(text_mask_cond)
                        bundles_s.append(speaker_state_uncond)
                        bundles_sm.append(speaker_mask_uncond)
                        bundles_c.append(caption_state_cond)
                        bundles_cm.append(caption_mask_cond)
                    if has_caption_cfg:
                        bundles_x.append(x_t)
                        bundles_t.append(text_state_cond)
                        bundles_tm.append(text_mask_cond)
                        bundles_s.append(speaker_state_cond)
                        bundles_sm.append(speaker_mask_cond)
                        bundles_c.append(caption_state_uncond)
                        bundles_cm.append(caption_mask_uncond)
                    n_b = len(bundles_x)
                    x_cfg = mx.concatenate(bundles_x, axis=0)
                    t_cfg = mx.full((batch_size * n_b,), t, dtype=mx.float32)
                    v_out = model.forward_with_conditions(
                        x_t=x_cfg,
                        t=t_cfg,
                        text_state=mx.concatenate(bundles_t, axis=0),
                        text_mask=mx.concatenate(bundles_tm, axis=0),
                        speaker_state=mx.concatenate(bundles_s, axis=0),
                        speaker_mask=mx.concatenate(bundles_sm, axis=0),
                        kv_text=kv_text_cfg,
                        kv_speaker=kv_speaker_cfg,
                        caption_state=mx.concatenate(bundles_c, axis=0),
                        caption_mask=mx.concatenate(bundles_cm, axis=0),
                        kv_caption=kv_caption_cfg,
                    )
                    splits = mx.split(v_out, n_b, axis=0)
                    v_cond = splits[0]
                    v_pred = v_cond
                    idx = 1
                    if has_text_cfg:
                        v_pred = v_pred + cfg_scale_text * (v_cond - splits[idx])
                        idx += 1
                    if has_speaker_cfg:
                        v_pred = v_pred + cfg_scale_speaker * (v_cond - splits[idx])
                        idx += 1
                    if has_caption_cfg:
                        v_pred = v_pred + cfg_scale_caption * (v_cond - splits[idx])

                elif has_text_cfg and has_context_cfg:
                    # 3x batch: [cond, text-uncond, context-uncond]
                    x_cfg = mx.concatenate([x_t, x_t, x_t], axis=0)
                    t_cfg = mx.full((batch_size * 3,), t, dtype=mx.float32)
                    text_mask_cfg = mx.concatenate(
                        [text_mask_cond, text_mask_uncond, text_mask_cond], axis=0
                    )
                    speaker_mask_cfg = mx.concatenate(
                        [speaker_mask_cond, speaker_mask_cond, speaker_mask_uncond],
                        axis=0,
                    )
                    v_out = model.forward_with_conditions(
                        x_t=x_cfg,
                        t=t_cfg,
                        text_state=mx.concatenate(
                            [text_state_cond, text_state_uncond, text_state_cond],
                            axis=0,
                        ),
                        text_mask=text_mask_cfg,
                        speaker_state=mx.concatenate(
                            [
                                speaker_state_cond,
                                speaker_state_cond,
                                speaker_state_uncond,
                            ],
                            axis=0,
                        ),
                        speaker_mask=speaker_mask_cfg,
                        kv_text=kv_text_cfg,
                        kv_speaker=kv_speaker_cfg,
                    )
                    v_cond, v_uncond_text, v_uncond_speaker = mx.split(v_out, 3, axis=0)
                    v_pred = (
                        v_cond
                        + cfg_scale_text * (v_cond - v_uncond_text)
                        + cfg_scale_context * (v_cond - v_uncond_speaker)
                    )

                elif has_text_cfg:
                    x_cfg = mx.concatenate([x_t, x_t], axis=0)
                    t_cfg = mx.full((batch_size * 2,), t, dtype=mx.float32)
                    v_out = model.forward_with_conditions(
                        x_t=x_cfg,
                        t=t_cfg,
                        text_state=mx.concatenate(
                            [text_state_cond, text_state_uncond], axis=0
                        ),
                        text_mask=mx.concatenate(
                            [text_mask_cond, text_mask_uncond], axis=0
                        ),
                        speaker_state=mx.concatenate(
                            [speaker_state_cond, speaker_state_cond], axis=0
                        ),
                        speaker_mask=mx.concatenate(
                            [speaker_mask_cond, speaker_mask_cond], axis=0
                        ),
                        kv_text=kv_text_cfg,
                        kv_speaker=kv_speaker_cfg,
                    )
                    v_cond, v_uncond_text = mx.split(v_out, 2, axis=0)
                    v_pred = v_cond + cfg_scale_text * (v_cond - v_uncond_text)

                else:  # has_context_cfg only
                    x_cfg = mx.concatenate([x_t, x_t], axis=0)
                    t_cfg = mx.full((batch_size * 2,), t, dtype=mx.float32)
                    v_out = model.forward_with_conditions(
                        x_t=x_cfg,
                        t=t_cfg,
                        text_state=mx.concatenate(
                            [text_state_cond, text_state_cond], axis=0
                        ),
                        text_mask=mx.concatenate(
                            [text_mask_cond, text_mask_cond], axis=0
                        ),
                        speaker_state=mx.concatenate(
                            [speaker_state_cond, speaker_state_uncond], axis=0
                        ),
                        speaker_mask=mx.concatenate(
                            [speaker_mask_cond, speaker_mask_uncond], axis=0
                        ),
                        kv_text=kv_text_cfg,
                        kv_speaker=kv_speaker_cfg,
                    )
                    v_cond, v_uncond_speaker = mx.split(v_out, 2, axis=0)
                    v_pred = v_cond + cfg_scale_context * (v_cond - v_uncond_speaker)

            elif cfg_guidance_mode == "joint":
                has_any_cfg = (
                    (has_text_cfg or has_speaker_cfg or has_caption_cfg)
                    if is_dual
                    else (has_text_cfg or has_context_cfg)
                )
                if is_dual:
                    all_scales = [
                        s
                        for s, a in [
                            (cfg_scale_text, has_text_cfg),
                            (cfg_scale_speaker, has_speaker_cfg),
                            (cfg_scale_caption, has_caption_cfg),
                        ]
                        if a
                    ]
                    joint_scale = all_scales[0] if all_scales else cfg_scale_text
                elif has_text_cfg and has_context_cfg:
                    if abs(cfg_scale_text - cfg_scale_context) > 1e-6:
                        raise ValueError(
                            "cfg_guidance_mode='joint' requires equal text/speaker scales. "
                            "Use cfg_scale or set both to the same value."
                        )
                    joint_scale = cfg_scale_text
                else:
                    joint_scale = cfg_scale_text if has_text_cfg else cfg_scale_context

                v_cond = model.forward_with_conditions(
                    x_t=x_t,
                    t=t_arr,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    kv_text=kv_text_cond,
                    kv_speaker=kv_speaker_cond,
                    caption_state=caption_state_cond,
                    caption_mask=caption_mask_cond,
                    kv_caption=kv_caption_cond,
                )
                v_uncond = model.forward_with_conditions(
                    x_t=x_t,
                    t=t_arr,
                    text_state=text_state_uncond,
                    text_mask=text_mask_uncond,
                    speaker_state=speaker_state_uncond,
                    speaker_mask=speaker_mask_uncond,
                    kv_text=kv_text_uncond_joint,
                    kv_speaker=kv_speaker_uncond_joint,
                    caption_state=caption_state_uncond,
                    caption_mask=caption_mask_uncond,
                    kv_caption=kv_caption_uncond_joint,
                )
                v_pred = v_cond + joint_scale * (v_cond - v_uncond)

            else:  # alternating (single-context only)
                v_cond = model.forward_with_conditions(
                    x_t=x_t,
                    t=t_arr,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    kv_text=kv_text_cond,
                    kv_speaker=kv_speaker_cond,
                )
                use_text_uncond = (has_text_cfg and has_context_cfg and i % 2 == 0) or (
                    has_text_cfg and not has_context_cfg
                )
                if use_text_uncond:
                    v_uncond = model.forward_with_conditions(
                        x_t=x_t,
                        t=t_arr,
                        text_state=text_state_uncond,
                        text_mask=text_mask_uncond,
                        speaker_state=speaker_state_cond,
                        speaker_mask=speaker_mask_cond,
                        kv_text=kv_text_uncond_alt,
                        kv_speaker=kv_speaker_cond,
                    )
                    v_pred = v_cond + cfg_scale_text * (v_cond - v_uncond)
                else:
                    v_uncond = model.forward_with_conditions(
                        x_t=x_t,
                        t=t_arr,
                        text_state=text_state_cond,
                        text_mask=text_mask_cond,
                        speaker_state=speaker_state_uncond,
                        speaker_mask=speaker_mask_uncond,
                        kv_text=kv_text_cond,
                        kv_speaker=kv_speaker_uncond_alt,
                    )
                    v_pred = v_cond + cfg_scale_context * (v_cond - v_uncond)

        else:
            # no CFG this step
            v_pred = model.forward_with_conditions(
                x_t=x_t,
                t=t_arr,
                text_state=text_state_cond,
                text_mask=text_mask_cond,
                speaker_state=speaker_state_cond,
                speaker_mask=speaker_mask_cond,
                kv_text=kv_text_cond,
                kv_speaker=kv_speaker_cond,
                caption_state=caption_state_cond,
                caption_mask=caption_mask_cond,
                kv_caption=kv_caption_cond,
            )

        # optional temporal score rescaling
        if rescale_k is not None and rescale_sigma is not None:
            v_pred = _temporal_score_rescale(v_pred, x_t, t, rescale_k, rescale_sigma)

        # speaker KV scale rollback at threshold
        if (
            speaker_kv_active
            and speaker_kv_min_t is not None
            and t_next < speaker_kv_min_t <= t
            and kv_speaker_cond is not None
        ):
            inv = 1.0 / speaker_kv_scale
            kv_speaker_cond = _scale_kv_cache(
                kv_speaker_cond, inv, max_layers=speaker_kv_max_layers
            )
            if kv_speaker_cfg is not None:
                n_rep = 3 if (not is_dual and has_text_cfg and has_context_cfg) else 2
                kv_speaker_cfg = _concat_kv_caches(*([kv_speaker_cond] * n_rep))
            if kv_speaker_uncond_alt is not None:
                kv_speaker_uncond_alt = _scale_kv_cache(
                    kv_speaker_uncond_alt, inv, max_layers=speaker_kv_max_layers
                )
            speaker_kv_active = False

        # Euler update: x_{t-dt} = x_t + v * (t_next - t)
        x_t = x_t + v_pred * (t_next - t)
        mx.eval(x_t)

    return x_t
