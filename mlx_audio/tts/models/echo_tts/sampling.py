from __future__ import annotations

from typing import List, Tuple

import mlx.core as mx
import numpy as np

from .model import EchoDiT

KVCache = List[Tuple[mx.array, mx.array]]
DEFAULT_TRUNCATION_FACTOR = 0.96


def _concat_kv_caches(*caches: KVCache) -> KVCache:
    num_layers = len(caches[0])
    result: KVCache = []
    for i in range(num_layers):
        k = mx.concatenate([c[i][0] for c in caches], axis=0)
        v = mx.concatenate([c[i][1] for c in caches], axis=0)
        result.append((k, v))
    return result


def _multiply_kv_cache(
    cache: KVCache, scale: float, max_layers: int | None = None
) -> KVCache:
    num_layers = len(cache) if max_layers is None else min(max_layers, len(cache))
    result: KVCache = []
    for i, (k, v) in enumerate(cache):
        if i < num_layers:
            result.append((k * scale, v * scale))
        else:
            result.append((k, v))
    return result


def _temporal_score_rescale(
    v_pred: mx.array,
    x_t: mx.array,
    t: float,
    rescale_k: float,
    rescale_sigma: float,
) -> mx.array:
    if t < 1.0:
        snr = (1.0 - t) ** 2 / (t**2)
        ratio = (snr * rescale_sigma**2 + 1.0) / (
            snr * rescale_sigma**2 / rescale_k + 1.0
        )
        return (1.0 / (1.0 - t)) * (ratio * ((1.0 - t) * v_pred + x_t) - x_t)
    return v_pred


def sample_euler_cfg_independent_guidances(
    model: EchoDiT,
    speaker_latent: mx.array,
    speaker_mask: mx.array,
    text_input_ids: mx.array,
    text_mask: mx.array,
    rng_seed: int,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 8.0,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
    sequence_length: int = 640,
) -> mx.array:
    init_scale = 0.999
    batch_size = text_input_ids.shape[0]

    mx.random.seed(rng_seed)
    t_schedule = np.linspace(1.0 * init_scale, 0.0, num_steps + 1, dtype=np.float32)

    text_mask_uncond = mx.zeros_like(text_mask)
    speaker_mask_uncond = mx.zeros_like(speaker_mask)

    kv_text_cond = model.get_kv_cache_text(text_input_ids, text_mask)
    kv_speaker_cond = model.get_kv_cache_speaker(speaker_latent)

    if speaker_kv_scale is not None:
        kv_speaker_cond = _multiply_kv_cache(
            kv_speaker_cond, speaker_kv_scale, speaker_kv_max_layers
        )

    kv_text_full = _concat_kv_caches(kv_text_cond, kv_text_cond, kv_text_cond)
    kv_speaker_full = _concat_kv_caches(
        kv_speaker_cond, kv_speaker_cond, kv_speaker_cond
    )

    full_text_mask = mx.concatenate([text_mask, text_mask_uncond, text_mask], axis=0)
    full_speaker_mask = mx.concatenate(
        [speaker_mask, speaker_mask, speaker_mask_uncond], axis=0
    )

    latent_size = model.out_proj.weight.shape[0]
    x_t = mx.random.normal((batch_size, sequence_length, latent_size))
    trunc = (
        DEFAULT_TRUNCATION_FACTOR if truncation_factor is None else truncation_factor
    )
    x_t = x_t * trunc

    for i in range(num_steps):
        t = float(t_schedule[i])
        t_next = float(t_schedule[i + 1])
        has_cfg = cfg_min_t <= t <= cfg_max_t

        if has_cfg:
            x_t_full = mx.concatenate([x_t, x_t, x_t], axis=0)
            t_full = mx.full((batch_size * 3,), t, dtype=mx.float32)
            out = model(
                x=x_t_full,
                t=t_full,
                text_mask=full_text_mask,
                speaker_mask=full_speaker_mask,
                kv_cache_text=kv_text_full,
                kv_cache_speaker=kv_speaker_full,
            )
            v_cond, v_uncond_text, v_uncond_speaker = mx.split(out, 3, axis=0)
            v_pred = (
                v_cond
                + cfg_scale_text * (v_cond - v_uncond_text)
                + cfg_scale_speaker * (v_cond - v_uncond_speaker)
            )
        else:
            t_cond = mx.full((batch_size,), t, dtype=mx.float32)
            v_pred = model(
                x=x_t,
                t=t_cond,
                text_mask=text_mask,
                speaker_mask=speaker_mask,
                kv_cache_text=kv_text_cond,
                kv_cache_speaker=kv_speaker_cond,
            )

        if rescale_k is not None and rescale_sigma is not None:
            v_pred = _temporal_score_rescale(v_pred, x_t, t, rescale_k, rescale_sigma)

        if (
            speaker_kv_scale is not None
            and speaker_kv_min_t is not None
            and t_next < speaker_kv_min_t <= t
        ):
            kv_speaker_cond = _multiply_kv_cache(
                kv_speaker_cond,
                1.0 / speaker_kv_scale,
                speaker_kv_max_layers,
            )
            kv_speaker_full = _concat_kv_caches(
                kv_speaker_cond,
                kv_speaker_cond,
                kv_speaker_cond,
            )

        x_t = x_t + v_pred * (t_next - t)

    return x_t


def sample_blockwise_euler_cfg_independent_guidances(
    model: EchoDiT,
    speaker_latent: mx.array,
    speaker_mask: mx.array,
    text_input_ids: mx.array,
    text_mask: mx.array,
    rng_seed: int,
    block_sizes: List[int],
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 8.0,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
    continuation_latent: mx.array | None = None,
) -> mx.array:
    init_scale = 0.999
    batch_size = text_input_ids.shape[0]
    latent_size = model.out_proj.weight.shape[0]

    mx.random.seed(rng_seed)
    t_schedule = np.linspace(1.0 * init_scale, 0.0, num_steps + 1, dtype=np.float32)

    text_mask_uncond = mx.zeros_like(text_mask)
    speaker_mask_uncond = mx.zeros_like(speaker_mask)

    kv_text_cond = model.get_kv_cache_text(text_input_ids, text_mask)
    kv_speaker_cond = model.get_kv_cache_speaker(speaker_latent)

    kv_text_full = _concat_kv_caches(kv_text_cond, kv_text_cond, kv_text_cond)
    kv_speaker_full = _concat_kv_caches(
        kv_speaker_cond, kv_speaker_cond, kv_speaker_cond
    )

    full_text_mask = mx.concatenate([text_mask, text_mask_uncond, text_mask], axis=0)
    full_speaker_mask = mx.concatenate(
        [speaker_mask, speaker_mask, speaker_mask_uncond], axis=0
    )

    generated_chunks: List[mx.array] = []
    start_pos = 0

    if continuation_latent is not None:
        generated_chunks.append(continuation_latent)
        start_pos = continuation_latent.shape[1]

    for block_size in block_sizes:
        if speaker_kv_scale is not None:
            kv_speaker_cond = _multiply_kv_cache(
                kv_speaker_cond, speaker_kv_scale, speaker_kv_max_layers
            )
            kv_speaker_full = _concat_kv_caches(
                kv_speaker_cond, kv_speaker_cond, kv_speaker_cond
            )

        prefix_latent = (
            mx.concatenate(generated_chunks, axis=1)
            if generated_chunks
            else mx.zeros((batch_size, 0, latent_size), dtype=mx.float32)
        )

        full_prefix_latent = mx.concatenate(
            [prefix_latent, prefix_latent, prefix_latent], axis=0
        )
        kv_latent_full = model.get_kv_cache_latent(full_prefix_latent)
        kv_latent_cond = [(k[:batch_size], v[:batch_size]) for (k, v) in kv_latent_full]

        x_t = mx.random.normal((batch_size, block_size, latent_size))
        trunc = (
            DEFAULT_TRUNCATION_FACTOR
            if truncation_factor is None
            else truncation_factor
        )
        x_t = x_t * trunc

        for i in range(num_steps):
            t = float(t_schedule[i])
            t_next = float(t_schedule[i + 1])
            has_cfg = cfg_min_t <= t <= cfg_max_t

            if has_cfg:
                out = model(
                    x=mx.concatenate([x_t, x_t, x_t], axis=0),
                    t=mx.full((batch_size * 3,), t, dtype=mx.float32),
                    text_mask=full_text_mask,
                    speaker_mask=full_speaker_mask,
                    start_pos=start_pos,
                    kv_cache_text=kv_text_full,
                    kv_cache_speaker=kv_speaker_full,
                    kv_cache_latent=kv_latent_full,
                )
                v_cond, v_uncond_text, v_uncond_speaker = mx.split(out, 3, axis=0)
                v_pred = (
                    v_cond
                    + cfg_scale_text * (v_cond - v_uncond_text)
                    + cfg_scale_speaker * (v_cond - v_uncond_speaker)
                )
            else:
                v_pred = model(
                    x=x_t,
                    t=mx.full((batch_size,), t, dtype=mx.float32),
                    text_mask=text_mask,
                    speaker_mask=speaker_mask,
                    start_pos=start_pos,
                    kv_cache_text=kv_text_cond,
                    kv_cache_speaker=kv_speaker_cond,
                    kv_cache_latent=kv_latent_cond,
                )

            if rescale_k is not None and rescale_sigma is not None:
                v_pred = _temporal_score_rescale(
                    v_pred, x_t, t, rescale_k, rescale_sigma
                )

            if (
                speaker_kv_scale is not None
                and speaker_kv_min_t is not None
                and t_next < speaker_kv_min_t <= t
            ):
                kv_speaker_cond = _multiply_kv_cache(
                    kv_speaker_cond,
                    1.0 / speaker_kv_scale,
                    speaker_kv_max_layers,
                )
                kv_speaker_full = _concat_kv_caches(
                    kv_speaker_cond,
                    kv_speaker_cond,
                    kv_speaker_cond,
                )

            x_t = x_t + v_pred * (t_next - t)

        generated_chunks.append(x_t)
        start_pos += block_size

    return mx.concatenate(generated_chunks, axis=1)
