from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from .config import AudioConfig
from .duration import estimate_speech_duration
from .guidance import MultiModalGuiderParams, calculate_guided_prediction
from .latent import AudioLatentShape, LatentState
from .scheduler import euler_step, ltx2_sigmas
from .transformer import Modality


def aligned_frame_count(duration: float, fps: float = 25.0) -> int:
    frames = int(round(duration * fps)) + 1
    return ((frames - 1 + 4) // 8) * 8 + 1


def target_shape_for_duration(
    duration: float,
    audio_config: AudioConfig,
    batch: int = 1,
) -> AudioLatentShape:
    frames = aligned_frame_count(duration, fps=audio_config.fps)
    latent_duration = float(frames) / float(audio_config.fps)
    return AudioLatentShape.from_duration(
        batch=batch,
        duration=latent_duration,
        channels=audio_config.vae_channels,
        mel_bins=audio_config.mel_bins,
        sample_rate=audio_config.latent_sample_rate,
        hop_length=audio_config.hop_length,
        audio_latent_downsample_factor=audio_config.latent_downsample_factor,
    )


def resolve_generation_duration(
    prompt: str,
    speed: float = 1.0,
    duration_multiplier: float = 1.1,
    gen_duration: float = 0.0,
) -> float:
    if gen_duration and gen_duration > 0:
        return float(gen_duration)
    return max(
        3.0, round(estimate_speech_duration(prompt, speed) * duration_multiplier, 1)
    )


def patch_long_clip_silence_prior(latent: mx.array) -> mx.array:
    if latent.shape[2] <= 513:
        return latent
    patched = mx.array(latent)
    f0, f1 = 511, 514
    denom = f1 - f0
    for frame in (512, 513):
        t = (frame - f0) / denom
        interp = (1.0 - t) * latent[:, :, f0, :] + t * latent[:, :, f1, :]
        patched[:, :, frame, :] = interp
    return patched


def _make_audio_modality(
    state: LatentState,
    sigma: mx.array,
    context: mx.array,
    context_mask: mx.array | None,
) -> Modality:
    timesteps = state.denoise_mask[..., 0] * sigma.reshape(1, 1)
    return Modality(
        latent=state.latent,
        sigma=sigma.reshape(-1),
        timesteps=timesteps,
        positions=state.positions,
        context=context,
        context_mask=context_mask,
        attention_mask=state.attention_mask,
    )


def guided_euler_loop(
    state: LatentState,
    x0_model: Callable[[Modality, set[int] | None], mx.array],
    context: mx.array,
    negative_context: mx.array | None = None,
    context_mask: mx.array | None = None,
    negative_context_mask: mx.array | None = None,
    steps: int = 30,
    guider_params: MultiModalGuiderParams | None = None,
) -> LatentState:
    guider_params = guider_params or MultiModalGuiderParams()
    sigmas = ltx2_sigmas(steps=steps, latent=state.latent)
    current = state.latent

    for step_index in range(len(sigmas) - 1):
        sigma = sigmas[step_index : step_index + 1]
        step_state = LatentState(
            latent=current,
            denoise_mask=state.denoise_mask,
            positions=state.positions,
            clean_latent=state.clean_latent,
            attention_mask=state.attention_mask,
        )
        modality = _make_audio_modality(step_state, sigma, context, context_mask)
        cond = x0_model(modality, None)

        if (
            guider_params.cfg_scale == 1.0
            and guider_params.stg_scale == 0.0
            and guider_params.modality_scale == 1.0
        ):
            denoised = cond
        else:
            if negative_context is None:
                uncond_text = cond
            else:
                neg_modality = _make_audio_modality(
                    step_state,
                    sigma,
                    negative_context,
                    (
                        negative_context_mask
                        if negative_context_mask is not None
                        else context_mask
                    ),
                )
                uncond_text = x0_model(neg_modality, None)
            if guider_params.stg_scale == 0.0:
                uncond_perturbed = cond
            else:
                perturbed_blocks = set(guider_params.stg_blocks)
                uncond_perturbed = x0_model(modality, perturbed_blocks)
            denoised = calculate_guided_prediction(
                cond=cond,
                uncond_text=uncond_text,
                uncond_perturbed=uncond_perturbed,
                uncond_modality=cond,
                params=guider_params,
            )

        denoised = (
            denoised * state.denoise_mask
            + state.clean_latent.astype(mx.float32) * (1.0 - state.denoise_mask)
        ).astype(denoised.dtype)
        current = euler_step(current, denoised, sigmas, step_index)

    return LatentState(
        latent=current,
        denoise_mask=state.denoise_mask,
        positions=state.positions,
        clean_latent=state.clean_latent,
        attention_mask=state.attention_mask,
    )
