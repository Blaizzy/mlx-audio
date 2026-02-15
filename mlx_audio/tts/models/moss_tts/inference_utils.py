"""Pure delay-scheduler helpers for MOSS-TTS generation."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .config import ModelConfig

DELAY_INACTIVE = -1


@dataclass(frozen=True)
class DelaySchedulerState:
    is_stopping: mx.array
    is_audio: mx.array
    audio_lengths: mx.array
    delayed_lengths: mx.array


def _find_last_token_indices(text_ids: mx.array, token_id: int) -> mx.array:
    text_np = np.array(text_ids)
    last_indices = np.full((text_np.shape[0],), -1, dtype=np.int64)
    for row_idx in range(text_np.shape[0]):
        matches = np.nonzero(text_np[row_idx] == int(token_id))[0]
        if matches.size > 0:
            last_indices[row_idx] = int(matches[-1])
    return mx.array(last_indices, dtype=mx.int64)


def initialize_delay_scheduler_state(
    input_ids: mx.array,
    config: ModelConfig,
) -> DelaySchedulerState:
    """
    Initialize delay state from prompt inputs.

    `audio_lengths` tracks the number of generated assistant audio rows.
    `delayed_lengths` tracks flush progress (`-1` means inactive).
    """

    text_channel = input_ids[:, :, 0]
    batch_size = int(text_channel.shape[0])
    sequence_length = int(text_channel.shape[1])

    last_text_token = text_channel[:, -1]
    is_continuation = (last_text_token == config.audio_start_token_id) | (
        last_text_token == config.audio_assistant_gen_slot_token_id
    )

    audio_start_indices = _find_last_token_indices(
        text_channel, config.audio_start_token_id
    )
    has_audio_start = audio_start_indices >= 0
    audio_start_mask = is_continuation & has_audio_start
    sequence_lengths = mx.full((batch_size,), sequence_length, dtype=mx.int64)
    audio_lengths = mx.where(
        audio_start_mask,
        sequence_lengths - audio_start_indices,
        mx.zeros((batch_size,), dtype=mx.int64),
    )

    return DelaySchedulerState(
        is_stopping=mx.zeros((batch_size,), dtype=mx.bool_),
        is_audio=audio_start_mask.astype(mx.bool_),
        audio_lengths=audio_lengths.astype(mx.int64),
        delayed_lengths=mx.full((batch_size,), DELAY_INACTIVE, dtype=mx.int64),
    )


def build_delay_forced_text_tokens(
    state: DelaySchedulerState,
    config: ModelConfig,
    n_vq: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Build per-batch forced text tokens for Delay flush phases.

    Returns:
        - next_text_token: initialized with forced values (pad/delay/audio_end)
        - sampling_mask: rows that should sample from text logits
        - forcing_audio_eos: rows that just emitted forced audio_end
    """

    active = ~state.is_stopping
    forcing_delay = (
        active & (state.delayed_lengths >= 0) & (state.delayed_lengths < n_vq)
    )
    forcing_audio_eos = active & (state.delayed_lengths == n_vq)
    sampling_mask = active & (state.delayed_lengths < 0)

    next_text_token = mx.full(
        state.is_stopping.shape,
        config.pad_token_id,
        dtype=mx.int32,
    )
    next_text_token = mx.where(
        forcing_delay,
        int(config.audio_assistant_delay_slot_token_id),
        next_text_token,
    )
    next_text_token = mx.where(
        forcing_audio_eos,
        int(config.audio_end_token_id),
        next_text_token,
    )
    return (
        next_text_token.astype(mx.int32),
        sampling_mask.astype(mx.bool_),
        forcing_audio_eos,
    )


def build_delay_audio_sampling_mask(
    state: DelaySchedulerState,
    *,
    n_vq: int,
) -> mx.array:
    """
    Build `(B, n_vq)` mask for channels that should sample real audio tokens.
    """

    channel_index = mx.arange(n_vq, dtype=mx.int64)[None, :]
    pre_audio_mask = state.audio_lengths[:, None] > channel_index
    post_audio_mask = channel_index > (state.delayed_lengths[:, None] - 1)
    inactive_flush_mask = state.delayed_lengths[:, None] < 0
    post_audio_mask = mx.where(
        inactive_flush_mask,
        mx.ones(post_audio_mask.shape, dtype=mx.bool_),
        post_audio_mask,
    )
    return pre_audio_mask & post_audio_mask & (~state.is_stopping[:, None])


def update_delay_scheduler_state(
    state: DelaySchedulerState,
    *,
    next_text_token: mx.array,
    config: ModelConfig,
    n_vq: int,
    forcing_audio_eos: mx.array,
) -> DelaySchedulerState:
    next_text_token = next_text_token.astype(mx.int32)

    is_audio = mx.where(
        next_text_token == config.audio_start_token_id, True, state.is_audio
    )
    is_audio = mx.where(
        forcing_audio_eos | (next_text_token == config.audio_end_token_id),
        False,
        is_audio,
    )

    is_stopping = state.is_stopping | (next_text_token == config.im_end_token_id)

    audio_increment_mask = (
        (next_text_token == config.audio_start_token_id)
        | (next_text_token == config.audio_assistant_gen_slot_token_id)
        | (next_text_token == config.audio_assistant_delay_slot_token_id)
    )
    audio_lengths = state.audio_lengths + audio_increment_mask.astype(mx.int64)
    audio_lengths = mx.where(
        next_text_token == config.audio_end_token_id, 0, audio_lengths
    )

    delayed_lengths = state.delayed_lengths
    delay_start = (delayed_lengths < 0) & (
        next_text_token == config.audio_assistant_delay_slot_token_id
    )
    delayed_lengths = mx.where(delay_start, 0, delayed_lengths)

    delayed_active = delayed_lengths >= 0
    delayed_lengths = delayed_lengths + delayed_active.astype(mx.int64)
    delayed_lengths = mx.where(delayed_lengths > n_vq, DELAY_INACTIVE, delayed_lengths)

    return DelaySchedulerState(
        is_stopping=is_stopping.astype(mx.bool_),
        is_audio=is_audio.astype(mx.bool_),
        audio_lengths=audio_lengths.astype(mx.int64),
        delayed_lengths=delayed_lengths.astype(mx.int64),
    )


__all__ = [
    "DELAY_INACTIVE",
    "DelaySchedulerState",
    "build_delay_audio_sampling_mask",
    "build_delay_forced_text_tokens",
    "initialize_delay_scheduler_state",
    "update_delay_scheduler_state",
]
