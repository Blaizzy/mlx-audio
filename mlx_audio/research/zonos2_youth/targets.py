from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlx_audio.tts.models.zonos2.prompt import shear, shear_up


INVALID_TARGET_ID = -100


@dataclass(frozen=True)
class TeacherForcedBatch:
    input_ids: mx.array
    targets: mx.array
    loss_mask: mx.array
    prompt_len: int
    target_frames: int


def build_teacher_forced_batch(
    prompt_rows: mx.array,
    unsheared_codes: mx.array,
    *,
    audio_pad_id: int,
    text_vocab: int,
) -> TeacherForcedBatch:
    """Build prompt-plus-target inputs and masked delayed-codebook targets.

    The model predicts delayed/sheared audio rows. Target row 0 is predicted from
    the final prompt row; target row n is predicted from target row n-1.
    """

    if prompt_rows.ndim != 2:
        raise ValueError("prompt_rows must be [prompt_len, frame_width]")
    if unsheared_codes.ndim != 2:
        raise ValueError("unsheared_codes must be [frames, codebooks]")
    codebooks = int(unsheared_codes.shape[1])
    frame_width = codebooks + 1
    if int(prompt_rows.shape[1]) != frame_width:
        raise ValueError("prompt frame width must equal codebooks + text column")

    delayed = shear(unsheared_codes.astype(mx.int32), int(audio_pad_id))
    text_col = mx.full((delayed.shape[0], 1), int(text_vocab), dtype=mx.int32)
    delayed_rows = mx.concatenate([delayed, text_col], axis=1)

    if delayed_rows.shape[0] == 0:
        input_ids = prompt_rows.astype(mx.int32)
    else:
        input_ids = mx.concatenate(
            [prompt_rows.astype(mx.int32), delayed_rows[:-1]],
            axis=0,
        )

    targets = mx.full(
        (input_ids.shape[0], codebooks),
        INVALID_TARGET_ID,
        dtype=mx.int32,
    )
    if delayed.shape[0] > 0:
        start = int(prompt_rows.shape[0]) - 1
        end = start + int(delayed.shape[0])
        targets = mx.concatenate(
            [
                targets[:start],
                delayed,
                targets[end:],
            ],
            axis=0,
        )
    loss_mask = targets != int(audio_pad_id)
    loss_mask = loss_mask & (targets != INVALID_TARGET_ID)
    return TeacherForcedBatch(
        input_ids=input_ids,
        targets=targets,
        loss_mask=loss_mask,
        prompt_len=int(prompt_rows.shape[0]),
        target_frames=int(unsheared_codes.shape[0]),
    )


def unshear_targets(delayed_targets: mx.array, *, audio_pad_id: int) -> mx.array:
    return shear_up(delayed_targets, int(audio_pad_id))


def masked_cross_entropy(
    logits: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
) -> dict[str, mx.array]:
    if logits.ndim != 3:
        raise ValueError("logits must be [time, codebooks, vocab]")
    if targets.shape != loss_mask.shape or logits.shape[:2] != targets.shape:
        raise ValueError("logits, targets, and loss_mask shapes are inconsistent")
    safe_targets = mx.where(loss_mask, targets, mx.zeros_like(targets))
    gathered = mx.take_along_axis(logits, safe_targets[..., None], axis=-1)[..., 0]
    nll = mx.logsumexp(logits, axis=-1) - gathered
    mask_f = loss_mask.astype(mx.float32)
    valid = mx.maximum(mask_f.sum(), mx.array(1.0, dtype=mx.float32))
    loss = (nll * mask_f).sum() / valid
    pred = mx.argmax(logits, axis=-1)
    acc = ((pred == safe_targets) & loss_mask).astype(mx.float32).sum() / valid
    probs = mx.softmax(logits, axis=-1)
    entropy = -(probs * mx.log(probs + 1e-9)).sum(axis=-1)
    per_codebook_valid = mx.maximum(mask_f.sum(axis=0), mx.ones((targets.shape[1],)))
    per_codebook_loss = (nll * mask_f).sum(axis=0) / per_codebook_valid
    per_codebook_accuracy = (
        ((pred == safe_targets) & loss_mask).astype(mx.float32).sum(axis=0)
        / per_codebook_valid
    )
    return {
        "loss": loss,
        "accuracy": acc,
        "entropy": (entropy * mask_f).sum() / valid,
        "valid_token_count": loss_mask.sum(),
        "per_codebook_loss": per_codebook_loss,
        "per_codebook_accuracy": per_codebook_accuracy,
    }

