"""Request normalization for MOSS-TTS-Realtime generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RealtimeNormalizedRequest:
    text: str
    include_system_prompt: bool = True
    reset_cache: bool = True
    chunk_frames: int = 40
    overlap_frames: int = 4
    decode_chunk_duration: Optional[float] = 0.32
    max_pending_frames: int = 4096
    repetition_window: Optional[int] = 50
    reference_audio: Optional[Any] = None

    def __post_init__(self):
        if self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if self.overlap_frames < 0:
            raise ValueError("overlap_frames must be non-negative")
        if self.max_pending_frames <= 0:
            raise ValueError("max_pending_frames must be positive")
        if self.overlap_frames >= self.chunk_frames:
            raise ValueError("overlap_frames must be smaller than chunk_frames")
        if self.decode_chunk_duration is not None and self.decode_chunk_duration <= 0:
            raise ValueError("decode_chunk_duration must be positive when provided")
        if self.repetition_window is not None and self.repetition_window <= 0:
            raise ValueError("repetition_window must be positive when provided")

    @classmethod
    def from_generate_kwargs(
        cls,
        *,
        text: Optional[str],
        ref_audio: Optional[Any] = None,
        include_system_prompt: Optional[bool] = None,
        reset_cache: Optional[bool] = None,
        chunk_frames: Optional[int] = None,
        overlap_frames: Optional[int] = None,
        decode_chunk_duration: Optional[float] = 0.32,
        max_pending_frames: Optional[int] = None,
        repetition_window: Optional[int] = None,
        **_: Any,
    ) -> "RealtimeNormalizedRequest":
        resolved_text = "" if text is None else str(text)
        resolved_repetition_window: Optional[int]
        if repetition_window is None:
            resolved_repetition_window = 50
        else:
            repetition_window_value = int(repetition_window)
            resolved_repetition_window = (
                None if repetition_window_value <= 0 else repetition_window_value
            )
        return cls(
            text=resolved_text,
            include_system_prompt=(
                True if include_system_prompt is None else bool(include_system_prompt)
            ),
            reset_cache=True if reset_cache is None else bool(reset_cache),
            chunk_frames=40 if chunk_frames is None else int(chunk_frames),
            overlap_frames=4 if overlap_frames is None else int(overlap_frames),
            decode_chunk_duration=(
                None if decode_chunk_duration is None else float(decode_chunk_duration)
            ),
            max_pending_frames=(
                4096 if max_pending_frames is None else int(max_pending_frames)
            ),
            repetition_window=resolved_repetition_window,
            reference_audio=ref_audio,
        )
