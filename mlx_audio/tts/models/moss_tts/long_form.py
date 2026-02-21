"""Long-form planning and continuity contracts for MOSS-TTS."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import mlx.core as mx
import numpy as np

from .request import MOSS_AUDIO_TOKENS_PER_SECOND

_SENTENCE_ENDING_CHARS = frozenset(".!?;。！？；")
_TRAILING_CLOSERS = frozenset("\"'”’)]}》】」』")


@dataclass(frozen=True)
class SegmentPlannerConfig:
    """Character-budget controls for deterministic long-form segmentation."""

    min_chars: int = 160
    target_chars: int = 320
    max_chars: int = 520

    def __post_init__(self):
        min_chars = int(self.min_chars)
        target_chars = int(self.target_chars)
        max_chars = int(self.max_chars)
        if min_chars <= 0:
            raise ValueError("min_chars must be positive")
        if target_chars < min_chars:
            raise ValueError("target_chars must be >= min_chars")
        if max_chars < target_chars:
            raise ValueError("max_chars must be >= target_chars")


@dataclass(frozen=True)
class PlannedSegment:
    """Single planned segment extracted from long-form source text."""

    segment_idx: int
    text: str
    start_offset: int
    end_offset: int

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class ContinuityConfig:
    """Bounded carry-forward controls between long-form segments."""

    prefix_audio_seconds: float = 2.0
    prefix_audio_max_tokens: int = 25
    prefix_text_max_chars: int = 0

    def __post_init__(self):
        if float(self.prefix_audio_seconds) < 0.0:
            raise ValueError("prefix_audio_seconds must be >= 0")
        if int(self.prefix_audio_max_tokens) < 0:
            raise ValueError("prefix_audio_max_tokens must be >= 0")
        if int(self.prefix_text_max_chars) < 0:
            raise ValueError("prefix_text_max_chars must be >= 0")


@dataclass(frozen=True)
class LongFormRuntimeConfig:
    """Runtime controls for segmented long-form execution."""

    planner: SegmentPlannerConfig = field(default_factory=SegmentPlannerConfig)
    continuity: ContinuityConfig = field(default_factory=ContinuityConfig)
    retry_attempts: int = 0

    def __post_init__(self):
        if int(self.retry_attempts) < 0:
            raise ValueError("retry_attempts must be >= 0")


@dataclass(frozen=True)
class ContinuityState:
    """Explicit cross-segment continuity payload."""

    prefix_audio: Optional[mx.array] = None
    prefix_text: Optional[str] = None


@dataclass(frozen=True)
class LongFormSegmentMetric:
    """Per-segment metric payload used by Phase 6 artifacts."""

    segment_idx: int
    segment_chars: int
    start_offset: int
    end_offset: int
    retry_count: int
    prompt_tokens_generated: int
    emitted_samples: int
    emitted_seconds: float
    segment_latency_seconds: float
    segment_peak_memory_gb: float
    total_peak_memory_gb: float
    prefix_audio_samples: int
    prefix_text_chars: int
    boundary_note: str = ""


@dataclass(frozen=True)
class BoundarySeamMetric:
    """Objective seam metric between consecutive emitted segments."""

    left_segment_idx: int
    right_segment_idx: int
    sample_window: int
    left_tail_rms: float
    right_head_rms: float
    boundary_jump: float
    normalized_jump: float
    energy_ratio: float
    flagged: bool


def _normalize_source_text(text: str) -> str:
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized


def _to_mono_waveform_np(audio: Optional[mx.array]) -> Optional[np.ndarray]:
    if audio is None:
        return None
    if audio.ndim == 1:
        return np.array(audio, dtype=np.float32, copy=False)
    if audio.ndim == 2:
        audio_np = np.array(audio, dtype=np.float32, copy=False)
        rows, cols = int(audio_np.shape[0]), int(audio_np.shape[1])
        # Heuristic: smaller axis is channel axis.
        if rows <= cols:
            return audio_np.mean(axis=0)
        return audio_np.mean(axis=1)
    raise ValueError(f"Expected 1D or 2D audio for seam analysis, got {audio.shape}")


def _collect_paragraph_boundaries(text: str) -> set[int]:
    return {match.end() for match in re.finditer(r"\n\s*\n+", text)}


def _collect_sentence_boundaries(text: str) -> set[int]:
    boundaries: set[int] = set()
    idx = 0
    text_len = len(text)
    while idx < text_len:
        char = text[idx]
        if char in _SENTENCE_ENDING_CHARS:
            end = idx + 1
            while end < text_len and text[end] in _TRAILING_CLOSERS:
                end += 1
            boundaries.add(end)
            idx = end
            continue
        idx += 1
    return boundaries


def _collect_whitespace_boundaries(text: str) -> set[int]:
    return {idx + 1 for idx, char in enumerate(text) if char.isspace()}


def _skip_leading_whitespace(text: str, start: int) -> int:
    idx = int(start)
    text_len = len(text)
    while idx < text_len and text[idx].isspace():
        idx += 1
    return idx


def _choose_boundary(
    candidates: Iterable[int],
    *,
    min_end: int,
    target_end: int,
    max_end: int,
) -> Optional[int]:
    filtered = [
        candidate for candidate in candidates if min_end <= candidate <= max_end
    ]
    if not filtered:
        return None
    # Deterministic tie-break: nearest target, then later split (larger chunk).
    return min(
        filtered, key=lambda candidate: (abs(candidate - target_end), -candidate)
    )


def plan_text_segments(
    text: str,
    *,
    config: SegmentPlannerConfig,
) -> list[PlannedSegment]:
    """Plan sentence/paragraph-first long-form segments under character budgets."""

    source = _normalize_source_text(text)
    if not source:
        return []

    if len(source) <= int(config.max_chars):
        return [
            PlannedSegment(
                segment_idx=0,
                text=source,
                start_offset=0,
                end_offset=len(source),
            )
        ]

    paragraph_boundaries = _collect_paragraph_boundaries(source)
    sentence_boundaries = _collect_sentence_boundaries(source)
    whitespace_boundaries = _collect_whitespace_boundaries(source)

    segments: list[PlannedSegment] = []
    source_len = len(source)
    start = 0

    while start < source_len:
        start = _skip_leading_whitespace(source, start)
        if start >= source_len:
            break

        remaining = source_len - start
        if remaining <= int(config.max_chars):
            end = source_len
        else:
            min_end = min(source_len, start + int(config.min_chars))
            target_end = min(source_len, start + int(config.target_chars))
            max_end = min(source_len, start + int(config.max_chars))

            end = _choose_boundary(
                paragraph_boundaries,
                min_end=min_end,
                target_end=target_end,
                max_end=max_end,
            )
            if end is None:
                end = _choose_boundary(
                    sentence_boundaries,
                    min_end=min_end,
                    target_end=target_end,
                    max_end=max_end,
                )
            if end is None:
                end = _choose_boundary(
                    whitespace_boundaries,
                    min_end=min_end,
                    target_end=target_end,
                    max_end=max_end,
                )
            if end is None:
                end = max_end

        if end <= start:
            end = min(source_len, start + int(config.max_chars))
            if end <= start:
                break

        segment_text = source[start:end].strip()
        if not segment_text:
            start = end + 1
            continue

        segments.append(
            PlannedSegment(
                segment_idx=len(segments),
                text=segment_text,
                start_offset=start,
                end_offset=end,
            )
        )
        start = end

    return segments


def compute_prefix_audio_sample_cap(
    *,
    sample_rate: int,
    config: ContinuityConfig,
) -> int:
    """Resolve prefix-audio cap from seconds and token budgets."""

    sr = int(sample_rate)
    if sr <= 0:
        return 0

    seconds_cap = int(round(float(config.prefix_audio_seconds) * sr))
    token_seconds = float(config.prefix_audio_max_tokens) / float(
        MOSS_AUDIO_TOKENS_PER_SECOND
    )
    token_cap = int(round(token_seconds * sr))

    positive_caps = [cap for cap in (seconds_cap, token_cap) if cap > 0]
    if not positive_caps:
        return 0
    return min(positive_caps)


def extract_prefix_audio_tail(
    audio: Optional[mx.array],
    *,
    sample_cap: int,
) -> Optional[mx.array]:
    """Extract a copied, bounded tail for continuity carry-forward."""

    if audio is None:
        return None

    cap = int(sample_cap)
    if cap <= 0:
        return None

    if int(audio.shape[0]) == 0:
        return None

    if audio.ndim == 1:
        tail = audio[-cap:]
    elif audio.ndim == 2:
        rows = int(audio.shape[0])
        cols = int(audio.shape[1])
        if rows >= cols:
            tail = audio[-cap:, :]
        else:
            tail = audio[:, -cap:]
    else:
        raise ValueError(
            f"Expected 1D or 2D audio for continuity tail, got {audio.shape}"
        )

    tail_np = np.array(tail, copy=True)
    return mx.array(tail_np, dtype=audio.dtype)


def trim_prefix_text_window(text: Optional[str], *, max_chars: int) -> Optional[str]:
    """Deterministically trim carry-forward text to the configured max window."""

    if text is None:
        return None

    cap = int(max_chars)
    normalized = " ".join(str(text).split())
    if cap <= 0 or not normalized:
        return None

    if len(normalized) <= cap:
        return normalized

    window = normalized[-cap:]
    if window and not window[0].isspace():
        first_space = window.find(" ")
        if 0 < first_space < len(window) - 1:
            window = window[first_space + 1 :]
    trimmed = window.strip()
    return trimmed or None


def compose_segment_text(
    *,
    segment_text: str,
    prefix_text: Optional[str],
) -> str:
    """Compose segment prompt text with optional bounded prefix context."""

    segment = str(segment_text).strip()
    prefix = "" if prefix_text is None else str(prefix_text).strip()
    if not prefix:
        return segment
    return f"{prefix}\n{segment}"


def evaluate_segment_boundary(
    *,
    left_audio: Optional[mx.array],
    right_audio: Optional[mx.array],
    left_segment_idx: int,
    right_segment_idx: int,
    sample_window: int = 480,
    normalized_jump_threshold: float = 1.5,
    energy_ratio_threshold: float = 3.0,
) -> Optional[BoundarySeamMetric]:
    """Compute seam discontinuity heuristic between two adjacent segments."""

    left = _to_mono_waveform_np(left_audio)
    right = _to_mono_waveform_np(right_audio)
    if left is None or right is None:
        return None
    if left.size == 0 or right.size == 0:
        return None

    window = max(1, min(int(sample_window), int(left.size), int(right.size)))
    left_tail = left[-window:]
    right_head = right[:window]

    left_tail_rms = float(np.sqrt(np.mean(np.square(left_tail))))
    right_head_rms = float(np.sqrt(np.mean(np.square(right_head))))
    boundary_jump = float(abs(left[-1] - right[0]))

    scale = max(left_tail_rms, right_head_rms, 1e-8)
    normalized_jump = boundary_jump / scale
    energy_ratio = (max(left_tail_rms, right_head_rms) + 1e-8) / (
        min(left_tail_rms, right_head_rms) + 1e-8
    )
    flagged = bool(
        normalized_jump > float(normalized_jump_threshold)
        or energy_ratio > float(energy_ratio_threshold)
    )

    return BoundarySeamMetric(
        left_segment_idx=int(left_segment_idx),
        right_segment_idx=int(right_segment_idx),
        sample_window=window,
        left_tail_rms=left_tail_rms,
        right_head_rms=right_head_rms,
        boundary_jump=boundary_jump,
        normalized_jump=normalized_jump,
        energy_ratio=energy_ratio,
        flagged=flagged,
    )


def merge_reference_with_prefix_audio(
    *,
    base_reference: Optional[list[Optional[object]]],
    prefix_audio: Optional[mx.array],
) -> Optional[list[Optional[object]]]:
    """Append prefix audio to existing references while preserving user references."""

    merged: list[Optional[object]] = []
    if base_reference is not None:
        merged.extend(base_reference)
    if prefix_audio is not None:
        merged.append(prefix_audio)
    return merged if merged else None


def advance_continuity_state(
    *,
    previous_state: ContinuityState,
    segment_audio: Optional[mx.array],
    segment_text: str,
    sample_rate: int,
    config: ContinuityConfig,
) -> ContinuityState:
    """Compute the next bounded continuity payload from emitted segment outputs."""

    sample_cap = compute_prefix_audio_sample_cap(sample_rate=sample_rate, config=config)
    prefix_audio = extract_prefix_audio_tail(segment_audio, sample_cap=sample_cap)

    combined_text = segment_text
    if previous_state.prefix_text:
        combined_text = f"{previous_state.prefix_text} {segment_text}"
    prefix_text = trim_prefix_text_window(
        combined_text,
        max_chars=int(config.prefix_text_max_chars),
    )

    return ContinuityState(prefix_audio=prefix_audio, prefix_text=prefix_text)


__all__ = [
    "BoundarySeamMetric",
    "ContinuityConfig",
    "ContinuityState",
    "LongFormRuntimeConfig",
    "LongFormSegmentMetric",
    "PlannedSegment",
    "SegmentPlannerConfig",
    "advance_continuity_state",
    "compose_segment_text",
    "compute_prefix_audio_sample_cap",
    "evaluate_segment_boundary",
    "extract_prefix_audio_tail",
    "merge_reference_with_prefix_audio",
    "plan_text_segments",
    "trim_prefix_text_window",
]
