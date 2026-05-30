import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def _audio_duration_ms(num_samples: int, sr: int) -> int:
    return round(1000 * (num_samples / sr))


def _ms_to_sample(ms: int, sr: int) -> int:
    return int(ms * (sr / 1000.0))


def _quantize_pcm16(audio: np.ndarray) -> np.ndarray:
    return (audio * 32767.0).clip(-32768, 32767).astype(np.int16)


def _window_rms(pcm: np.ndarray, start_ms: int, end_ms: int, sr: int) -> float:
    start = _ms_to_sample(start_ms, sr)
    end = min(len(pcm), _ms_to_sample(end_ms, sr))
    if end <= start:
        return 0.0
    window = pcm[start:end].astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(window**2)))


def _detect_silent_ranges_ms(
    audio: np.ndarray,
    sr: int,
    min_silence_len: int = 1000,
    silence_thresh: float = -16.0,
    seek_step: int = 1,
) -> list[tuple[int, int]]:
    """Dependency-free port of pydub.silence.detect_silence for mono audio."""
    seg_len = _audio_duration_ms(len(audio), sr)
    if seg_len < min_silence_len:
        return []

    pcm = _quantize_pcm16(np.asarray(audio, dtype=np.float32))
    threshold = (10 ** (silence_thresh / 20.0)) * 32768.0

    silence_starts = []
    last_slice_start = seg_len - min_silence_len
    slice_starts = list(range(0, last_slice_start + 1, seek_step))
    if last_slice_start % seek_step:
        slice_starts.append(last_slice_start)

    for start_ms in slice_starts:
        if _window_rms(pcm, start_ms, start_ms + min_silence_len, sr) <= threshold:
            silence_starts.append(start_ms)

    if not silence_starts:
        return []

    silent_ranges = []
    prev_i = silence_starts.pop(0)
    current_range_start = prev_i

    for silence_start_i in silence_starts:
        continuous = silence_start_i == prev_i + seek_step
        silence_has_gap = silence_start_i > (prev_i + min_silence_len)
        if not continuous and silence_has_gap:
            silent_ranges.append((current_range_start, prev_i + min_silence_len))
            current_range_start = silence_start_i
        prev_i = silence_start_i

    silent_ranges.append((current_range_start, prev_i + min_silence_len))
    return silent_ranges


def _detect_nonsilent_ranges_ms(
    audio: np.ndarray,
    sr: int,
    min_silence_len: int = 1000,
    silence_thresh: float = -16.0,
    seek_step: int = 1,
) -> list[tuple[int, int]]:
    """Dependency-free port of pydub.silence.detect_nonsilent for mono audio."""
    seg_len = _audio_duration_ms(len(audio), sr)
    if seg_len == 0:
        return []

    silent_ranges = _detect_silent_ranges_ms(
        audio,
        sr,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=seek_step,
    )
    if not silent_ranges:
        return [(0, seg_len)]

    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == seg_len:
        return []

    prev_end = 0
    nonsilent_ranges = []
    for start_ms, end_ms in silent_ranges:
        nonsilent_ranges.append((prev_end, start_ms))
        prev_end = end_ms

    if silent_ranges[-1][1] != seg_len:
        nonsilent_ranges.append((prev_end, seg_len))

    if nonsilent_ranges and nonsilent_ranges[0] == (0, 0):
        nonsilent_ranges.pop(0)

    return nonsilent_ranges


def _split_on_silence_ranges_ms(
    audio: np.ndarray,
    sr: int,
    min_silence_len: int = 1000,
    silence_thresh: float = -16.0,
    keep_silence: int | bool = 100,
    seek_step: int = 1,
) -> list[tuple[int, int]]:
    """Dependency-free port of pydub.silence.split_on_silence for mono audio."""
    if isinstance(keep_silence, bool):
        keep_silence = _audio_duration_ms(len(audio), sr) if keep_silence else 0

    ranges = [
        (start_ms - keep_silence, end_ms + keep_silence)
        for start_ms, end_ms in _detect_nonsilent_ranges_ms(
            audio,
            sr,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            seek_step=seek_step,
        )
    ]

    for idx in range(len(ranges) - 1):
        last_end = ranges[idx][1]
        next_start = ranges[idx + 1][0]
        if next_start < last_end:
            midpoint = (last_end + next_start) // 2
            ranges[idx] = (ranges[idx][0], midpoint)
            ranges[idx + 1] = (midpoint, ranges[idx + 1][1])

    seg_len = _audio_duration_ms(len(audio), sr)
    return [(max(start_ms, 0), min(end_ms, seg_len)) for start_ms, end_ms in ranges]


def _slice_audio_ms(
    audio: np.ndarray, sr: int, start_ms: int, end_ms: int
) -> np.ndarray:
    start = max(0, _ms_to_sample(start_ms, sr))
    end = min(len(audio), _ms_to_sample(end_ms, sr))
    return np.asarray(audio[start:end], dtype=np.float32)


def _remove_silence(
    audio: np.ndarray,
    sr: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
) -> np.ndarray:
    """Remove middle and edge silences with pydub-compatible fixed thresholds."""
    processed = np.asarray(audio, dtype=np.float32)

    if mid_sil > 0:
        ranges = _split_on_silence_ranges_ms(
            processed,
            sr,
            min_silence_len=mid_sil,
            silence_thresh=-50,
            keep_silence=mid_sil,
            seek_step=10,
        )
        if not ranges:
            return processed[:0]
        processed = np.concatenate(
            [
                _slice_audio_ms(processed, sr, start_ms, end_ms)
                for start_ms, end_ms in ranges
            ]
        )

    ranges = _detect_nonsilent_ranges_ms(
        processed, sr, min_silence_len=1, silence_thresh=-50
    )
    if ranges:
        start_ms = max(0, ranges[0][0] - lead_sil)
        end_ms = min(_audio_duration_ms(len(processed), sr), ranges[-1][1] + trail_sil)
        processed = _slice_audio_ms(processed, sr, start_ms, end_ms)

    return processed.astype(np.float32, copy=False)


def _trim_long_audio(
    audio: np.ndarray,
    sr: int,
    max_duration: float = 15.0,
    trim_threshold: float = 20.0,
) -> np.ndarray:
    """Trim audio >trim_threshold seconds at the largest silence gap."""
    duration = len(audio) / sr
    if duration <= trim_threshold:
        return np.asarray(audio, dtype=np.float32)

    ranges = _detect_nonsilent_ranges_ms(
        audio, sr, min_silence_len=100, silence_thresh=-40, seek_step=10
    )
    if not ranges:
        return np.asarray(audio, dtype=np.float32)

    max_ms = int(max_duration * 1000)
    best_split = 0
    for start, end in ranges:
        if start > best_split and start <= max_ms:
            best_split = start
        if end > max_ms:
            break

    if best_split < int(3.0 * 1000):
        best_split = min(max_ms, _audio_duration_ms(len(audio), sr))

    return _slice_audio_ms(audio, sr, 0, best_split)


def create_voice_clone_prompt(
    ref_audio_path: str,
    tokenizer=None,
    ref_text: Optional[str] = None,
    preprocess: bool = True,
    max_duration_s: float = 15.0,
) -> mx.array:
    """Encode reference audio for voice cloning, matching k2-fsa/OmniVoice.

    Preprocessing (when enabled):
    - Resample to 24kHz using torchaudio-compatible sinc interpolation
    - RMS normalization (boost quiet audio to RMS=0.1)
    - Silence removal (middle silences >300ms, edge silences >100ms)
    - Trim audio >20s at largest silence gap (only when ref_text is None)
    """
    if tokenizer is None:
        return mx.zeros((0, 8), dtype=mx.int32)

    from mlx_audio.audio_io import read as audio_read
    from mlx_audio.codec.models.higgs_audio.higgs_audio import _sinc_resample

    path = Path(ref_audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    audio, sr = audio_read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)

    if sr != 24000:
        mono = _sinc_resample(mono, sr, 24000)
    sr = 24000

    if preprocess:
        rms = np.sqrt(np.mean(mono**2))
        if 0 < rms < 0.1:
            mono = mono * (0.1 / rms)

        if ref_text is None:
            mono = _trim_long_audio(mono, sr, max_duration=max_duration_s)
        elif len(mono) / sr > 20.0:
            logger.warning(
                "Reference audio is %.1fs (>20s) and ref_text was provided, "
                "skipping automatic trimming.",
                len(mono) / sr,
            )

        mono = _remove_silence(mono, sr)

    wav = mx.array(mono)[None, :, None]
    tokens = tokenizer.encode(wav)
    return tokens[0]
