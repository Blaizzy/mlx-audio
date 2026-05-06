from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

DEFAULT_SILERO_REPO = "mlx-community/silero-vad"
_CHUNK_SAMPLES = 512
_BLOCKS_PER_256MS = 8
_BLOCK_SAMPLES = _CHUNK_SAMPLES * _BLOCKS_PER_256MS
_BLOCK_DUR_S = _BLOCK_SAMPLES / 16000


@dataclass
class SpeechRun:
    start_sample: int
    end_sample: int


class VADBackend(Protocol):
    sample_rate: int

    def detect_speech(self, waveform: np.ndarray) -> List[SpeechRun]: ...


class SileroMlxBackend:
    sample_rate: int = 16000

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_SILERO_REPO,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> None:
        self.repo_id = repo_id
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self._model = None

    def _load(self) -> None:
        from mlx_audio.vad import load as load_vad

        self._model = load_vad(self.repo_id)

    def detect_speech(self, waveform: np.ndarray) -> List[SpeechRun]:
        if self._model is None:
            self._load()
        sr = self.sample_rate
        probs_32 = np.array(
            self._model._predict_proba_array(waveform.astype(np.float32), sr)
        ).reshape(-1)
        n = (probs_32.shape[0] // _BLOCKS_PER_256MS) * _BLOCKS_PER_256MS
        if n == 0:
            return []
        probs_256 = (
            1.0 - np.prod((1.0 - probs_32[:n]).reshape(-1, _BLOCKS_PER_256MS), axis=1)
        ).astype(np.float32)
        speech_pad_blocks = max(0, int(self.speech_pad_ms / 1000 / _BLOCK_DUR_S))
        min_speech_blocks = max(
            1, int(self.min_speech_duration_ms / 1000 / _BLOCK_DUR_S)
        )
        min_silence_blocks = max(
            1, int(self.min_silence_duration_ms / 1000 / _BLOCK_DUR_S)
        )
        actual_len = int(waveform.shape[0])
        runs: List[SpeechRun] = []
        in_speech = False
        seg_start = 0
        last_speech = -1
        silent_run = 0
        for idx, p in enumerate(probs_256):
            if p >= self.threshold:
                if not in_speech:
                    seg_start = max(0, idx - speech_pad_blocks)
                    in_speech = True
                last_speech = idx
                silent_run = 0
            elif in_speech:
                silent_run += 1
                if silent_run >= min_silence_blocks:
                    seg_end = min(last_speech + 1 + speech_pad_blocks, len(probs_256))
                    if seg_end - seg_start >= min_speech_blocks:
                        s = seg_start * _BLOCK_SAMPLES
                        e = min(seg_end * _BLOCK_SAMPLES, actual_len)
                        if s < e:
                            runs.append(SpeechRun(s, e))
                    in_speech = False
                    silent_run = 0
                    last_speech = -1
        if in_speech:
            end_idx = min(len(probs_256), last_speech + 1 + speech_pad_blocks)
            if end_idx - seg_start >= min_speech_blocks:
                s = seg_start * _BLOCK_SAMPLES
                e = min(end_idx * _BLOCK_SAMPLES, actual_len)
                if s < e:
                    runs.append(SpeechRun(s, e))
        return runs


def get_backend(name) -> VADBackend:
    if name is True or name == "silero-mlx":
        return SileroMlxBackend()
    raise ValueError(f"unknown vad backend: {name!r} (supported: True, 'silero-mlx')")


def _split_long(start: int, end: int, max_chunk_samples: int) -> List[List[int]]:
    if end - start <= max_chunk_samples:
        return [[start, end]]
    parts: List[List[int]] = []
    cur = start
    while cur < end:
        nxt = min(cur + max_chunk_samples, end)
        parts.append([cur, nxt])
        cur = nxt
    return parts


def merge_runs(
    runs: List[SpeechRun],
    sample_rate: int,
    *,
    merge_gap_s: float = 1.0,
    max_chunk_s: float = 30.0,
) -> List[SpeechRun]:
    if not runs:
        return runs
    max_chunk_samples = int(max_chunk_s * sample_rate)
    max_gap_samples = int(merge_gap_s * sample_rate)
    merged: List[List[int]] = list(
        _split_long(runs[0].start_sample, runs[0].end_sample, max_chunk_samples)
    )
    for r in runs[1:]:
        prev = merged[-1]
        gap = r.start_sample - prev[1]
        new_dur = r.end_sample - prev[0]
        if gap <= max_gap_samples and new_dur <= max_chunk_samples:
            prev[1] = r.end_sample
        else:
            merged.extend(_split_long(r.start_sample, r.end_sample, max_chunk_samples))
    return [SpeechRun(s, e) for s, e in merged]


def segment_audio(
    waveform: np.ndarray,
    backend: VADBackend,
    *,
    merge_gap_s: float = 1.0,
    max_chunk_s: float = 30.0,
) -> List[SpeechRun]:
    runs = backend.detect_speech(waveform)
    return merge_runs(
        runs, backend.sample_rate, merge_gap_s=merge_gap_s, max_chunk_s=max_chunk_s
    )
