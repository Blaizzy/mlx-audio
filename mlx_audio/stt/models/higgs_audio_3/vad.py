from typing import List, Optional, Tuple

import numpy as np

DEFAULT_SILERO_REPO = "mlx-community/silero-vad"


class SileroVADBackend:
    def __init__(
        self,
        sample_rate: int = 16000,
        repo_id: str = DEFAULT_SILERO_REPO,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> None:
        self.sample_rate = sample_rate
        self.repo_id = repo_id
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self._model = None

    def _load(self):
        from mlx_audio.vad import load as load_vad

        self._model = load_vad(self.repo_id)
        return self._model

    def speech_ranges(self, wav: np.ndarray) -> List[Tuple[int, int]]:
        model = self._model if self._model is not None else self._load()
        timestamps = model.get_speech_timestamps(
            wav.astype(np.float32),
            sample_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )
        ranges = [(int(t["start"]), int(t["end"])) for t in timestamps]
        return [(s, e) for s, e in ranges if e > s]


def _split_long(start: int, end: int, max_samples: int) -> List[Tuple[int, int]]:
    out = []
    pos = start
    while pos < end:
        nxt = min(end, pos + max_samples)
        out.append((pos, nxt))
        pos = nxt
    return out


def vad_chunk_ranges(
    wav: np.ndarray,
    chunk_samples: int,
    backend: Optional[SileroVADBackend] = None,
    split_vads: bool = False,
) -> List[Tuple[int, int]]:
    total = len(wav)
    cuts: List[Tuple[int, int]] = []
    if backend is not None:
        try:
            cuts = backend.speech_ranges(wav)
        except Exception:
            cuts = []
    if not cuts:
        return _split_long(0, total, chunk_samples)

    if split_vads:
        wv_chunks = list(cuts)
    else:
        wv_chunks = []
        prev_e = 0
        for idx, (start, end) in enumerate(cuts):
            s = min(prev_e, start)
            e = total if idx == len(cuts) - 1 else end
            if e > s:
                wv_chunks.append((s, e))
            prev_e = e

    out: List[Tuple[int, int]] = []
    for s, e in wv_chunks:
        out.extend(_split_long(s, e, chunk_samples))
    return out or _split_long(0, total, chunk_samples)
