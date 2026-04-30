from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np


@dataclass
class SpeechRun:
    start_sample: int
    end_sample: int


class VADBackend(Protocol):
    sample_rate: int

    def detect_speech(self, waveform: np.ndarray) -> List[SpeechRun]: ...


class SileroCoremlBackend:
    sample_rate: int = 16000

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> None:
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self._model = None

    def _load(self):
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError(
                "vad='silero-coreml' requires coremltools. "
                "Install with: pip install coremltools"
            ) from exc

        override = os.environ.get("MLX_AUDIO_SILERO_COREML")
        if override and os.path.isdir(override):
            model_path = override
        else:
            from huggingface_hub import snapshot_download

            cache_dir = os.path.expanduser(
                "~/.cache/mlx-audio/vad/silero-vad-coreml-mlx-audio"
            )
            snapshot_root = snapshot_download(
                repo_id="beshkenadze/silero-vad-coreml-mlx-audio",
                allow_patterns=["silero_vad_v6_256ms.mlpackage/**"],
                local_dir=cache_dir,
            )
            model_path = os.path.join(snapshot_root, "silero_vad_v6_256ms.mlpackage")
        self._model = ct.models.MLModel(model_path)

    def detect_speech(self, waveform: np.ndarray) -> List[SpeechRun]:
        if self._model is None:
            self._load()
        m = self._model
        sr = self.sample_rate
        chunk_total = 4096
        ctx_size = 64
        chunks_per_block = 8
        block_dur_s = chunks_per_block * 512 / sr

        audio = waveform.astype(np.float32)
        pad = chunk_total - len(audio) % chunk_total
        if pad and pad < chunk_total:
            audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])
        n_blocks = len(audio) // chunk_total

        h = np.zeros((1, 128), dtype=np.float32)
        c = np.zeros((1, 128), dtype=np.float32)
        context = np.zeros(ctx_size, dtype=np.float32)
        block_probs = np.zeros(n_blocks, dtype=np.float32)
        for i in range(n_blocks):
            block = audio[i * chunk_total : (i + 1) * chunk_total]
            with_ctx = np.concatenate([context, block])[None, :].astype(np.float32)
            out = m.predict(
                {
                    "audio_input": with_ctx,
                    "hidden_state": h,
                    "cell_state": c,
                }
            )
            block_probs[i] = out["vad_output"].flatten()[0]
            h = out["new_hidden_state"]
            c = out["new_cell_state"]
            context = block[-ctx_size:]

        runs = _hysteresis_runs(
            probs=block_probs,
            block_size=chunk_total,
            block_dur_s=block_dur_s,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        actual_len = len(waveform)
        return [
            SpeechRun(min(r.start_sample, actual_len), min(r.end_sample, actual_len))
            for r in runs
            if r.start_sample < actual_len
        ]


class WebRTCBackend:
    sample_rate: int = 16000

    def __init__(
        self,
        *,
        aggressiveness: int = 3,
        frame_ms: int = 30,
        padding_ms: int = 300,
        min_speech_ms: int = 200,
    ) -> None:
        self.aggressiveness = aggressiveness
        self.frame_ms = frame_ms
        self.padding_ms = padding_ms
        self.min_speech_ms = min_speech_ms
        self._vad = None

    def detect_speech(self, waveform: np.ndarray) -> List[SpeechRun]:
        if self._vad is None:
            try:
                import webrtcvad
            except ImportError as exc:
                raise ImportError(
                    "vad='webrtc' requires webrtcvad. "
                    "Install with: pip install webrtcvad"
                ) from exc
            self._vad = webrtcvad.Vad(self.aggressiveness)

        sr = self.sample_rate
        audio_int16 = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
        frame_samples = int(sr * self.frame_ms / 1000)
        pad_frames = self.padding_ms // self.frame_ms
        min_speech_frames = self.min_speech_ms // self.frame_ms

        runs: List[SpeechRun] = []
        cur_start = None
        speech_run = 0
        silent_run = 0
        for i in range(0, len(audio_int16) - frame_samples + 1, frame_samples):
            frame = audio_int16[i : i + frame_samples]
            is_speech = self._vad.is_speech(frame.tobytes(), sr)
            if is_speech:
                if cur_start is None:
                    cur_start = max(0, i - pad_frames * frame_samples)
                speech_run += 1
                silent_run = 0
            else:
                if cur_start is not None:
                    silent_run += 1
                    if silent_run > pad_frames:
                        if speech_run >= min_speech_frames:
                            runs.append(SpeechRun(cur_start, i + frame_samples))
                        cur_start = None
                        speech_run = 0
                        silent_run = 0
        if cur_start is not None and speech_run >= min_speech_frames:
            runs.append(SpeechRun(cur_start, len(audio_int16)))

        min_dur_samples = int(0.3 * sr)
        return [r for r in runs if r.end_sample - r.start_sample >= min_dur_samples]


def _hysteresis_runs(
    probs: np.ndarray,
    *,
    block_size: int,
    block_dur_s: float,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
) -> List[SpeechRun]:
    speech_pad_blocks = max(0, int(speech_pad_ms / 1000 / block_dur_s))
    min_speech_blocks = max(1, int(min_speech_duration_ms / 1000 / block_dur_s))
    min_silence_blocks = max(1, int(min_silence_duration_ms / 1000 / block_dur_s))

    runs: List[SpeechRun] = []
    in_speech = False
    seg_start = 0
    last_speech_idx = -1
    silent_run = 0
    n_blocks = len(probs)
    for idx, p in enumerate(probs):
        if p >= threshold:
            if not in_speech:
                seg_start = max(0, idx - speech_pad_blocks)
                in_speech = True
            last_speech_idx = idx
            silent_run = 0
        else:
            if in_speech:
                silent_run += 1
                if silent_run >= min_silence_blocks:
                    seg_end = min(last_speech_idx + 1 + speech_pad_blocks, n_blocks)
                    if seg_end - seg_start >= min_speech_blocks:
                        runs.append(
                            SpeechRun(seg_start * block_size, seg_end * block_size)
                        )
                    in_speech = False
                    silent_run = 0
                    last_speech_idx = -1
    if in_speech:
        if n_blocks - seg_start >= min_speech_blocks:
            runs.append(SpeechRun(seg_start * block_size, n_blocks * block_size))
    return runs


def get_backend(name) -> VADBackend:
    if name is True or name == "silero-coreml":
        return SileroCoremlBackend()
    if name == "webrtc":
        return WebRTCBackend()
    raise ValueError(f"unknown vad backend: {name!r}")


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
