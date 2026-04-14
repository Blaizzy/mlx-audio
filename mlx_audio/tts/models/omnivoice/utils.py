import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def _remove_silence(
    audio: np.ndarray,
    sr: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
) -> np.ndarray:
    """Remove middle and edge silences using pydub, matching k2-fsa/OmniVoice."""
    try:
        from pydub import AudioSegment
        from pydub.silence import split_on_silence
    except ModuleNotFoundError:
        return audio

    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    seg = AudioSegment(pcm.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    if mid_sil > 0:
        parts = split_on_silence(
            seg,
            min_silence_len=mid_sil,
            silence_thresh=-50,
            keep_silence=mid_sil,
            seek_step=10,
        )
        seg = AudioSegment.silent(duration=0)
        for p in parts:
            seg += p

    from pydub.silence import detect_nonsilent

    ranges = detect_nonsilent(seg, min_silence_len=1, silence_thresh=-50)
    if ranges:
        start = max(0, ranges[0][0] - lead_sil)
        end = min(len(seg), ranges[-1][1] + trail_sil)
        seg = seg[start:end]

    samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32767
    return samples


def _trim_long_audio(
    audio: np.ndarray,
    sr: int,
    max_duration: float = 15.0,
    trim_threshold: float = 20.0,
) -> np.ndarray:
    """Trim audio >trim_threshold seconds at the largest silence gap."""
    duration = len(audio) / sr
    if duration <= trim_threshold:
        return audio

    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
    except ModuleNotFoundError:
        return audio

    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    seg = AudioSegment(pcm.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    ranges = detect_nonsilent(
        seg, min_silence_len=100, silence_thresh=-40, seek_step=10
    )
    if not ranges:
        return audio

    max_ms = int(max_duration * 1000)
    best_split = 0
    for start, end in ranges:
        if start > best_split and start <= max_ms:
            best_split = start
        if end > max_ms:
            break

    if best_split < int(3.0 * 1000):
        best_split = min(max_ms, len(seg))

    trimmed = seg[:best_split]
    samples = np.frombuffer(trimmed.raw_data, dtype=np.int16).astype(np.float32) / 32767
    return samples


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

    import soundfile as sf

    from mlx_audio.codec.models.higgs_audio.higgs_audio import _sinc_resample

    path = Path(ref_audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
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
