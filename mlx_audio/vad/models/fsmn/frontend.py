"""
FSMN-VAD frontend: Kaldi-style Fbank + LFR + CMVN

Aligned with FunASR WavFrontendOnline:
- Kaldi fbank via mlx_audio.dsp.compute_fbank_kaldi
- LFR: lfr_m=5, lfr_n=1
- CMVN: Kaldi Nnet format (AddShift + Rescale)
"""

import re
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from mlx_audio.dsp import compute_fbank_kaldi as _compute_fbank_kaldi


def load_cmvn(cmvn_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Kaldi Nnet format CMVN file (am.mvn).

    Format:
        <AddShift> D D
        <LearnRateCoef> 0 [ shift_values ]
        <Rescale> D D
        <LearnRateCoef> 0 [ scale_values ]

    CMVN operation: output = (input + shift) * scale
    """
    with open(cmvn_path, "r") as f:
        content = f.read()

    shift_match = re.search(r"<AddShift>.*?\[(.*?)\]", content, re.DOTALL)
    scale_match = re.search(r"<Rescale>.*?\[(.*?)\]", content, re.DOTALL)

    if not shift_match or not scale_match:
        raise ValueError(f"Cannot parse CMVN file: {cmvn_path}")

    shift = np.array([float(x) for x in shift_match.group(1).split()], dtype=np.float32)
    scale = np.array([float(x) for x in scale_match.group(1).split()], dtype=np.float32)

    return shift, scale


def compute_fbank(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_length_ms: int = 25,
    frame_shift_ms: int = 10,
    dither: float = 0.0,
) -> np.ndarray:
    """
    Kaldi-style fbank feature extraction via mlx_audio.dsp.

    Returns:
        fbank: [num_frames, n_mels] float32
    """
    win_len = int(sample_rate * frame_length_ms / 1000)
    win_inc = int(sample_rate * frame_shift_ms / 1000)

    # Scale to Kaldi PCM convention (int16 range)
    waveform_mx = mx.array(waveform * (1 << 15), dtype=mx.float32)

    fbank = _compute_fbank_kaldi(
        waveform_mx,
        sample_rate=sample_rate,
        win_len=win_len,
        win_inc=win_inc,
        num_mels=n_mels,
        win_type="hamming",
        dither=dither,
    )

    mx.eval(fbank)
    return np.array(fbank)


def apply_lfr(features: np.ndarray, lfr_m: int = 5, lfr_n: int = 1) -> np.ndarray:
    """
    Low Frame Rate: stack lfr_m frames every lfr_n steps.

    FunASR pads the left side by repeating the first frame.

    Args:
        features: [T, D]
        lfr_m: number of frames to stack
        lfr_n: step size

    Returns:
        [T', D * lfr_m]
    """
    T, D = features.shape
    left_pad = (lfr_m - 1) // 2
    if left_pad > 0:
        pad_frames = np.tile(features[0:1], (left_pad, 1))
        features = np.concatenate([pad_frames, features], axis=0)

    T_padded = features.shape[0]
    T_out = (T_padded + lfr_n - 1) // lfr_n
    out = np.zeros((T_out, D * lfr_m), dtype=np.float32)

    for i in range(T_out):
        start = i * lfr_n
        for j in range(lfr_m):
            idx = start + j
            if idx < T_padded:
                out[i, j * D : (j + 1) * D] = features[idx]
            else:
                out[i, j * D : (j + 1) * D] = features[T_padded - 1]

    return out


def apply_cmvn(
    features: np.ndarray, shift: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """
    Kaldi CMVN: output = (input + shift) * scale
    """
    return (features + shift) * scale


def extract_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_length_ms: int = 25,
    frame_shift_ms: int = 10,
    lfr_m: int = 5,
    lfr_n: int = 1,
    cmvn_path: Optional[str] = None,
    cmvn_shift: Optional[np.ndarray] = None,
    cmvn_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Full frontend: waveform -> Kaldi fbank -> LFR -> CMVN -> [T', 400]

    CMVN can be provided via file path (cmvn_path) or directly as
    numpy arrays (cmvn_shift, cmvn_scale). Direct arrays take priority.
    """
    fbank = compute_fbank(
        waveform, sample_rate, n_mels, frame_length_ms, frame_shift_ms
    )
    features = apply_lfr(fbank, lfr_m, lfr_n)

    if cmvn_shift is not None and cmvn_scale is not None:
        if len(cmvn_shift) == features.shape[1]:
            features = apply_cmvn(features, cmvn_shift, cmvn_scale)
    elif cmvn_path is not None:
        shift, scale = load_cmvn(cmvn_path)
        if len(shift) == features.shape[1]:
            features = apply_cmvn(features, shift, scale)

    return features
