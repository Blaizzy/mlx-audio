from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.io import wavfile

FIX = Path(__file__).parent / "fixtures"


def test_logmel_matches_reference():
    from mlx_audio.stt.models.mega_asr.router import LogMel80

    sample_rate, waveform = wavfile.read(FIX / "clean.wav")
    reference = json.loads((FIX / "reference.json").read_text())

    assert sample_rate == 16000
    if np.issubdtype(waveform.dtype, np.integer):
        waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
    mel = LogMel80()(mx.array(waveform, mx.float32))
    got = np.array(mel[:10])
    expected = np.array(reference["router_logmel_clean_first10"], dtype=np.float32)

    assert got.shape == (10, 80)
    assert np.allclose(got, expected, atol=1e-3)


def test_conv_frontend_and_positional_encoding_shapes():
    from mlx_audio.stt.models.mega_asr.router import ConvFrontend, PositionalEncoding

    features = mx.random.normal((1, 40, 80))

    encoded = ConvFrontend()(features)
    positioned = PositionalEncoding(d_model=256)(encoded)

    assert encoded.shape == (1, 10, 256)
    assert positioned.shape == (1, 10, 256)
    assert np.isfinite(np.array(positioned)).all()
