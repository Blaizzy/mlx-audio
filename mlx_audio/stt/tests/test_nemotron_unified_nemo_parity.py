"""Nemotron Unified against NeMo's own tokens on the repo's test audio.

The fixture holds token ids from NeMo (``EncDecRNNTBPEModelWithPrompt``, fp32 on CPU,
the ``speech_to_text_streaming_infer_rnnt.py`` buffer loop at batch 1, 5.6 s left
context) for ``mega_asr/fixtures/clean.wav`` (16 kHz, so no resampler differs), offline and at NVIDIA's latency
rows. An fp32 conversion must match token for token; bf16 and quantized ones within
a small edit rate.

NEMOTRON_UNIFIED_MLX_PATH names the checkpoint, a local directory or a Hugging Face
repo id, as ``load`` takes either:

    NEMOTRON_UNIFIED_MLX_PATH=T0mSIlver/nemotron-labs-unified-asr-0.6b-8bit \
        pytest mlx_audio/stt/tests/test_nemotron_unified_nemo_parity.py
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

pytestmark = pytest.mark.requires_weights

FIXTURE = Path(__file__).parent / "data" / "nemotron_unified_nemo_tokens.json"
AUDIO = Path(__file__).parent / "mega_asr" / "fixtures" / "clean.wav"
CONTEXTS = {
    "160": (1, 1),
    "320": (1, 3),
    "560": (2, 5),
    "1120": (7, 7),
    "2080": (13, 13),
}


def _edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


@pytest.fixture(scope="module")
def model():
    path = os.environ.get("NEMOTRON_UNIFIED_MLX_PATH")
    if not path:
        pytest.skip(
            "set NEMOTRON_UNIFIED_MLX_PATH to a Nemotron Unified checkpoint "
            "(a local directory or a Hugging Face repo id)"
        )
    from mlx_audio.stt import load

    return load(path)


def _tolerance(model):
    weight = model.encoder.layers[0].feed_forward1.linear1.weight
    return 0.0 if weight.dtype == mx.float32 else 0.03


@pytest.mark.parametrize("latency", ["offline", *CONTEXTS])
def test_tokens_match_nemo(model, latency):
    from mlx_audio.stt.utils import load_audio

    expected = json.loads(FIXTURE.read_text())["clean"][latency]
    audio = load_audio(str(AUDIO), sr=16000)
    if latency == "offline":
        result = model.generate(audio, language="en")
    else:
        chunk, right = CONTEXTS[latency]
        result = list(
            model.stream_generate(
                audio, language="en", att_context_size=[70, chunk, right]
            )
        )[-1]
    tokens = [t.id for s in result.sentences for t in s.tokens]
    rate = _edits(tokens, expected) / max(len(expected), 1)
    assert rate <= _tolerance(model), (latency, rate, tokens, expected)
