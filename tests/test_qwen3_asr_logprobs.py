from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
    Qwen3ASRModel,
    _logprob_metadata,
)


def test_logprob_metadata_preserves_tokens_and_summarizes_them():
    metadata = _logprob_metadata([-0.25, -0.75])

    assert metadata == {
        "token_logprobs": [-0.25, -0.75],
        "avg_logprob": -0.5,
        "min_logprob": -0.75,
    }
    assert _logprob_metadata([]) == {"token_logprobs": []}


def test_single_chunk_keeps_selected_token_logprobs():
    model = SimpleNamespace(
        stream_generate=lambda *args, **kwargs: iter(
            [
                (1, mx.array([-2.0, -0.25, -1.0])),
                (2, mx.array([-3.0, -2.0, -0.5])),
            ]
        ),
        _preprocess_audio=lambda audio: (None, None, 4),
        _build_prompt=lambda *args: mx.zeros((1, 6), dtype=mx.int32),
        _tokenizer=SimpleNamespace(
            decode=lambda tokens, skip_special_tokens: "decoded"
        ),
    )

    text, prompt_tokens, generation_tokens, token_logprobs = (
        Qwen3ASRModel._generate_single_chunk(model, np.zeros(16000))
    )

    assert text == "decoded"
    assert prompt_tokens == 6
    assert generation_tokens == 2
    assert token_logprobs == [-0.25, -0.5]
