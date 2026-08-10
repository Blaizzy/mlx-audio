import mlx.core as mx
import mlx.nn as nn
import pytest

upstream = pytest.importorskip("mlx_lm.generate")

from mlx_audio.lm import generate as vendored


class EmptyCache:
    state = []


class ToyModel(nn.Module):
    layers = [object()]

    def make_cache(self):
        return [EmptyCache()]

    def __call__(self, tokens, cache=None, input_embeddings=None):
        del cache, input_embeddings
        vocab_size = 17
        return mx.eye(vocab_size)[(tokens + 1) % vocab_size]


def test_generate_step_token_stream_matches_upstream():
    prompt = mx.array([1, 4, 7])
    model = ToyModel()

    got = list(vendored.generate_step(prompt, model, max_tokens=8, prefill_step_size=2))
    want = list(
        upstream.generate_step(prompt, model, max_tokens=8, prefill_step_size=2)
    )

    assert [token for token, _ in got] == [token for token, _ in want]
    for (_, got_logprobs), (_, want_logprobs) in zip(got, want):
        assert mx.array_equal(got_logprobs, want_logprobs).item()
