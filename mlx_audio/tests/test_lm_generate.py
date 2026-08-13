import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_audio.lm.generate import generate_step, stream_generate

VOCAB = 17
EOS = 5


class _Cache:
    state = []


class CycleModel(nn.Module):
    """Emits token (t + 1) % VOCAB, so a prompt of EOS-1 produces EOS first."""

    layers = [object()]

    def make_cache(self):
        return [_Cache()]

    def __call__(self, tokens, cache=None, input_embeddings=None):
        del cache, input_embeddings
        return mx.eye(VOCAB)[(tokens + 1) % VOCAB]


class Tok:
    eos_token_ids = {EOS}
    eos_token_id = EOS
    bos_token = None
    clean_up_tokenization_spaces = False

    def encode(self, text, **kwargs):
        return [1]

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)


def responses(prompt, **kwargs):
    return list(stream_generate(CycleModel(), Tok(), mx.array(prompt), **kwargs))


def test_final_response_carries_eos_token_and_stop_reason():
    out = responses([EOS - 1], max_tokens=10)
    assert out, "expected at least the terminal response"
    assert out[-1].finish_reason == "stop"
    assert out[-1].token == EOS


def test_first_token_eos_still_yields_a_final_response():
    """A caller reading finish_reason off the last response must get one."""
    out = responses([EOS - 1], max_tokens=10)
    assert len(out) == 1
    assert out[-1].finish_reason == "stop"


def test_length_finish_reason_when_eos_never_reached():
    out = responses([EOS + 1], max_tokens=3)
    assert out[-1].finish_reason == "length"
    assert out[-1].token != EOS


def test_eos_token_is_not_a_duplicate_of_the_previous_token():
    """Re-emitting the last audio code instead of EOS shifts codec framing."""
    out = responses([EOS - 3], max_tokens=10)
    assert out[-1].token == EOS
    if len(out) > 1:
        assert out[-1].token != out[-2].token


@pytest.mark.parametrize("max_tokens", [1, 2, 5])
def test_generate_step_respects_max_tokens(max_tokens):
    toks = [
        int(t)
        for t, _ in generate_step(mx.array([1, 2]), CycleModel(), max_tokens=max_tokens)
    ]
    assert len(toks) == max_tokens


def test_generate_step_negative_max_tokens_is_unbounded():
    stream = generate_step(mx.array([1, 2]), CycleModel(), max_tokens=-1)
    produced = [int(pair[0]) for pair, _ in zip(stream, range(40))]
    assert len(produced) == 40
