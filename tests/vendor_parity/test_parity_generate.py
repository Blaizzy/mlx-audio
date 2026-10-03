import mlx.core as mx
import mlx.nn as nn
import pytest
from conftest import assert_exactly_equal

upstream = pytest.importorskip("mlx_lm.generate")
upstream_llama = pytest.importorskip("mlx_lm.models.llama")
upstream_sample = pytest.importorskip("mlx_lm.sample_utils")

from mlx_audio.lm import generate as vendored
from mlx_audio.lm import sample_utils as vendored_sample
from mlx_audio.lm.models import llama as vendored_llama

ARGS = dict(
    model_type="llama",
    hidden_size=64,
    num_hidden_layers=2,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    rms_norm_eps=1e-5,
    vocab_size=128,
)


def paired_models():
    """Two structurally identical llama models sharing the same weights."""
    v = vendored_llama.Model(vendored_llama.ModelArgs(**ARGS))
    u = upstream_llama.Model(upstream_llama.ModelArgs(**ARGS))
    mx.eval(v.parameters())
    u.update(v.parameters())
    mx.eval(u.parameters())
    return v, u


def stream(mod, model, sampler=None, processors=None, **kw):
    mx.random.seed(7)
    return [
        (int(tok), lp)
        for tok, lp in mod.generate_step(
            mx.array([3, 9, 14, 2, 8]),
            model,
            max_tokens=16,
            sampler=sampler,
            logits_processors=processors,
            **kw,
        )
    ]


def compare(got, want):
    assert [t for t, _ in got] == [t for t, _ in want], "token streams diverge"
    for (_, g), (_, w) in zip(got, want):
        assert_exactly_equal(g, w)


def test_greedy_stream_matches_upstream():
    v, u = paired_models()
    compare(stream(vendored, v), stream(upstream, u))


@pytest.mark.parametrize("prefill", [1, 2, 4, 4096])
def test_prefill_chunking_matches_upstream(prefill):
    v, u = paired_models()
    compare(
        stream(vendored, v, prefill_step_size=prefill),
        stream(upstream, u, prefill_step_size=prefill),
    )


def test_sampler_stream_matches_upstream():
    v, u = paired_models()
    compare(
        stream(vendored, v, sampler=vendored_sample.make_sampler(temp=0.8, top_p=0.9)),
        stream(upstream, u, sampler=upstream_sample.make_sampler(temp=0.8, top_p=0.9)),
    )


def test_logits_processors_stream_matches_upstream():
    v, u = paired_models()
    compare(
        stream(
            vendored,
            v,
            processors=vendored_sample.make_logits_processors(repetition_penalty=1.2),
        ),
        stream(
            upstream,
            u,
            processors=upstream_sample.make_logits_processors(repetition_penalty=1.2),
        ),
    )


def test_rotating_cache_stream_matches_upstream():
    """max_kv_size forces RotatingKVCache, whose wraparound is the subtlest path."""
    v, u = paired_models()
    compare(
        stream(vendored, v, max_kv_size=8),
        stream(upstream, u, max_kv_size=8),
    )


def test_vendored_loop_against_upstream_model():
    """Cross control: isolates a loop bug from a backbone bug."""
    v, u = paired_models()
    compare(stream(vendored, u), stream(upstream, u))


def test_max_tokens_boundary_matches_upstream():
    v, u = paired_models()
    for n in (1, 2, 5):
        got = [
            int(t) for t, _ in vendored.generate_step(mx.array([3, 9]), v, max_tokens=n)
        ]
        want = [
            int(t) for t, _ in upstream.generate_step(mx.array([3, 9]), u, max_tokens=n)
        ]
        assert got == want and len(got) == n


def test_input_embeddings_stream_matches_upstream():
    v, u = paired_models()
    mx.random.seed(3)
    emb = mx.random.normal((5, ARGS["hidden_size"]))
    got = [
        int(t)
        for t, _ in vendored.generate_step(
            mx.array([], dtype=mx.int32), v, max_tokens=8, input_embeddings=emb
        )
    ]
    want = [
        int(t)
        for t, _ in upstream.generate_step(
            mx.array([], dtype=mx.int32), u, max_tokens=8, input_embeddings=emb
        )
    ]
    assert got == want
