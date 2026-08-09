import mlx.core as mx
import pytest
from conftest import assert_exactly_equal, rng_fingerprint

upstream = pytest.importorskip("mlx_lm.sample_utils")

from mlx_audio.lm import sample_utils as vendored

VOCAB = 128


def logprobs(batch=1, seed=0, dtype=mx.float32):
    mx.random.seed(seed)
    x = mx.random.normal((batch, VOCAB)).astype(dtype)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def both(fn_name, x, *args, **kwargs):
    """Call vendored and upstream under identical RNG state; compare result and state."""
    mx.random.seed(1234)
    got = getattr(vendored, fn_name)(x, *args, **kwargs)
    got_rng = rng_fingerprint()
    mx.random.seed(1234)
    want = getattr(upstream, fn_name)(x, *args, **kwargs)
    want_rng = rng_fingerprint()
    assert_exactly_equal(got, want)
    assert_exactly_equal(got_rng, want_rng)


@pytest.mark.parametrize("k", [1, 5, 64, VOCAB - 1])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("call_twice", [False, True])
def test_apply_top_k_matches_upstream(k, batch, call_twice):
    x = logprobs(batch)
    both("apply_top_k", x, k)
    if call_twice:
        both("apply_top_k", x, k)


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("call_twice", [False, True])
def test_apply_top_p_matches_upstream(p, batch, call_twice):
    x = logprobs(batch)
    both("apply_top_p", x, p)
    if call_twice:
        both("apply_top_p", x, p)


@pytest.mark.parametrize("p", [0.05, 0.5])
def test_apply_min_p_matches_upstream(p):
    both("apply_min_p", logprobs(2), p)


@pytest.mark.parametrize(
    "args", [(0.0,), (0.05, 5)], ids=["min_p=0", "min_tokens_to_keep"]
)
def test_apply_min_p_fails_identically_to_upstream(args):
    """Both raise on these inputs: min_p=0 hits math.log(0), min_tokens_to_keep
    hits a put_along_axis signature mismatch in mlx-lm 0.31.3."""
    x = logprobs(2)
    with pytest.raises(Exception) as got:
        vendored.apply_min_p(x, *args)
    with pytest.raises(Exception) as want:
        upstream.apply_min_p(x, *args)
    assert type(got.value) is type(want.value)


def test_apply_top_k_with_neg_inf_and_ties():
    x = mx.zeros((1, VOCAB))
    both("apply_top_k", x, 4)
    y = logprobs()
    y = mx.concatenate([mx.full((1, 4), -mx.inf), y[:, 4:]], axis=-1)
    both("apply_top_k", y, 4)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_apply_top_p_dtypes_match_upstream(dtype):
    both("apply_top_p", logprobs(dtype=dtype), 0.8)


def test_categorical_sampling_matches_upstream():
    both("categorical_sampling", logprobs(2), 1.0)


SAMPLER_CASES = [
    dict(temp=0.0),
    dict(temp=1.0),
    dict(temp=0.7, top_p=0.9),
    dict(temp=0.7, min_p=0.05),
    dict(temp=0.7, top_k=10),
    dict(temp=0.7, top_p=0.95, top_k=20, min_p=0.02),
    dict(temp=0.8, xtc_probability=0.5, xtc_threshold=0.1),
]


@pytest.mark.parametrize("kwargs", SAMPLER_CASES)
def test_make_sampler_token_streams_match_upstream(kwargs):
    v_sampler = vendored.make_sampler(**kwargs)
    u_sampler = upstream.make_sampler(**kwargs)
    for step in range(20):
        x = logprobs(seed=step)
        mx.random.seed(99)
        got = v_sampler(x)
        got_rng = rng_fingerprint()
        mx.random.seed(99)
        want = u_sampler(x)
        want_rng = rng_fingerprint()
        assert_exactly_equal(got, want)
        assert_exactly_equal(got_rng, want_rng)


PROCESSOR_CASES = [
    dict(repetition_penalty=1.2),
    dict(repetition_penalty=1.1, repetition_context_size=4),
    dict(presence_penalty=0.5),
    dict(frequency_penalty=0.3),
    dict(logit_bias={5: 10.0, 7: -10.0}),
    dict(repetition_penalty=1.15, presence_penalty=0.2, frequency_penalty=0.1),
]


@pytest.mark.parametrize("kwargs", PROCESSOR_CASES)
def test_make_logits_processors_matches_upstream(kwargs):
    v_procs = vendored.make_logits_processors(**kwargs)
    u_procs = upstream.make_logits_processors(**kwargs)
    assert len(v_procs) == len(u_procs)
    tokens = mx.array([3, 5, 5, 9, 3, 3, 1])
    x = logprobs()
    for vp, up in zip(v_procs, u_procs):
        assert_exactly_equal(vp(tokens, x), up(tokens, x))
