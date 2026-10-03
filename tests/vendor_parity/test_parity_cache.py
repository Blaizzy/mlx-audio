import mlx.core as mx
import pytest
from conftest import assert_exactly_equal

upstream = pytest.importorskip("mlx_lm.models.cache")

from mlx_audio.lm.models import cache as vendored

B, H, D = 2, 4, 8


def kv(n, seed):
    mx.random.seed(seed)
    return mx.random.normal((B, H, n, D)), mx.random.normal((B, H, n, D))


def compare_state(a, b):
    sa, sb = a.state, b.state
    assert len(sa) == len(sb)
    for x, y in zip(sa, sb):
        if x is None or y is None:
            assert x is y
        else:
            assert_exactly_equal(x, y)
    if hasattr(a, "offset"):
        assert_exactly_equal(mx.array(a.offset), mx.array(b.offset))


def drive(vc, uc, steps):
    for i, n in enumerate(steps):
        k, val = kv(n, i)
        vk, vv = vc.update_and_fetch(k, val)
        uk, uv = uc.update_and_fetch(k, val)
        assert_exactly_equal(vk, uk)
        assert_exactly_equal(vv, uv)
        compare_state(vc, uc)


@pytest.mark.parametrize(
    "steps",
    [
        [1] * 8,
        [7, 1, 1, 1],
        [256, 1, 1],
        [255, 2, 1],
        [512],
    ],
)
def test_kvcache_update_and_fetch_sequence_matches_upstream(steps):
    drive(vendored.KVCache(), upstream.KVCache(), steps)


@pytest.mark.parametrize("max_size", [4, 8, 256])
@pytest.mark.parametrize("keep", [0, 1, 4])
@pytest.mark.parametrize(
    "steps",
    [
        [1, 1, 1, 1],
        [1] * 12,
        [16],
        [6, 1, 1, 1, 1, 1],
    ],
)
def test_rotating_kvcache_wraparound_matches_upstream(max_size, keep, steps):
    if keep >= max_size:
        pytest.skip("keep must be smaller than max_size")
    drive(
        vendored.RotatingKVCache(max_size=max_size, keep=keep),
        upstream.RotatingKVCache(max_size=max_size, keep=keep),
        steps,
    )


@pytest.mark.parametrize("max_size", [4, 8])
@pytest.mark.parametrize("n", [1, 3, 9])
def test_rotating_kvcache_make_mask_matches_upstream(max_size, n):
    vc = vendored.RotatingKVCache(max_size=max_size, keep=0)
    uc = upstream.RotatingKVCache(max_size=max_size, keep=0)
    drive(vc, uc, [1] * 6)
    got, want = vc.make_mask(n), uc.make_mask(n)
    assert type(got) is type(want)
    if isinstance(want, mx.array):
        assert_exactly_equal(got, want)
    else:
        assert got == want


@pytest.mark.parametrize("wrapped", [False, True])
def test_batch_kvcache_merge_matches_upstream(wrapped):
    """Mirrors higgs_audio_v3 continuous batching: merge extracted single rows."""
    steps = [4, 1, 1] if wrapped else [2]
    vc = vendored.BatchKVCache([0, 3])
    uc = upstream.BatchKVCache([0, 3])
    mx.random.seed(0)
    for n in steps:
        k, val = mx.random.normal((2, H, n, D)), mx.random.normal((2, H, n, D))
        vc.update_and_fetch(k, val)
        uc.update_and_fetch(k, val)
    vm = vendored.BatchKVCache.merge([vc.extract(i) for i in range(2)])
    um = upstream.BatchKVCache.merge([uc.extract(i) for i in range(2)])
    assert_exactly_equal(vm.keys, um.keys)
    assert_exactly_equal(vm.values, um.values)
    assert_exactly_equal(mx.array(vm.offset), mx.array(um.offset))
    assert_exactly_equal(mx.array(vm.left_padding), mx.array(um.left_padding))


def test_kvcache_merge_matches_upstream():
    vcs = [vendored.KVCache() for _ in range(2)]
    ucs = [upstream.KVCache() for _ in range(2)]
    for vc, uc in zip(vcs, ucs):
        k, val = kv(3, 1)
        vc.update_and_fetch(k[:1], val[:1])
        uc.update_and_fetch(k[:1], val[:1])
    vm, um = vendored.KVCache.merge(vcs), upstream.KVCache.merge(ucs)
    assert_exactly_equal(vm.keys, um.keys)
    assert_exactly_equal(mx.array(vm.offset), mx.array(um.offset))


def test_arrays_cache_matches_upstream():
    vc, uc = vendored.ArraysCache(size=2), upstream.ArraysCache(size=2)
    x = mx.random.normal((B, 3, D))
    vc[0], uc[0] = x, x
    assert_exactly_equal(vc[0], uc[0])
    compare_state(vc, uc)


class _Toy:
    def __init__(self, n):
        self.layers = list(range(n))


def test_make_prompt_cache_matches_upstream():
    got = vendored.make_prompt_cache(_Toy(3))
    want = upstream.make_prompt_cache(_Toy(3))
    assert len(got) == len(want)
    assert [type(c).__name__ for c in got] == [type(c).__name__ for c in want]


def test_make_prompt_cache_with_max_kv_size_matches_upstream():
    got = vendored.make_prompt_cache(_Toy(3), max_kv_size=16)
    want = upstream.make_prompt_cache(_Toy(3), max_kv_size=16)
    assert [type(c).__name__ for c in got] == [type(c).__name__ for c in want]


def test_state_roundtrip_matches_upstream():
    vc, uc = vendored.KVCache(), upstream.KVCache()
    drive(vc, uc, [5, 1])
    vc2 = vendored.KVCache.from_state(vc.state, vc.meta_state)
    uc2 = upstream.KVCache.from_state(uc.state, uc.meta_state)
    compare_state(vc2, uc2)
