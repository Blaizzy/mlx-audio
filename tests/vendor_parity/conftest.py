import mlx.core as mx
import pytest


@pytest.fixture(autouse=True)
def _seeded():
    mx.random.seed(0)


def assert_exactly_equal(got, want):
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    assert mx.array_equal(got, want).item()


def rng_fingerprint(n=8):
    return mx.random.uniform(shape=(n,))
