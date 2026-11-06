import pytest

from mlx_audio.stt.models.nemo.alignment import (
    AlignedToken,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
)


@pytest.mark.parametrize(
    "merge", [merge_longest_contiguous, merge_longest_common_subsequence]
)
def test_short_overlap_keeps_straddling_token_once(merge):
    a = [AlignedToken(id=7, text=" word", start=10.0, duration=1.0)]
    b = [AlignedToken(id=7, text=" word", start=10.0, duration=1.0)]

    assert merge(a, b, overlap_duration=2.0) == b


@pytest.mark.parametrize(
    "merge", [merge_longest_contiguous, merge_longest_common_subsequence]
)
def test_short_overlap_preserves_surrounding_and_repeated_words(merge):
    before = AlignedToken(id=7, text=" word", start=0.0, duration=0.5)
    shared = AlignedToken(id=7, text=" word", start=10.0, duration=1.0)
    after = AlignedToken(id=7, text=" word", start=20.0, duration=0.5)

    assert merge([before, shared], [shared, after], overlap_duration=2.0) == [
        before,
        shared,
        after,
    ]


def test_unmatched_lcs_partitions_straddling_tokens_by_center():
    a = [
        AlignedToken(id=1, text=" left", start=9.0, duration=1.0),
        AlignedToken(id=2, text=" crossing", start=10.0, duration=2.0),
    ]
    b = [
        AlignedToken(id=3, text=" crossing", start=10.0, duration=2.0),
        AlignedToken(id=4, text=" right", start=12.0, duration=1.0),
    ]

    assert merge_longest_common_subsequence(a, b, overlap_duration=2.0) == [
        a[0],
        b[0],
        b[1],
    ]
