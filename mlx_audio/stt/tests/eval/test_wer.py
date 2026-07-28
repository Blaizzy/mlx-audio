from mlx_audio.stt.eval.wer import aggregate_wer, compute_wer


def test_compute_wer_exact_match():
    result = compute_wer("hello world", "hello world")
    assert result.wer == 0.0
    assert result.edits == 0


def test_compute_wer_substitution():
    result = compute_wer("hello world", "hello there")
    assert result.substitutions == 1
    assert result.deletions == 0
    assert result.insertions == 0
    assert result.wer == 0.5


def test_compute_wer_deletion():
    result = compute_wer("hello world", "hello")
    assert result.deletions == 1
    assert result.wer == 0.5


def test_compute_wer_insertion():
    result = compute_wer("hello", "hello world")
    assert result.insertions == 1
    assert result.wer == 1.0


def test_aggregate_wer_reports_micro_and_macro():
    first = compute_wer("a b", "a")
    second = compute_wer("c", "x")
    aggregate = aggregate_wer([first, second])
    assert aggregate["wer_micro"] == 2 / 3
    assert aggregate["wer_macro"] == 0.75
    assert aggregate["total_reference_tokens"] == 3
