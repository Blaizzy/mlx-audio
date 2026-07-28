from mlx_audio.stt.eval.normalize import normalize_for_wer


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_for_wer("Hello, WORLD!") == "hello world"


def test_normalize_preserves_apostrophes_and_maps_curly_apostrophes():
    assert normalize_for_wer("You'll see That’s fine.") == "you'll see that's fine"


def test_normalize_collapses_whitespace():
    assert normalize_for_wer("  A\t  spaced\nsentence.  ") == "a spaced sentence"
