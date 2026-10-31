import pytest

from mlx_audio.tts.models.kitten_tts.preprocess import TextPreprocessor, expand_currency


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        # A decimal amount whose integer part is 1 must use the SINGULAR unit.
        # The decimal branch used to hard-code "{unit}s", so these regressed to
        # "one dollars"/"one euros"/"one pounds ..." even though the whole-number
        # branch already singularised correctly ("$1" -> "one dollar").
        ("$1.00", "one dollar"),
        ("$1.50", "one dollar and fifty cents"),
        ("$1.99", "one dollar and ninety-nine cents"),
        ("€1.00", "one euro"),
        ("£1.01", "one pound and one cent"),
        # Whole-number branch is unchanged (already correct).
        ("$1", "one dollar"),
        ("$2", "two dollars"),
        # Plural stays plural when the integer part is not 1.
        ("$2.50", "two dollars and fifty cents"),
        ("$21.00", "twenty-one dollars"),
        ("$0.00", "zero dollars"),
        ("$0.50", "zero dollars and fifty cents"),
    ],
)
def test_expand_currency_singular_unit(input_text, expected):
    assert expand_currency(input_text) == expected


def test_currency_singular_through_pipeline():
    # Regression through the full preprocessor: "$1.50" must speak as
    # "one dollar and fifty cents", not "one dollars and fifty cents".
    pp = TextPreprocessor(remove_punctuation=False, lowercase=False)
    assert pp("It costs $1.50 today.") == "It costs one dollar and fifty cents today."
