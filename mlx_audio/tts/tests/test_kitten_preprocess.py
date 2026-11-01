import pytest

from mlx_audio.tts.models.kitten_tts.preprocess import TextPreprocessor, expand_time


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        # A bare time (no am/pm) followed by a word must keep the separating
        # space; the trailing whitespace used to be swallowed by the regex.
        ("3:30 tomorrow", "three thirty tomorrow"),
        ("9:05 and later", "nine oh five and later"),
        ("14:00 sharp", "fourteen hundred sharp"),
        ("meet at 3:30 then", "meet at three thirty then"),
        # A bare time followed by another number must not glue onto it.
        ("3:30 7", "three thirty 7"),
        # am/pm handling is unchanged: the space before the suffix is kept and
        # a following word stays separated.
        ("3:30pm", "three thirty pm"),
        ("3:30 pm tomorrow", "three thirty pm tomorrow"),
        ("9:05 AM", "nine oh five am"),
        ("12:00pm", "twelve pm"),
        ("14:00", "fourteen hundred"),
        ("3:30", "three thirty"),
    ],
)
def test_expand_time_keeps_trailing_space(input_text, expected):
    assert expand_time(input_text) == expected


def test_time_does_not_glue_onto_following_decimal_in_pipeline():
    # Regression: "3:30 0.5" used to become "three thirty0 five" because the
    # time ate the space and the glued "0.5" no longer parsed as a number.
    pp = TextPreprocessor(remove_punctuation=False, lowercase=False)
    assert pp("Meet me at 3:30 tomorrow.") == "Meet me at three thirty tomorrow."
    assert pp("The value is 3:30 0.5 here") == (
        "The value is three thirty zero point five here"
    )
