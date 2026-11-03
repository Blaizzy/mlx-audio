"""Golden spoken forms and optional checks using MAYA2_REFERENCE_SOURCE."""

import importlib.util
import os
import random
import sys
import types
from pathlib import Path

import pytest

from mlx_audio.tts.models.maya2.normalization import _Language, normalise_text

LANGUAGES = ("as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te")
SAMPLES = ["25", "₹95.41", "15/08/2025", "10:30", "OTP 482913"]
EXPECTED = {
    "as": [
        "পঁচিশ",
        "পঁচানব্বৈ টকা একচল্লিশ পইচা",
        "পোন্ধৰ আগষ্ট দুই হাজাৰ পঁচিশ",
        "দহ বাজি ত্ৰিশ মিনিট",
        "OTP চাৰি, আঠ, দুই, ন, এক, তিনি",
    ],
    "bn": [
        "পঁচিশ",
        "পঁচানব্বই টাকা একচল্লিশ পয়সা",
        "পনেরো আগস্ট দুই হাজার পঁচিশ",
        "দশটা বেজে ত্রিশ মিনিট",
        "OTP চার, আট, দুই, নয়, এক, তিন",
    ],
    "en": [
        "twenty five",
        "ninety five rupees forty one paise",
        "fifteen August two thousand twenty five",
        "ten thirty",
        "OTP four, eight, two, nine, one, three",
    ],
    "gu": [
        "પચ્ચીસ",
        "પંચાણું રૂપિયા એકતાળીસ પૈસા",
        "પંદર ઓગસ્ટ બે હજાર પચ્ચીસ",
        "દસ વાગીને ત્રીસ મિનિટ",
        "OTP ચાર, આઠ, બે, નવ, એક, ત્રણ",
    ],
    "hi": [
        "पच्चीस",
        "पंचानवे रुपये इकतालीस पैसे",
        "पंद्रह अगस्त दो हजार पच्चीस",
        "दस बजकर तीस मिनट",
        "OTP चार, आठ, दो, नौ, एक, तीन",
    ],
    "kn": [
        "ಇಪ್ಪತ್ತೈದು",
        "ತೊಂಬತ್ತೈದು ರೂಪಾಯಿ ನಲವತ್ತೊಂದು ಪೈಸೆ",
        "ಹದಿನೈದು ಆಗಸ್ಟ್ ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತೈದು",
        "ಹತ್ತು ಗಂಟೆ ಮೂವತ್ತು ನಿಮಿಷ",
        "OTP ನಾಲ್ಕು, ಎಂಟು, ಎರಡು, ಒಂಬತ್ತು, ಒಂದು, ಮೂರು",
    ],
    "ml": [
        "ഇരുപത്തിയഞ്ച്",
        "തൊണ്ണൂറ്റിയഞ്ച് രൂപ നാൽപ്പത്തിയൊന്ന് പൈസ",
        "പതിനഞ്ച് ഓഗസ്റ്റ് രണ്ടായിരത്തി ഇരുപത്തിയഞ്ച്",
        "പത്ത് മണി മുപ്പത് മിനിറ്റ്",
        "OTP നാല്, എട്ട്, രണ്ട്, ഒമ്പത്, ഒന്ന്, മൂന്ന്",
    ],
    "mr": [
        "पंचवीस",
        "पंच्याण्णव रुपये एक्केचाळीस पैसे",
        "पंधरा ऑगस्ट दोन हजार पंचवीस",
        "दहा वाजून तीस मिनिटे",
        "OTP चार, आठ, दोन, नऊ, एक, तीन",
    ],
    "or": [
        "ପଚିଶି",
        "ପଞ୍ଚାନବେ ଟଙ୍କା ଏକଚାଳିଶି ପଇସା",
        "ପନ୍ଦର ଅଗଷ୍ଟ ଦୁଇ ହଜାର ପଚିଶି",
        "ଦଶଟା ତିରିଶି ମିନିଟ୍",
        "OTP ଚାରି, ଆଠ, ଦୁଇ, ନଅ, ଏକ, ତିନି",
    ],
    "pa": [
        "ਪੱਚੀ",
        "ਪਚਾਨਵੇਂ ਰੁਪਏ ਇਕਤਾਲੀ ਪੈਸੇ",
        "ਪੰਦਰਾਂ ਅਗਸਤ ਦੋ ਹਜ਼ਾਰ ਪੱਚੀ",
        "ਦਸ ਵੱਜ ਕੇ ਤੀਹ ਮਿੰਟ",
        "OTP ਚਾਰ, ਅੱਠ, ਦੋ, ਨੌਂ, ਇੱਕ, ਤਿੰਨ",
    ],
    "ta": [
        "இருபத்தைந்து",
        "தொண்ணூற்றைந்து ரூபாய் நாற்பத்தொன்று பைசா",
        "ஆகஸ்ட் பதினைந்தாம் தேதி இரண்டாயிரத்து இருபத்தைந்து",
        "பத்து மணி முப்பது நிமிடம்",
        "OTP நான்கு, எட்டு, இரண்டு, ஒன்பது, ஒன்று, மூன்று",
    ],
    "te": [
        "ఇరవై ఐదు",
        "తొంభై ఐదు రూపాయలు నలభై ఒకటి పైసలు",
        "పదిహేను ఆగస్టు రెండు వేల ఇరవై ఐదు",
        "పది గంటల ముప్పై నిమిషాలు",
        "OTP నాలుగు, ఎనిమిది, రెండు, తొమ్మిది, ఒకటి, మూడు",
    ],
}


@pytest.mark.parametrize("lang", LANGUAGES)
@pytest.mark.parametrize("index", range(len(SAMPLES)))
def test_all_languages_have_verified_spoken_forms(lang, index):
    assert normalise_text(SAMPLES[index], lang) == EXPECTED[lang][index]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("[laughs] 2 [sighs]", "[laughs] two [sighs]"),
        ("[pause 0.5s] 12", "[pause 0.5s] twelve"),
        ("[OTP: १२] 2", "[OTP: १२] two"),
        ("[pause   0.5s] 2", "[pause   0.5s] two"),
        ("NASA and MAYA", "NASA and MAYA"),
        ("123456", "one lakh twenty three thousand four hundred fifty six"),
        ("1,23,456", "one lakh twenty three thousand four hundred fifty six"),
        ("2,500 people", "two thousand five hundred people"),
        ("Room 314", "Room three hundred fourteen"),
        (
            "OTP 1234. I have 200 cats.",
            "OTP one, two, three, four. I have two hundred cats.",
        ),
        ("OTP 0042", "OTP zero, zero, four, two"),
        (
            "Phone +919876543210",
            "Phone plus nine, one, nine, eight, seven, six, five, four, three, two, one, zero",
        ),
        ("Track ABC-1234", "Track A B C one two three four"),
        ("The value is -0.05", "The value is minus zero point zero five"),
        ("Meet on 03/04/25.", "Meet on three April two thousand twenty five."),
        (
            "Visit https://www.example.com/abc.",
            "Visit W W W dot example dot com slash abc.",
        ),
    ],
)
def test_boundaries_and_conservative_defaults(text, expected):
    assert normalise_text(text, "en") == expected


@pytest.mark.parametrize(
    "text",
    ["31/02/2025", "2025-13-04", "01-02/2025", "24:00", "12:75", "₹12,34", "12,34"],
)
def test_invalid_structured_values_remain_intact(text):
    assert normalise_text(text, "en") == text


@pytest.mark.parametrize(
    "text,lang",
    [
        ("२५", "hi"),
        ("২৫", "bn"),
        ("૨૫", "gu"),
        ("੨੫", "pa"),
        ("୨୫", "or"),
        ("೨೫", "kn"),
        ("൨൫", "ml"),
    ],
)
def test_native_numerals(text, lang):
    assert normalise_text(text, lang) == normalise_text("25", lang)


def test_rejects_invalid_language_and_type():
    with pytest.raises(ValueError, match="Unsupported Maya2 language"):
        normalise_text("25", "fr")
    with pytest.raises(TypeError, match="string"):
        normalise_text(None, "en")


@pytest.fixture(scope="module")
def upstream():
    source = os.environ.get("MAYA2_REFERENCE_SOURCE")
    if not source:
        pytest.skip("Set MAYA2_REFERENCE_SOURCE for differential normalization tests")
    path = Path(source) / "maya3" / "text"
    previous = {
        key: value
        for key, value in sys.modules.items()
        if key == "maya3" or key.startswith("maya3.")
    }
    package = types.ModuleType("maya3")
    package.__path__ = [str(path.parent)]
    sys.modules["maya3"] = package
    spec = importlib.util.spec_from_file_location(
        "maya3.text", path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        module.normalise_text("0", "en")
        yield module
    finally:
        for key in list(sys.modules):
            if key == "maya3" or key.startswith("maya3."):
                del sys.modules[key]
        sys.modules.update(previous)


@pytest.mark.parametrize("lang", LANGUAGES)
def test_cardinals_and_years_against_upstream(upstream, lang):
    reference_module = sys.modules["maya3.text.tn.normalize"]
    reference = reference_module.language_table(lang)
    language = _Language(lang)
    numbers = list(range(1000)) + [
        1000,
        1001,
        9999,
        10000,
        25000,
        99999,
        100000,
        100001,
        999999,
        1000000,
        10000001,
    ]
    numbers += random.Random(42).sample(range(1, 1000000000), 100)

    def fold(text):
        return (
            text.translate(reference_module._HINDI_NUKTA_FOLD) if lang == "hi" else text
        )

    for value in numbers:
        assert language.cardinal(value) == fold(reference.cardinal(value)), (
            lang,
            value,
        )
    for value in [1100, 1900, 1947, 1999, 2000, 2026, 2050]:
        assert language.year(value) == fold(reference.year(value)), (lang, value)


@pytest.mark.parametrize("lang", LANGUAGES)
def test_supported_formats_against_upstream(upstream, lang):
    cases = [
        "25",
        "1,250",
        "1,23,456",
        "0.05",
        "-12.50",
        "123.40",
        "0123",
        "50%",
        "₹1",
        "₹500",
        "₹95.41",
        "₹0.50",
        "₹1.01",
        "₹2.5",
        "$84.20",
        "€12",
        "£19.00",
        "₹1 lakh",
        "₹2 lakh",
        "₹1.5 lakh",
        "INR 1,250",
        "Rs. 25",
        "-₹50",
        "₹-50",
        "15/08/2025",
        "15-08-25",
        "2026-10-27",
        "15 August 2025",
        "August 15, 2025",
        "15/08/1947",
        "29/02/2024",
        "03/04/25",
        "9/9/99",
        "15 August",
        "15-Aug-2025",
        "August 15",
        "10:30",
        "14:30",
        "10:00 AM",
        "2 pm",
        "00:00",
        "1:01 pm",
        "12:00 AM",
        "12:00 PM",
        "00:30",
        "6.45 pm",
        "OTP 482913",
        "PIN 4532",
        "Call 98765 43210.",
        "Call +91 98765 43210.",
        "Account 00123456789",
        "Room 314",
        "code AB1234",
        "Email a.b@gmail.com",
        "Visit https://www.example.com/abc today",
        "Phone +919876543210",
        "Card 4111 2233 4455 6677",
        "PIN: 0042",
        "Order #12345",
        "Flight AI203",
        "Track ABC-1234",
        "a-b@example.com",
        "Visit sbi.co.in",
        "Version v4.5.2",
        "[laughs] 25 [sighs]",
        "I have 25 apples.",
    ]
    native_digits = (
        sys.modules["maya3.text.tn.normalize"].language_table(lang).native_digits
    )
    cases.append("".join(native_digits[int(c)] for c in "25"))
    table = sys.modules["maya3.text.tn.normalize"].language_table(lang)
    cases.append(f"15 {table.MONTHS[7]} 2025")
    if table.TIME_MARKERS:
        cases.extend(
            f"{clock} {marker}"
            for clock in ("1:00", "10:30")
            for marker in table.TIME_MARKERS
        )
    cases.append(table.PERIODS["morning"] + " 10:30 AM")
    cases.append("10:30 " + table.PERIODS["morning"])
    cases += [
        f"25 {word}"
        for word, currency in sys.modules["maya3.text.tn.normalize"]
        .language_table(lang)
        .CURRENCY_WORDS.items()
        if currency == "rupee"
    ]
    for text in cases:
        assert normalise_text(text, lang) == upstream.normalise_text(text, lang), (
            lang,
            text,
        )
