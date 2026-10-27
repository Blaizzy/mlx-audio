# Vendored from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya Research,
# licensed under the Apache License 2.0 (see LICENSE in this directory). Modified for mlx-audio: reformatted with black.
"""Native language tables recovered from MayaTeam/enterprise.

Source: 7384e91b02848ed791060c2eb368d52c9b92203c:gpu-origin/normalize_all.py
Only LangBase subclasses were extracted; the legacy parser was NOT copied.
The review branch also repairs singular units and Hindi irregular ordinals, and
adds explicit Hindi/Telugu month-spelling aliases for calendar validation.
These are inherited candidate lexicons, not independently native-speaker approved.
See docs/NORMALIZATION_REVIEW.md for provenance, limitations and release gates.
"""

if __package__:
    from .normalize import LangBase
else:
    from normalize import LangBase


class Hindi(LangBase):
    code = "hi"
    name = "Hindi"
    native_digits = "०१२३४५६७८९"
    SINGULAR_NOUNS = {"घंटे": "घंटा", "रुपये": "रुपया", "पैसे": "पैसा"}

    ONES = [
        "शून्य",
        "एक",
        "दो",
        "तीन",
        "चार",
        "पाँच",
        "छह",
        "सात",
        "आठ",
        "नौ",
        "दस",
        "ग्यारह",
        "बारह",
        "तेरह",
        "चौदह",
        "पंद्रह",
        "सोलह",
        "सत्रह",
        "अठारह",
        "उन्नीस",
        "बीस",
        "इक्कीस",
        "बाईस",
        "तेईस",
        "चौबीस",
        "पच्चीस",
        "छब्बीस",
        "सत्ताईस",
        "अट्ठाईस",
        "उनतीस",
        "तीस",
        "इकतीस",
        "बत्तीस",
        "तैंतीस",
        "चौंतीस",
        "पैंतीस",
        "छत्तीस",
        "सैंतीस",
        "अड़तीस",
        "उनतालीस",
        "चालीस",
        "इकतालीस",
        "बयालीस",
        "तैंतालीस",
        "चौवालीस",
        "पैंतालीस",
        "छियालीस",
        "सैंतालीस",
        "अड़तालीस",
        "उनचास",
        "पचास",
        "इक्यावन",
        "बावन",
        "तिरपन",
        "चौवन",
        "पचपन",
        "छप्पन",
        "सत्तावन",
        "अट्ठावन",
        "उनसठ",
        "साठ",
        "इकसठ",
        "बासठ",
        "तिरसठ",
        "चौंसठ",
        "पैंसठ",
        "छियासठ",
        "सड़सठ",
        "अड़सठ",
        "उनहत्तर",
        "सत्तर",
        "इकहत्तर",
        "बहत्तर",
        "तिहत्तर",
        "चौहत्तर",
        "पचहत्तर",
        "छिहत्तर",
        "सतहत्तर",
        "अठहत्तर",
        "उन्यासी",
        "अस्सी",
        "इक्यासी",
        "बयासी",
        "तिरासी",
        "चौरासी",
        "पचासी",
        "छियासी",
        "सत्तासी",
        "अठासी",
        "नवासी",
        "नब्बे",
        "इक्यानवे",
        "बानवे",
        "तिरानवे",
        "चौरानवे",
        "पंचानवे",
        "छियानवे",
        "सत्तानवे",
        "अट्ठानवे",
        "निन्यानवे",
    ]
    HUNDRED = "सौ"
    THOUSAND = "हज़ार"
    LAKH = "लाख"
    CRORE = "करोड़"

    LETTERS = {
        "A": "ए",
        "B": "बी",
        "C": "सी",
        "D": "डी",
        "E": "ई",
        "F": "एफ़",
        "G": "जी",
        "H": "एच",
        "I": "आई",
        "J": "जे",
        "K": "के",
        "L": "एल",
        "M": "एम",
        "N": "एन",
        "O": "ओ",
        "P": "पी",
        "Q": "क्यू",
        "R": "आर",
        "S": "एस",
        "T": "टी",
        "U": "यू",
        "V": "वी",
        "W": "डब्ल्यू",
        "X": "एक्स",
        "Y": "वाई",
        "Z": "ज़ेड",
    }

    MONTHS = [
        "जनवरी",
        "फ़रवरी",
        "मार्च",
        "अप्रैल",
        "मई",
        "जून",
        "जुलाई",
        "अगस्त",
        "सितंबर",
        "अक्टूबर",
        "नवंबर",
        "दिसंबर",
    ]
    MONTH_ALIASES = {"फरवरी": 2, "सितम्बर": 9, "नवम्बर": 11, "दिसम्बर": 12}

    SYM = {
        "rupee": "रुपये",
        "dollar": "डॉलर",
        "euro": "यूरो",
        "pound": "पाउंड",
        "cent": "सेंट",
        "paisa": "पैसे",
        "percent": "प्रतिशत",
        "point": "दशमलव",
        "minus": "माइनस",
        "plus": "प्लस",
        "equals": "बराबर",
        "times": "गुणा",
        "divide": "भाग",
        "squared": "का वर्ग",
        "sqrt": "वर्गमूल",
        "lt": "से कम",
        "gt": "से ज़्यादा",
        "plus_minus": "धन-ऋण",
        "at": "एट द रेट",
        "underscore": "अंडरस्कोर",
        "dot": "डॉट",
        "slash": "स्लैश",
        "hash": "हैश",
        "amp": "और",
        "half": "आधा",
        "quarter": "चौथाई",
        "three_quarters": "तीन चौथाई",
        "degree": "डिग्री",
        "celsius": "डिग्री सेल्सियस",
        "fahrenheit": "डिग्री फ़ारेनहाइट",
        "minute": "मिनट",
        "second": "सेकंड",
        "per": "प्रति",
    }

    TIME_ON_HOUR = "{h} बजे"
    TIME_HM = "{h} बजकर {m} मिनट"
    RANGE = "{a} से {b}"
    RANGE_TO = "{a} से {b} तक"
    RATIO = "{a} अनुपात {b}"
    POWER = "{base} की घात {exp}"
    FRACTION = "{num} बटा {den}"
    DECADE_SHORT = "{tens} के"
    DECADE_FULL = "{tens} का दशक"

    PERIODS = {
        "morning": "सुबह",
        "afternoon": "दोपहर",
        "evening": "शाम",
        "night": "रात",
    }

    TIME_MARKERS = ["बजे", "बजकर"]
    PERIOD_DETECT = {
        "सुबह": "morning",
        "दोपहर": "afternoon",
        "शाम": "evening",
        "रात": "night",
        "सवेरे": "morning",
    }
    YEAR_WORDS = ["साल", "वर्ष", "सन्", "सन"]
    ID_KEYWORDS = [
        "नंबर",
        "नम्बर",
        "खाता",
        "खाते",
        "कोड",
        "ऑर्डर",
        "फ़ोन",
        "फोन",
        "मोबाइल",
    ]
    LABEL_WORDS = ["अनुच्छेद", "धारा", "कमरा", "कक्षा", "वार्ड", "अध्याय", "भाग"]
    ID_AFTER_WORDS = ["खत्म", "डायल", "कॉल", "संपर्क"]
    NUMBER_WORD = "नंबर"
    ABBREV_UNITS = {"MINS": "मिनट", "HRS": "घंटे", "SECS": "सेकंड", "KMS": "किलोमीटर"}
    MINUTE_WORDS = ["मिनट"]
    SCALE_WORDS = {
        "करोड़": "crore",
        "लाख": "lakh",
        "हज़ार": "thousand",
        "हजार": "thousand",
        "अरब": "arab",
    }
    CURRENCY_WORDS = {"रुपये": "rupee", "रुपए": "rupee", "रुपया": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "पहला", 2: "दूसरा", 3: "तीसरा", 4: "चौथा", 6: "छठा"}
    ORDINAL_SUFFIX = "वाँ"
    ORDINAL_INPUT_SUFFIXES = ["वीं", "वाँ", "वां", "वें", "वे"]

    def _ordinal_compose(self, n, src_suffix):
        # 12वीं -> बारहवीं : cardinal + preserved suffix
        if src_suffix:
            if n in self.ORDINAL_EXCEPTIONS:
                # Irregular stems retain gender/case: 1वीं -> पहली, not एकवीं.
                ending = {"वीं": "ी", "वें": "े", "वे": "े"}.get(src_suffix, "ा")
                return self.ORDINAL_EXCEPTIONS[n][:-1] + ending
            return f"{self.cardinal(n)}{src_suffix}"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"


class Bengali(LangBase):
    code = "bn"
    name = "Bengali"
    native_digits = "০১২৩৪৫৬৭৮৯"

    ONES = [
        "শূন্য",
        "এক",
        "দুই",
        "তিন",
        "চার",
        "পাঁচ",
        "ছয়",
        "সাত",
        "আট",
        "নয়",
        "দশ",
        "এগারো",
        "বারো",
        "তেরো",
        "চৌদ্দ",
        "পনেরো",
        "ষোলো",
        "সতেরো",
        "আঠারো",
        "উনিশ",
        "বিশ",
        "একুশ",
        "বাইশ",
        "তেইশ",
        "চব্বিশ",
        "পঁচিশ",
        "ছাব্বিশ",
        "সাতাশ",
        "আঠাশ",
        "ঊনত্রিশ",
        "ত্রিশ",
        "একত্রিশ",
        "বত্রিশ",
        "তেত্রিশ",
        "চৌত্রিশ",
        "পঁয়ত্রিশ",
        "ছত্রিশ",
        "সাঁইত্রিশ",
        "আটত্রিশ",
        "ঊনচল্লিশ",
        "চল্লিশ",
        "একচল্লিশ",
        "বিয়াল্লিশ",
        "তেতাল্লিশ",
        "চুয়াল্লিশ",
        "পঁয়তাল্লিশ",
        "ছেচল্লিশ",
        "সাতচল্লিশ",
        "আটচল্লিশ",
        "ঊনপঞ্চাশ",
        "পঞ্চাশ",
        "একান্ন",
        "বায়ান্ন",
        "তিপ্পান্ন",
        "চুয়ান্ন",
        "পঞ্চান্ন",
        "ছাপ্পান্ন",
        "সাতান্ন",
        "আটান্ন",
        "ঊনষাট",
        "ষাট",
        "একষট্টি",
        "বাষট্টি",
        "তেষট্টি",
        "চৌষট্টি",
        "পঁয়ষট্টি",
        "ছেষট্টি",
        "সাতষট্টি",
        "আটষট্টি",
        "ঊনসত্তর",
        "সত্তর",
        "একাত্তর",
        "বাহাত্তর",
        "তিয়াত্তর",
        "চুয়াত্তর",
        "পঁচাত্তর",
        "ছিয়াত্তর",
        "সাতাত্তর",
        "আটাত্তর",
        "ঊনআশি",
        "আশি",
        "একাশি",
        "বিরাশি",
        "তিরাশি",
        "চুরাশি",
        "পঁচাশি",
        "ছিয়াশি",
        "সাতাশি",
        "আটাশি",
        "ঊননব্বই",
        "নব্বই",
        "একানব্বই",
        "বিরানব্বই",
        "তিরানব্বই",
        "চুরানব্বই",
        "পঁচানব্বই",
        "ছিয়ানব্বই",
        "সাতানব্বই",
        "আটানব্বই",
        "নিরানব্বই",
    ]
    HUNDRED = "শো"
    THOUSAND = "হাজার"
    LAKH = "লাখ"
    CRORE = "কোটি"

    LETTERS = {
        "A": "এ",
        "B": "বি",
        "C": "সি",
        "D": "ডি",
        "E": "ই",
        "F": "এফ",
        "G": "জি",
        "H": "এইচ",
        "I": "আই",
        "J": "জে",
        "K": "কে",
        "L": "এল",
        "M": "এম",
        "N": "এন",
        "O": "ও",
        "P": "পি",
        "Q": "কিউ",
        "R": "আর",
        "S": "এস",
        "T": "টি",
        "U": "ইউ",
        "V": "ভি",
        "W": "ডব্লিউ",
        "X": "এক্স",
        "Y": "ওয়াই",
        "Z": "জেড",
    }

    MONTHS = [
        "জানুয়ারি",
        "ফেব্রুয়ারি",
        "মার্চ",
        "এপ্রিল",
        "মে",
        "জুন",
        "জুলাই",
        "আগস্ট",
        "সেপ্টেম্বর",
        "অক্টোবর",
        "নভেম্বর",
        "ডিসেম্বর",
    ]

    SYM = {
        "rupee": "টাকা",
        "dollar": "ডলার",
        "euro": "ইউরো",
        "pound": "পাউন্ড",
        "cent": "সেন্ট",
        "percent": "শতাংশ",
        "point": "দশমিক",
        "minus": "মাইনাস",
        "plus": "প্লাস",
        "equals": "সমান",
        "times": "গুণ",
        "divide": "ভাগ",
        "squared": "এর বর্গ",
        "sqrt": "বর্গমূল",
        "lt": "এর চেয়ে কম",
        "gt": "এর চেয়ে বেশি",
        "plus_minus": "প্লাস-মাইনাস",
        "at": "অ্যাট দ্য রেট",
        "underscore": "আন্ডারস্কোর",
        "dot": "ডট",
        "slash": "স্ল্যাশ",
        "hash": "হ্যাশ",
        "amp": "অ্যান্ড",
        "half": "অর্ধেক",
        "quarter": "এক চতুর্থাংশ",
        "three_quarters": "তিন চতুর্থাংশ",
        "degree": "ডিগ্রি",
        "celsius": "ডিগ্রি সেলসিয়াস",
        "fahrenheit": "ডিগ্রি ফারেনহাইট",
        "minute": "মিনিট",
        "second": "সেকেন্ড",
    }

    TIME_ON_HOUR = "{h}টা"
    TIME_HM = "{h}টা বেজে {m} মিনিট"
    RANGE = "{a} থেকে {b}"
    RANGE_TO = "{a} থেকে {b} পর্যন্ত"
    RATIO = "{a} অনুপাত {b}"
    POWER = "{base} এর ঘাত {exp}"
    FRACTION = "{num} বাই {den}"
    DECADE_SHORT = "{tens} এর দশক"
    DECADE_FULL = "{tens} এর দশক"

    PERIODS = {
        "morning": "সকাল",
        "afternoon": "দুপুর",
        "evening": "সন্ধ্যা",
        "night": "রাত",
    }

    TIME_MARKERS = ["টা", "টায়", "টার", "বেজে", "বাজে", "বাজবে"]
    PERIOD_DETECT = {
        "সকাল": "morning",
        "সকালে": "morning",
        "ভোর": "morning",
        "ভোরে": "morning",
        "দুপুর": "afternoon",
        "দুপুরে": "afternoon",
        "বিকেল": "evening",
        "বিকেলে": "evening",
        "বিকাল": "evening",
        "সন্ধ্যা": "evening",
        "সন্ধ্যায়": "evening",
        "রাত": "night",
        "রাতে": "night",
    }
    YEAR_WORDS = ["সাল", "সালে", "সালের", "সন", "বছর", "বছরে"]
    ID_KEYWORDS = [
        "নম্বর",
        "নাম্বার",
        "অ্যাকাউন্ট",
        "একাউন্ট",
        "কোড",
        "অর্ডার",
        "ফোন",
        "মোবাইল",
        "ওটিপি",
        "পিন",
        "আইডি",
        "পাসওয়ার্ড",
        "ট্রেন",
        "এক্সপ্রেস",
        "পিএনআর",
        "কার্ড",
        "রেফারেন্স",
        "ট্রানজেকশন",
        "টিকিট",
        "অ্যাকাউন্টের",
        "নম্বরে",
        "আইডিতে",
        "ইউজার",
        "ইনস্টলমেন্ট",
        "কিস্তি",
        "ডিজিট",
        "অঙ্ক",
        "লেনদেন",
    ]
    ID_AFTER_WORDS = ["ডায়াল", "কল", "যোগাযোগ"]
    NUMBER_WORD = "নম্বর"
    ABBREV_UNITS = {
        "MINS": "মিনিট",
        "MIN": "মিনিট",
        "HRS": "ঘণ্টা",
        "HR": "ঘণ্টা",
        "SECS": "সেকেন্ড",
        "SEC": "সেকেন্ড",
        "KMS": "কিলোমিটার",
    }
    LABEL_WORDS = [
        "অনুচ্ছেদ",
        "ধারা",
        "রুম",
        "ঘর",
        "কক্ষ",
        "ওয়ার্ড",
        "অধ্যায়",
        "গেট",
        "কাউন্টার",
    ]
    MINUTE_WORDS = ["মিনিট"]
    SCALE_WORDS = {"কোটি": "crore", "লাখ": "lakh", "লক্ষ": "lakh", "হাজার": "thousand"}
    CURRENCY_WORDS = {
        "টাকা": "rupee",
        "টাকার": "rupee",
        "টাকায়": "rupee",
        "রুপি": "rupee",
    }

    ORDINAL_EXCEPTIONS = {
        1: "প্রথম",
        2: "দ্বিতীয়",
        3: "তৃতীয়",
        4: "চতুর্থ",
        5: "পঞ্চম",
        6: "ষষ্ঠ",
        7: "সপ্তম",
        8: "অষ্টম",
        9: "নবম",
        10: "দশম",
    }
    ORDINAL_SUFFIX = "তম"
    ORDINAL_INPUT_SUFFIXES = ["তম", "লা", "রা", "ঠা", "ই", "শে"]

    # Bengali date-of-month ordinals for ১লা / ২রা / ৩রা / ৪ঠা
    _DATE_ORDINALS = {1: "পহেলা", 2: "দোসরা", 3: "তেসরা", 4: "চৌঠা"}

    # ------------------------------------------------------------------
    # Hundreds fuse: এক+শো -> একশো, চার+শো -> চারশো (written solid).
    # ------------------------------------------------------------------
    def hundreds(self, h, rest):
        out = f"{self.ONES[h]}{self.HUNDRED}"
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def year(self, y):
        """1947 -> উনিশশো সাতচল্লিশ (fused শো); 2026 -> দুই হাজার ছাব্বিশ."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            out = f"{self.two_digit(hi)}{self.HUNDRED}"
            if lo:
                out += f" {self.two_digit(lo)}"
            return out
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # Date-style ordinals: ১লা -> পহেলা, ২রা -> দোসরা, ৪ঠা -> চৌঠা,
        # ৫ই -> পাঁচই, ১৯শে -> উনিশে. General: ১২তম -> বারোতম.
        if src_suffix in ("লা", "রা", "ঠা") and n in self._DATE_ORDINALS:
            return self._DATE_ORDINALS[n]
        if src_suffix == "ই":
            return f"{self.cardinal(n)}ই"
        if src_suffix == "শে":
            return f"{self.cardinal(n)}ে"
        if src_suffix == "তম":
            return f"{self.cardinal(n)}তম"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"

    # postpositions attach to noun-words with Bengali sandhi
    # (টাকা+এর->টাকার, শতাংশ+এ->শতাংশে, টা+মধ্যে->টার মধ্যে)
    _ATTACH_NOUNS = {
        "টাকা",
        "পয়সা",
        "ডলার",
        "শতাংশ",
        "কিলোমিটার",
        "কিলোগ্রাম",
        "মিটার",
        "লিটার",
        "গ্রাম",
        "ডিগ্রি",
        "মিনিট",
        "সেকেন্ড",
        "টা",
        "টো",
    }

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if last not in self._ATTACH_NOUNS:
            return None
        if t == last:
            return rep  # duplicate marker
        if t in ("এ", "তে", "য়"):  # locative
            if last.endswith(("া", "ো", "ে")):
                return rep + "য়"  # টাকায়, এগারোটায়
            return rep + "ে"  # শতাংশে, মিনিটে
        if t in ("এর", "র"):  # genitive
            if last.endswith(("া", "ো", "ে")):
                return rep + "র"  # টাকার, টার
            return rep + "ের"  # মিনিটের
        if t in ("মধ্যে", "নাগাদ", "থেকে", "পর্যন্ত", "দিয়ে"):
            g = rep + ("র" if last.endswith(("া", "ো", "ে")) else "ের")
            return g + " " + t  # এগারোটার মধ্যে
        return None


class Gujarati(LangBase):
    code = "gu"
    name = "Gujarati"
    native_digits = "૦૧૨૩૪૫૬૭૮૯"

    ONES = [
        "શૂન્ય",
        "એક",
        "બે",
        "ત્રણ",
        "ચાર",
        "પાંચ",
        "છ",
        "સાત",
        "આઠ",
        "નવ",
        "દસ",
        "અગિયાર",
        "બાર",
        "તેર",
        "ચૌદ",
        "પંદર",
        "સોળ",
        "સત્તર",
        "અઢાર",
        "ઓગણીસ",
        "વીસ",
        "એકવીસ",
        "બાવીસ",
        "ત્રેવીસ",
        "ચોવીસ",
        "પચ્ચીસ",
        "છવ્વીસ",
        "સત્તાવીસ",
        "અઠ્ઠાવીસ",
        "ઓગણત્રીસ",
        "ત્રીસ",
        "એકત્રીસ",
        "બત્રીસ",
        "તેત્રીસ",
        "ચોત્રીસ",
        "પાંત્રીસ",
        "છત્રીસ",
        "સાડત્રીસ",
        "આડત્રીસ",
        "ઓગણચાળીસ",
        "ચાળીસ",
        "એકતાળીસ",
        "બેતાળીસ",
        "તેંતાળીસ",
        "ચુંમાળીસ",
        "પિસ્તાળીસ",
        "છેંતાળીસ",
        "સુડતાળીસ",
        "અડતાળીસ",
        "ઓગણપચાસ",
        "પચાસ",
        "એકાવન",
        "બાવન",
        "ત્રેપન",
        "ચોપન",
        "પંચાવન",
        "છપ્પન",
        "સત્તાવન",
        "અઠ્ઠાવન",
        "ઓગણસાઠ",
        "સાઠ",
        "એકસઠ",
        "બાસઠ",
        "ત્રેસઠ",
        "ચોસઠ",
        "પાંસઠ",
        "છાસઠ",
        "સડસઠ",
        "અડસઠ",
        "અગણોસિત્તેર",
        "સિત્તેર",
        "એકોતેર",
        "બોતેર",
        "તોંતેર",
        "ચુંમોતેર",
        "પંચોતેર",
        "છોતેર",
        "સિત્તોતેર",
        "અઠ્ઠોતેર",
        "ઓગણએંસી",
        "એંસી",
        "એક્યાસી",
        "બ્યાસી",
        "ત્યાસી",
        "ચોરાસી",
        "પંચાસી",
        "છયાસી",
        "સત્યાસી",
        "અઠયાસી",
        "નેવ્યાસી",
        "નેવું",
        "એકાણું",
        "બાણું",
        "ત્રાણું",
        "ચોરાણું",
        "પંચાણું",
        "છન્નું",
        "સત્તાણું",
        "અઠ્ઠાણું",
        "નવ્વાણું",
    ]
    HUNDRED = "સો"
    THOUSAND = "હજાર"
    LAKH = "લાખ"
    CRORE = "કરોડ"

    # Gujarati hundreds are fused compounds (200 = બસો, not "બે સો").
    HUNDREDS_FUSED = {
        1: "એકસો",
        2: "બસો",
        3: "ત્રણસો",
        4: "ચારસો",
        5: "પાંચસો",
        6: "છસો",
        7: "સાતસો",
        8: "આઠસો",
        9: "નવસો",
    }

    LETTERS = {
        "A": "એ",
        "B": "બી",
        "C": "સી",
        "D": "ડી",
        "E": "ઈ",
        "F": "એફ",
        "G": "જી",
        "H": "એચ",
        "I": "આઈ",
        "J": "જે",
        "K": "કે",
        "L": "એલ",
        "M": "એમ",
        "N": "એન",
        "O": "ઓ",
        "P": "પી",
        "Q": "ક્યુ",
        "R": "આર",
        "S": "એસ",
        "T": "ટી",
        "U": "યુ",
        "V": "વી",
        "W": "ડબલ્યુ",
        "X": "એક્સ",
        "Y": "વાય",
        "Z": "ઝેડ",
    }

    MONTHS = [
        "જાન્યુઆરી",
        "ફેબ્રુઆરી",
        "માર્ચ",
        "એપ્રિલ",
        "મે",
        "જૂન",
        "જુલાઈ",
        "ઓગસ્ટ",
        "સપ્ટેમ્બર",
        "ઓક્ટોબર",
        "નવેમ્બર",
        "ડિસેમ્બર",
    ]

    SYM = {
        "rupee": "રૂપિયા",
        "dollar": "ડોલર",
        "euro": "યુરો",
        "pound": "પાઉન્ડ",
        "cent": "સેન્ટ",
        "percent": "ટકા",
        "point": "દશાંશ",
        "minus": "માઇનસ",
        "plus": "પ્લસ",
        "equals": "બરાબર",
        "times": "ગુણ્યા",
        "divide": "ભાગ્યા",
        "squared": "નો વર્ગ",
        "sqrt": "વર્ગમૂળ",
        "lt": "થી ઓછું",
        "gt": "થી વધારે",
        "plus_minus": "વત્તા-ઓછા",
        "at": "એટ ધ રેટ",
        "underscore": "અન્ડરસ્કોર",
        "dot": "ડોટ",
        "slash": "સ્લેશ",
        "hash": "હેશ",
        "amp": "અને",
        "half": "અડધો",
        "quarter": "પા",
        "three_quarters": "પોણો",
        "degree": "ડિગ્રી",
        "celsius": "ડિગ્રી સેલ્સિયસ",
        "fahrenheit": "ડિગ્રી ફેરનહીટ",
        "minute": "મિનિટ",
        "second": "સેકન્ડ",
    }

    TIME_ON_HOUR = "{h} વાગ્યે"
    TIME_HM = "{h} વાગીને {m} મિનિટ"
    RANGE = "{a} થી {b}"
    RANGE_TO = "{a} થી {b} સુધી"
    RATIO = "{a} ગુણોત્તર {b}"
    POWER = "{base} ની ઘાત {exp}"
    FRACTION = "{num} બટા {den}"
    DECADE_SHORT = "{tens} ના"
    DECADE_FULL = "{tens} નો દાયકો"

    PERIODS = {
        "morning": "સવારે",
        "afternoon": "બપોરે",
        "evening": "સાંજે",
        "night": "રાત્રે",
    }

    TIME_MARKERS = ["વાગ્યે", "વાગે", "વાગ્યા", "વાગ્યું", "વાગીને", "કલાકે"]
    PERIOD_DETECT = {
        "સવારે": "morning",
        "સવાર": "morning",
        "સવારના": "morning",
        "બપોરે": "afternoon",
        "બપોર": "afternoon",
        "બપોરના": "afternoon",
        "સાંજે": "evening",
        "સાંજ": "evening",
        "સાંજના": "evening",
        "રાત્રે": "night",
        "રાતે": "night",
        "રાત": "night",
        "રાતના": "night",
    }
    YEAR_WORDS = ["વર્ષ", "વર્ષે", "સાલ", "સન", "સને"]
    ID_KEYWORDS = [
        "નંબર",
        "નમ્બર",
        "ખાતા",
        "ખાતું",
        "ખાતાં",
        "કોડ",
        "ઓર્ડર",
        "ફોન",
        "મોબાઇલ",
        "મોબાઈલ",
        "એકાઉન્ટ",
        "એકાઉન્ટમાં",
        "એકાઉન્ટનો",
        "પિન",
        "આઈડી",
        "પાસવર્ડ",
        "ટ્રેન",
        "એક્સપ્રેસ",
        "પીએનઆર",
        "કાર્ડ",
        "રેફરન્સ",
        "ટ્રાન્ઝેક્શન",
        "ટિકિટ",
        "ઓટીપી",
        "નંબરમાં",
        "યુઝર",
        "હપ્તો",
        "અંક",
        "ડિજિટ",
    ]
    ID_AFTER_WORDS = ["ડાયલ", "કૉલ", "સંપર્ક"]
    NUMBER_WORD = "નંબર"
    ABBREV_UNITS = {
        "MINS": "મિનિટ",
        "MIN": "મિનિટ",
        "HRS": "કલાક",
        "HR": "કલાક",
        "SECS": "સેકન્ડ",
        "SEC": "સેકન્ડ",
        "KMS": "કિલોમીટર",
    }
    LABEL_WORDS = [
        "અનુચ્છેદ",
        "કલમ",
        "ઓરડો",
        "રૂમ",
        "વોર્ડ",
        "ધોરણ",
        "પ્રકરણ",
        "ભાગ",
        "વર્ગ",
    ]
    MINUTE_WORDS = ["મિનિટ", "મિનીટ"]
    SCALE_WORDS = {"કરોડ": "crore", "લાખ": "lakh", "હજાર": "thousand", "અબજ": "arab"}
    CURRENCY_WORDS = {
        "રૂપિયા": "rupee",
        "રૂપિયો": "rupee",
        "રૂપિયે": "rupee",
        "રૂ": "rupee",
    }

    ORDINAL_EXCEPTIONS = {1: "પહેલો", 2: "બીજો", 3: "ત્રીજો", 4: "ચોથો", 6: "છઠ્ઠો"}
    # Stems for gender/case agreement with a source suffix (1લી -> પહેલી).
    ORDINAL_STEMS = {1: "પહેલ", 2: "બીજ", 3: "ત્રીજ", 4: "ચોથ", 6: "છઠ્ઠ"}
    ORDINAL_SUFFIX = "મો"
    ORDINAL_INPUT_SUFFIXES = [
        "મી",
        "મો",
        "મું",
        "મા",
        "મે",
        "લી",
        "લો",
        "લું",
        "લા",
        "જી",
        "જો",
        "જું",
        "જા",
        "થી",
        "થો",
        "થું",
        "થા",
    ]

    # ------------------------------------------------------------------
    # Gujarati-specific composition
    # ------------------------------------------------------------------

    def hundreds(self, h, rest):
        """Fused hundreds: 250 -> 'બસો પચાસ', 147 -> 'એકસો સુડતાળીસ'."""
        out = self.HUNDREDS_FUSED[h]
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def year(self, y):
        """1100-1999: fused hundred style — 1947 -> 'ઓગણીસો સુડતાળીસ',
        1857 -> 'અઢારસો સત્તાવન'. A trailing સ merges with સો
        (ઓગણીસ + સો -> ઓગણીસો). Everything else: full cardinal."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            w = self.two_digit(hi)
            w = w[:-1] + self.HUNDRED if w.endswith("સ") else w + self.HUNDRED
            if lo:
                return f"{w} {self.two_digit(lo)}"
            return w
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # 1-4, 6 have suppletive stems: 1લી -> પહેલી, 2જી -> બીજી, 4થા -> ચોથા.
        if n in self.ORDINAL_STEMS:
            if src_suffix and len(src_suffix) > 1:
                return self.ORDINAL_STEMS[n] + src_suffix[1:]
            return self.ORDINAL_EXCEPTIONS[n]
        # Regular: cardinal + મ-suffix (12મી -> બારમી, 5મો -> પાંચમો).
        if src_suffix and src_suffix.startswith("મ"):
            return f"{self.cardinal(n)}{src_suffix}"
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"

    # Gujarati postpositions agglutinate directly onto the preceding word
    # (રૂપિયા + માં -> રૂપિયામાં). Oblique of the plural currency word before
    # a genitive: રૂપિયા stays (already the citation form here).
    _PP = {
        "માં",
        "માંથી",
        "નું",
        "નો",
        "ની",
        "ના",
        "ને",
        "થી",
        "સુધી",
        "પર",
        "પરથી",
        "વડે",
        "કરતાં",
        "જેટલા",
        "જેટલું",
    }

    # attach postpositions only to noun-words we insert (currency/unit/
    # percent), never to a bare cardinal (Terminal બે + પર stays "બે પર")
    _ATTACH_NOUNS = {
        "રૂપિયા",
        "પૈસા",
        "ડોલર",
        "ટકા",
        "ટકાવારી",
        "કિલોમીટર",
        "કિલોગ્રામ",
        "મીટર",
        "લિટર",
        "ગ્રામ",
        "ડિગ્રી",
        "કલાક",
        "મિનિટ",
        "સેકન્ડ",
    }

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if t == last and last in self._ATTACH_NOUNS:
            return rep
        if t in self._PP and last in self._ATTACH_NOUNS:
            return rep + t
        return None


class Kannada(LangBase):
    code = "kn"
    name = "Kannada"
    native_digits = "೦೧೨೩೪೫೬೭೮೯"

    ONES = [
        "ಸೊನ್ನೆ",
        "ಒಂದು",
        "ಎರಡು",
        "ಮೂರು",
        "ನಾಲ್ಕು",
        "ಐದು",
        "ಆರು",
        "ಏಳು",
        "ಎಂಟು",
        "ಒಂಬತ್ತು",
        "ಹತ್ತು",
        "ಹನ್ನೊಂದು",
        "ಹನ್ನೆರಡು",
        "ಹದಿಮೂರು",
        "ಹದಿನಾಲ್ಕು",
        "ಹದಿನೈದು",
        "ಹದಿನಾರು",
        "ಹದಿನೇಳು",
        "ಹದಿನೆಂಟು",
        "ಹತ್ತೊಂಬತ್ತು",
        "ಇಪ್ಪತ್ತು",
        "ಇಪ್ಪತ್ತೊಂದು",
        "ಇಪ್ಪತ್ತೆರಡು",
        "ಇಪ್ಪತ್ತಮೂರು",
        "ಇಪ್ಪತ್ತನಾಲ್ಕು",
        "ಇಪ್ಪತ್ತೈದು",
        "ಇಪ್ಪತ್ತಾರು",
        "ಇಪ್ಪತ್ತೇಳು",
        "ಇಪ್ಪತ್ತೆಂಟು",
        "ಇಪ್ಪತ್ತೊಂಬತ್ತು",
        "ಮೂವತ್ತು",
        "ಮೂವತ್ತೊಂದು",
        "ಮೂವತ್ತೆರಡು",
        "ಮೂವತ್ತಮೂರು",
        "ಮೂವತ್ತನಾಲ್ಕು",
        "ಮೂವತ್ತೈದು",
        "ಮೂವತ್ತಾರು",
        "ಮೂವತ್ತೇಳು",
        "ಮೂವತ್ತೆಂಟು",
        "ಮೂವತ್ತೊಂಬತ್ತು",
        "ನಲವತ್ತು",
        "ನಲವತ್ತೊಂದು",
        "ನಲವತ್ತೆರಡು",
        "ನಲವತ್ತಮೂರು",
        "ನಲವತ್ತನಾಲ್ಕು",
        "ನಲವತ್ತೈದು",
        "ನಲವತ್ತಾರು",
        "ನಲವತ್ತೇಳು",
        "ನಲವತ್ತೆಂಟು",
        "ನಲವತ್ತೊಂಬತ್ತು",
        "ಐವತ್ತು",
        "ಐವತ್ತೊಂದು",
        "ಐವತ್ತೆರಡು",
        "ಐವತ್ತಮೂರು",
        "ಐವತ್ತನಾಲ್ಕು",
        "ಐವತ್ತೈದು",
        "ಐವತ್ತಾರು",
        "ಐವತ್ತೇಳು",
        "ಐವತ್ತೆಂಟು",
        "ಐವತ್ತೊಂಬತ್ತು",
        "ಅರವತ್ತು",
        "ಅರವತ್ತೊಂದು",
        "ಅರವತ್ತೆರಡು",
        "ಅರವತ್ತಮೂರು",
        "ಅರವತ್ತನಾಲ್ಕು",
        "ಅರವತ್ತೈದು",
        "ಅರವತ್ತಾರು",
        "ಅರವತ್ತೇಳು",
        "ಅರವತ್ತೆಂಟು",
        "ಅರವತ್ತೊಂಬತ್ತು",
        "ಎಪ್ಪತ್ತು",
        "ಎಪ್ಪತ್ತೊಂದು",
        "ಎಪ್ಪತ್ತೆರಡು",
        "ಎಪ್ಪತ್ತಮೂರು",
        "ಎಪ್ಪತ್ತನಾಲ್ಕು",
        "ಎಪ್ಪತ್ತೈದು",
        "ಎಪ್ಪತ್ತಾರು",
        "ಎಪ್ಪತ್ತೇಳು",
        "ಎಪ್ಪತ್ತೆಂಟು",
        "ಎಪ್ಪತ್ತೊಂಬತ್ತು",
        "ಎಂಬತ್ತು",
        "ಎಂಬತ್ತೊಂದು",
        "ಎಂಬತ್ತೆರಡು",
        "ಎಂಬತ್ತಮೂರು",
        "ಎಂಬತ್ತನಾಲ್ಕು",
        "ಎಂಬತ್ತೈದು",
        "ಎಂಬತ್ತಾರು",
        "ಎಂಬತ್ತೇಳು",
        "ಎಂಬತ್ತೆಂಟು",
        "ಎಂಬತ್ತೊಂಬತ್ತು",
        "ತೊಂಬತ್ತು",
        "ತೊಂಬತ್ತೊಂದು",
        "ತೊಂಬತ್ತೆರಡು",
        "ತೊಂಬತ್ತಮೂರು",
        "ತೊಂಬತ್ತನಾಲ್ಕು",
        "ತೊಂಬತ್ತೈದು",
        "ತೊಂಬತ್ತಾರು",
        "ತೊಂಬತ್ತೇಳು",
        "ತೊಂಬತ್ತೆಂಟು",
        "ತೊಂಬತ್ತೊಂಬತ್ತು",
    ]
    HUNDRED = "ನೂರು"
    THOUSAND = "ಸಾವಿರ"
    LAKH = "ಲಕ್ಷ"
    CRORE = "ಕೋಟಿ"

    # Fused hundreds (standalone) and their combining (oblique) forms used
    # when a 0-99 remainder follows: 147 -> ನೂರ ನಲವತ್ತೇಳು.
    HUNDREDS = [
        "",
        "ನೂರು",
        "ಇನ್ನೂರು",
        "ಮುನ್ನೂರು",
        "ನಾನೂರು",
        "ಐನೂರು",
        "ಆರುನೂರು",
        "ಏಳುನೂರು",
        "ಎಂಟುನೂರು",
        "ಒಂಬೈನೂರು",
    ]
    HUNDREDS_COMB = [
        "",
        "ನೂರ",
        "ಇನ್ನೂರ",
        "ಮುನ್ನೂರ",
        "ನಾನೂರ",
        "ಐನೂರ",
        "ಆರುನೂರ",
        "ಏಳುನೂರ",
        "ಎಂಟುನೂರ",
        "ಒಂಬೈನೂರ",
    ]
    THOUSAND_COMB = "ಸಾವಿರದ"
    LAKH_COMB = "ಲಕ್ಷದ"

    LETTERS = {
        "A": "ಎ",
        "B": "ಬಿ",
        "C": "ಸಿ",
        "D": "ಡಿ",
        "E": "ಇ",
        "F": "ಎಫ್",
        "G": "ಜಿ",
        "H": "ಎಚ್",
        "I": "ಐ",
        "J": "ಜೆ",
        "K": "ಕೆ",
        "L": "ಎಲ್",
        "M": "ಎಂ",
        "N": "ಎನ್",
        "O": "ಒ",
        "P": "ಪಿ",
        "Q": "ಕ್ಯೂ",
        "R": "ಆರ್",
        "S": "ಎಸ್",
        "T": "ಟಿ",
        "U": "ಯು",
        "V": "ವಿ",
        "W": "ಡಬ್ಲ್ಯೂ",
        "X": "ಎಕ್ಸ್",
        "Y": "ವೈ",
        "Z": "ಝಡ್",
    }

    MONTHS = [
        "ಜನವರಿ",
        "ಫೆಬ್ರವರಿ",
        "ಮಾರ್ಚ್",
        "ಏಪ್ರಿಲ್",
        "ಮೇ",
        "ಜೂನ್",
        "ಜುಲೈ",
        "ಆಗಸ್ಟ್",
        "ಸೆಪ್ಟೆಂಬರ್",
        "ಅಕ್ಟೋಬರ್",
        "ನವೆಂಬರ್",
        "ಡಿಸೆಂಬರ್",
    ]

    SYM = {
        "rupee": "ರೂಪಾಯಿ",
        "dollar": "ಡಾಲರ್",
        "euro": "ಯೂರೋ",
        "pound": "ಪೌಂಡ್",
        "cent": "ಸೆಂಟ್",
        "percent": "ಪ್ರತಿಶತ",
        "point": "ದಶಮಾಂಶ",
        "minus": "ಮೈನಸ್",
        "plus": "ಪ್ಲಸ್",
        "equals": "ಸಮ",
        "times": "ಗುಣಿಸು",
        "divide": "ಭಾಗಿಸು",
        "squared": "ವರ್ಗ",
        "sqrt": "ವರ್ಗಮೂಲ",
        "lt": "ಗಿಂತ ಕಡಿಮೆ",
        "gt": "ಗಿಂತ ಹೆಚ್ಚು",
        "plus_minus": "ಪ್ಲಸ್ ಮೈನಸ್",
        "at": "ಅಟ್ ದಿ ರೇಟ್",
        "underscore": "ಅಂಡರ್‌ಸ್ಕೋರ್",
        "dot": "ಡಾಟ್",
        "slash": "ಸ್ಲ್ಯಾಶ್",
        "hash": "ಹ್ಯಾಶ್",
        "amp": "ಮತ್ತು",
        "half": "ಅರ್ಧ",
        "quarter": "ಕಾಲು",
        "three_quarters": "ಮುಕ್ಕಾಲು",
        "degree": "ಡಿಗ್ರಿ",
        "celsius": "ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್",
        "fahrenheit": "ಡಿಗ್ರಿ ಫ್ಯಾರನ್‌ಹೀಟ್",
        "minute": "ನಿಮಿಷ",
        "second": "ಸೆಕೆಂಡ್",
    }

    TIME_ON_HOUR = "{h} ಗಂಟೆ"
    TIME_HM = "{h} ಗಂಟೆ {m} ನಿಮಿಷ"
    RANGE = "{a} ರಿಂದ {b}"
    RANGE_TO = "{a} ರಿಂದ {b} ವರೆಗೆ"
    RATIO = "{a} ಅನುಪಾತ {b}"
    POWER = "{base} ಘಾತ {exp}"
    FRACTION = "{num} ಬೈ {den}"
    DECADE_SHORT = "{tens}ರ"  # decade() passes the ು-stripped stem
    DECADE_FULL = "{tens}ರ ದಶಕ"

    PERIODS = {
        "morning": "ಬೆಳಿಗ್ಗೆ",
        "afternoon": "ಮಧ್ಯಾಹ್ನ",
        "evening": "ಸಂಜೆ",
        "night": "ರಾತ್ರಿ",
    }

    TIME_MARKERS = ["ಗಂಟೆಗೆ", "ಗಂಟೆಯ", "ಗಂಟೆ"]
    MARKER_SUFFIXES = {"ಗಂಟೆಗೆ": "ಗೆ"}
    PERIOD_DETECT = {
        "ಬೆಳಿಗ್ಗೆ": "morning",
        "ಬೆಳಗ್ಗೆ": "morning",
        "ಮುಂಜಾನೆ": "morning",
        "ಮಧ್ಯಾಹ್ನ": "afternoon",
        "ಸಂಜೆ": "evening",
        "ರಾತ್ರಿ": "night",
    }
    YEAR_WORDS = ["ವರ್ಷ", "ಇಸವಿ", "ಸಾಲಿನ", "ಸಾಲು", "ಸನ್"]
    ID_KEYWORDS = [
        "ನಂಬರ್",
        "ಸಂಖ್ಯೆ",
        "ಖಾತೆ",
        "ಕೋಡ್",
        "ಆರ್ಡರ್",
        "ಫೋನ್",
        "ಮೊಬೈಲ್",
        "ಐಡಿ",
        "ಪಿನ್",
        "ಪಾಸ್‌ವರ್ಡ್",
        "ರೈಲು",
        "ಎಕ್ಸ್‌ಪ್ರೆಸ್",
        "ಪಿಎನ್‌ಆರ್",
        "ಕಾರ್ಡ್",
        "ರೆಫರೆನ್ಸ್",
        "ಟ್ರಾನ್ಸಾಕ್ಷನ್",
        "ಟಿಕೆಟ್",
        "ಓಟಿಪಿ",
        "ಅಕೌಂಟ್",
        "ಅಂಕಿ",
        "ಬಳಕೆದಾರ",
        "ಕಂತು",
        "ವಹಿವಾಟು",
    ]
    ID_AFTER_WORDS = ["ಡಯಲ್", "ಕರೆ", "ಸಂಪರ್ಕ"]
    NUMBER_WORD = "ನಂಬರ್"
    ABBREV_UNITS = {
        "MINS": "ನಿಮಿಷ",
        "MIN": "ನಿಮಿಷ",
        "HRS": "ಗಂಟೆ",
        "HR": "ಗಂಟೆ",
        "SECS": "ಸೆಕೆಂಡ್",
        "SEC": "ಸೆಕೆಂಡ್",
        "KMS": "ಕಿಲೋಮೀಟರ್",
    }
    LABEL_WORDS = ["ವಿಧಿ", "ಕಲಂ", "ಕೊಠಡಿ", "ವಾರ್ಡ್", "ಅಧ್ಯಾಯ", "ಭಾಗ", "ತರಗತಿ"]
    MINUTE_WORDS = ["ನಿಮಿಷ"]
    SCALE_WORDS = {
        "ಕೋಟಿ": "crore",
        "ಲಕ್ಷ": "lakh",
        "ಲಕ್ಷದ": "lakh",
        "ಸಾವಿರ": "thousand",
        "ಸಾವಿರದ": "thousand",
    }
    CURRENCY_WORDS = {
        "ರೂಪಾಯಿ": "rupee",
        "ರುಪಾಯಿ": "rupee",
        "ರೂ.": "rupee",
        "ರೂ": "rupee",
    }

    ORDINAL_EXCEPTIONS = {1: "ಮೊದಲನೇ"}
    ORDINAL_SUFFIX = "ನೇ"
    ORDINAL_INPUT_SUFFIXES = ["ನೆಯ", "ನೇ", "ನೆ"]

    # ==================================================================
    # Dravidian composition — fused hundreds/thousands with sandhi.
    # ==================================================================

    def hundreds(self, h, rest):
        if rest:
            return f"{self.HUNDREDS_COMB[h]} {self.two_digit(rest)}"
        return self.HUNDREDS[h]

    def under_thousand(self, n):
        if n < 100:
            return self.two_digit(n)
        return self.hundreds(n // 100, n % 100)

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 1000:
            return self.under_thousand(n)
        parts = []
        crore, n = divmod(n, 10**7)
        lakh, n = divmod(n, 10**5)
        thousand, n = divmod(n, 1000)
        if crore:
            # ಕೋಟಿ stays plain even when a remainder follows.
            parts.append(f"{self.cardinal(crore)} {self.CRORE}")
        if lakh:
            word = self.LAKH_COMB if (thousand or n) else self.LAKH
            parts.append(f"{self.two_digit(lakh)} {word}")
        if thousand:
            word = self.THOUSAND_COMB if n else self.THOUSAND
            if thousand == 1 and not (crore or lakh):
                # 1000 -> ಸಾವಿರ; 1947 -> ಸಾವಿರದ ಒಂಬೈನೂರ ನಲವತ್ತೇಳು (no ಒಂದು)
                parts.append(word)
            else:
                parts.append(f"{self.under_thousand(thousand)} {word}")
        if n:
            parts.append(self.under_thousand(n))
        return " ".join(parts)

    def year(self, y):
        """Kannada reads years as full cardinals (news/TTS style):
        1947 -> ಸಾವಿರದ ಒಂಬೈನೂರ ನಲವತ್ತೇಳು, 2026 -> ಎರಡು ಸಾವಿರದ ಇಪ್ಪತ್ತಾರು."""
        return self.cardinal(y)

    def decade(self, tens, full):
        # Genitive stem: final ು drops before -ರ (ತೊಂಬತ್ತು -> ತೊಂಬತ್ತರ).
        w = self.two_digit(tens)
        stem = w[:-1] if w.endswith("ು") else w
        tpl = self.DECADE_FULL if full else self.DECADE_SHORT
        return tpl.format(tens=stem)

    def _ordinal_compose(self, n, src_suffix):
        # ಎರಡು -> ಎರಡನೇ : final ು of the cardinal drops before ನೇ.
        base = self.cardinal(n)
        if base.endswith("ು"):
            base = base[:-1]
        suffix = (
            src_suffix
            if src_suffix in self.ORDINAL_INPUT_SUFFIXES
            else self.ORDINAL_SUFFIX
        )
        return f"{base}{suffix}"

    # Kannada case markers attach with oblique on ು-final words:
    # ಐದು+ಇಂದ->ಐದರಿಂದ, ಮೂರು+ಗೆ->ಮೂರಕ್ಕೆ, ಒಂಬತ್ತು+ನಲ್ಲಿ->ಒಂಬತ್ತರಲ್ಲಿ;
    # consonant/ೆ-final just attach (ನಿಮಿಷ+ಕ್ಕೆ->ನಿಮಿಷಕ್ಕೆ, ಸೊನ್ನೆ+ಗೆ->ಸೊನ್ನೆಗೆ)
    _PP = {
        "ಗೆ",
        "ಕ್ಕೆ",
        "ಇಂದ",
        "ದಿಂದ",
        "ರಿಂದ",
        "ನಲ್ಲಿ",
        "ಅಲ್ಲಿ",
        "ರಲ್ಲಿ",
        "ವರೆಗೆ",
        "ಗಾಗಿ",
        "ಕ್ಕಾಗಿ",
    }

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if not last or not ("\u0c80" <= last[-1] <= "\u0cff"):
            return None
        if t == last:
            return rep
        if t not in self._PP:
            return None
        base = rep[: len(rep) - len(last)]
        dative = t in ("ಗೆ", "ಕ್ಕೆ")
        ablative = t in ("ಇಂದ", "ದಿಂದ", "ರಿಂದ")
        locative = t in ("ನಲ್ಲಿ", "ಅಲ್ಲಿ", "ರಲ್ಲಿ")
        if last.endswith("\u0cc1"):  # ು-final -> drop, oblique -ರ/-ಕ್ಕೆ
            stem = last[:-1]
            suf = (
                "ಕ್ಕೆ" if dative else "ರಿಂದ" if ablative else "ರಲ್ಲಿ" if locative else t
            )
            return base + stem + suf
        if last.endswith(("\u0cc6", "\u0cbf", "\u0cc0")):  # ೆ/ಿ/ೀ-final
            suf = "ಗೆ" if dative else "ಯಿಂದ" if ablative else "ಯಲ್ಲಿ" if locative else t
            return base + last + suf
        # consonant/inherent-a-final -> -ಕ್ಕೆ/-ದಿಂದ/-ದಲ್ಲಿ
        suf = "ಕ್ಕೆ" if dative else "ದಿಂದ" if ablative else "ದಲ್ಲಿ" if locative else t
        return base + last + suf


class Malayalam(LangBase):
    code = "ml"
    name = "Malayalam"
    native_digits = "൦൧൨൩൪൫൬൭൮൯"
    COUNT_ONE = "ഒരു"

    ONES = [
        "പൂജ്യം",
        "ഒന്ന്",
        "രണ്ട്",
        "മൂന്ന്",
        "നാല്",
        "അഞ്ച്",
        "ആറ്",
        "ഏഴ്",
        "എട്ട്",
        "ഒമ്പത്",
        "പത്ത്",
        "പതിനൊന്ന്",
        "പന്ത്രണ്ട്",
        "പതിമൂന്ന്",
        "പതിനാല്",
        "പതിനഞ്ച്",
        "പതിനാറ്",
        "പതിനേഴ്",
        "പതിനെട്ട്",
        "പത്തൊമ്പത്",
        "ഇരുപത്",
        "ഇരുപത്തിയൊന്ന്",
        "ഇരുപത്തിരണ്ട്",
        "ഇരുപത്തിമൂന്ന്",
        "ഇരുപത്തിനാല്",
        "ഇരുപത്തിയഞ്ച്",
        "ഇരുപത്തിയാറ്",
        "ഇരുപത്തിയേഴ്",
        "ഇരുപത്തിയെട്ട്",
        "ഇരുപത്തിയൊമ്പത്",
        "മുപ്പത്",
        "മുപ്പത്തിയൊന്ന്",
        "മുപ്പത്തിരണ്ട്",
        "മുപ്പത്തിമൂന്ന്",
        "മുപ്പത്തിനാല്",
        "മുപ്പത്തിയഞ്ച്",
        "മുപ്പത്തിയാറ്",
        "മുപ്പത്തിയേഴ്",
        "മുപ്പത്തിയെട്ട്",
        "മുപ്പത്തിയൊമ്പത്",
        "നാൽപ്പത്",
        "നാൽപ്പത്തിയൊന്ന്",
        "നാൽപ്പത്തിരണ്ട്",
        "നാൽപ്പത്തിമൂന്ന്",
        "നാൽപ്പത്തിനാല്",
        "നാൽപ്പത്തിയഞ്ച്",
        "നാൽപ്പത്തിയാറ്",
        "നാൽപ്പത്തിയേഴ്",
        "നാൽപ്പത്തിയെട്ട്",
        "നാൽപ്പത്തിയൊമ്പത്",
        "അമ്പത്",
        "അമ്പത്തിയൊന്ന്",
        "അമ്പത്തിരണ്ട്",
        "അമ്പത്തിമൂന്ന്",
        "അമ്പത്തിനാല്",
        "അമ്പത്തിയഞ്ച്",
        "അമ്പത്തിയാറ്",
        "അമ്പത്തിയേഴ്",
        "അമ്പത്തിയെട്ട്",
        "അമ്പത്തിയൊമ്പത്",
        "അറുപത്",
        "അറുപത്തിയൊന്ന്",
        "അറുപത്തിരണ്ട്",
        "അറുപത്തിമൂന്ന്",
        "അറുപത്തിനാല്",
        "അറുപത്തിയഞ്ച്",
        "അറുപത്തിയാറ്",
        "അറുപത്തിയേഴ്",
        "അറുപത്തിയെട്ട്",
        "അറുപത്തിയൊമ്പത്",
        "എഴുപത്",
        "എഴുപത്തിയൊന്ന്",
        "എഴുപത്തിരണ്ട്",
        "എഴുപത്തിമൂന്ന്",
        "എഴുപത്തിനാല്",
        "എഴുപത്തിയഞ്ച്",
        "എഴുപത്തിയാറ്",
        "എഴുപത്തിയേഴ്",
        "എഴുപത്തിയെട്ട്",
        "എഴുപത്തിയൊമ്പത്",
        "എൺപത്",
        "എൺപത്തിയൊന്ന്",
        "എൺപത്തിരണ്ട്",
        "എൺപത്തിമൂന്ന്",
        "എൺപത്തിനാല്",
        "എൺപത്തിയഞ്ച്",
        "എൺപത്തിയാറ്",
        "എൺപത്തിയേഴ്",
        "എൺപത്തിയെട്ട്",
        "എൺപത്തിയൊമ്പത്",
        "തൊണ്ണൂറ്",
        "തൊണ്ണൂറ്റിയൊന്ന്",
        "തൊണ്ണൂറ്റിരണ്ട്",
        "തൊണ്ണൂറ്റിമൂന്ന്",
        "തൊണ്ണൂറ്റിനാല്",
        "തൊണ്ണൂറ്റിയഞ്ച്",
        "തൊണ്ണൂറ്റിയാറ്",
        "തൊണ്ണൂറ്റിയേഴ്",
        "തൊണ്ണൂറ്റിയെട്ട്",
        "തൊണ്ണൂറ്റിയൊമ്പത്",
    ]
    HUNDRED = "നൂറ്"  # not used in composition (Dravidian override below)
    THOUSAND = "ആയിരം"
    LAKH = "ലക്ഷം"
    CRORE = "കോടി"

    # Fused hundreds: standalone and combining (before a remainder).
    _H_STAND = {
        1: "നൂറ്",
        2: "ഇരുനൂറ്",
        3: "മുന്നൂറ്",
        4: "നാനൂറ്",
        5: "അഞ്ഞൂറ്",
        6: "അറുനൂറ്",
        7: "എഴുനൂറ്",
        8: "എണ്ണൂറ്",
        9: "തൊള്ളായിരം",
    }
    _H_COMB = {
        1: "നൂറ്റി",
        2: "ഇരുനൂറ്റി",
        3: "മുന്നൂറ്റി",
        4: "നാനൂറ്റി",
        5: "അഞ്ഞൂറ്റി",
        6: "അറുനൂറ്റി",
        7: "എഴുനൂറ്റി",
        8: "എണ്ണൂറ്റി",
        9: "തൊള്ളായിരത്തി",
    }

    # Fused thousands for multipliers 1..10; others via sandhi (see _thousand).
    _TH_STAND = {
        1: "ആയിരം",
        2: "രണ്ടായിരം",
        3: "മൂവായിരം",
        4: "നാലായിരം",
        5: "അയ്യായിരം",
        6: "ആറായിരം",
        7: "ഏഴായിരം",
        8: "എണ്ണായിരം",
        9: "ഒമ്പതിനായിരം",
        10: "പതിനായിരം",
    }

    _ONE_MULT = "ഒരു"  # multiplier word for 1 lakh / 1 crore

    LETTERS = {
        "A": "എ",
        "B": "ബി",
        "C": "സി",
        "D": "ഡി",
        "E": "ഇ",
        "F": "എഫ്",
        "G": "ജി",
        "H": "എച്ച്",
        "I": "ഐ",
        "J": "ജെ",
        "K": "കെ",
        "L": "എൽ",
        "M": "എം",
        "N": "എൻ",
        "O": "ഒ",
        "P": "പി",
        "Q": "ക്യു",
        "R": "ആർ",
        "S": "എസ്",
        "T": "ടി",
        "U": "യു",
        "V": "വി",
        "W": "ഡബ്ല്യു",
        "X": "എക്സ്",
        "Y": "വൈ",
        "Z": "ഇസഡ്",
    }

    MONTHS = [
        "ജനുവരി",
        "ഫെബ്രുവരി",
        "മാർച്ച്",
        "ഏപ്രിൽ",
        "മേയ്",
        "ജൂൺ",
        "ജൂലൈ",
        "ഓഗസ്റ്റ്",
        "സെപ്റ്റംബർ",
        "ഒക്ടോബർ",
        "നവംബർ",
        "ഡിസംബർ",
    ]

    SYM = {
        "rupee": "രൂപ",
        "dollar": "ഡോളർ",
        "euro": "യൂറോ",
        "pound": "പൗണ്ട്",
        "cent": "സെന്റ്",
        "percent": "ശതമാനം",
        "point": "ദശാംശം",
        "minus": "മൈനസ്",
        "plus": "പ്ലസ്",
        "equals": "സമം",
        "times": "ഗുണം",
        "divide": "ഹരണം",
        "squared": "വർഗം",
        "sqrt": "വർഗമൂലം",
        "lt": "നെക്കാൾ കുറവ്",
        "gt": "നെക്കാൾ കൂടുതൽ",
        "plus_minus": "പ്ലസ്-മൈനസ്",
        "at": "അറ്റ് ദ റേറ്റ്",
        "underscore": "അണ്ടർസ്കോർ",
        "dot": "ഡോട്ട്",
        "slash": "സ്ലാഷ്",
        "hash": "ഹാഷ്",
        "amp": "ആൻഡ്",
        "half": "അര",
        "quarter": "കാൽ",
        "three_quarters": "മുക്കാൽ",
        "degree": "ഡിഗ്രി",
        "celsius": "ഡിഗ്രി സെൽഷ്യസ്",
        "fahrenheit": "ഡിഗ്രി ഫാരൻഹീറ്റ്",
        "minute": "മിനിറ്റ്",
        "second": "സെക്കൻഡ്",
    }

    TIME_ON_HOUR = "{h} മണി"
    TIME_HM = "{h} മണി {m} മിനിറ്റ്"
    RANGE = "{a} മുതൽ {b}"
    RANGE_TO = "{a} മുതൽ {b} വരെ"
    RATIO = "{a} അനുപാതം {b}"
    POWER = "{base} ഘാതം {exp}"
    FRACTION = "{num} ബൈ {den}"
    DECADE_SHORT = "{tens}കൾ"
    DECADE_FULL = "{tens}കളുടെ ദശകം"

    PERIODS = {
        "morning": "രാവിലെ",
        "afternoon": "ഉച്ചയ്ക്ക്",
        "evening": "വൈകുന്നേരം",
        "night": "രാത്രി",
    }

    TIME_MARKERS = ["മണി", "മണിക്ക്", "മണിയോടെ"]
    MARKER_SUFFIXES = {"മണിക്ക്": "ന്", "മണിയോടെ": "ഓടെ"}
    PERIOD_DETECT = {
        "രാവിലെ": "morning",
        "കാലത്ത്": "morning",
        "ഉച്ചയ്ക്ക്": "afternoon",
        "ഉച്ചക്ക്": "afternoon",
        "ഉച്ച": "afternoon",
        "ഉച്ചകഴിഞ്ഞ്": "afternoon",
        "വൈകുന്നേരം": "evening",
        "വൈകിട്ട്": "evening",
        "സന്ധ്യക്ക്": "evening",
        "രാത്രി": "night",
    }
    YEAR_WORDS = ["വർഷം", "വർഷത്തിൽ", "വർഷത്തെ", "കൊല്ലം", "ആണ്ട്"]
    ID_KEYWORDS = [
        "നമ്പർ",
        "നമ്പര്",
        "നമ്പറിൽ",
        "അക്കൗണ്ട്",
        "കോഡ്",
        "ഓർഡർ",
        "ഫോൺ",
        "മൊബൈൽ",
        "ഒടിപി",
        "പിൻ",
        "ഐഡി",
        "പാസ്‌വേഡ്",
        "ട്രെയിൻ",
        "എക്സ്പ്രസ്",
        "പിഎൻആർ",
        "കാർഡ്",
        "റഫറൻസ്",
        "ട്രാൻസാക്ഷൻ",
        "ടിക്കറ്റ്",
        "അക്കൗണ്ടിൽ",
        "നമ്പറിലേക്ക്",
        "യൂസർ",
        "തവണ",
        "അക്കം",
        "ഇടപാട്",
    ]
    NUMBER_WORD = "നമ്പർ"
    ABBREV_UNITS = {
        "MINS": "മിനിറ്റ്",
        "MIN": "മിനിറ്റ്",
        "HRS": "മണിക്കൂർ",
        "HR": "മണിക്കൂർ",
        "SECS": "സെക്കൻഡ്",
        "SEC": "സെക്കൻഡ്",
        "KMS": "കിലോമീറ്റർ",
    }
    LABEL_WORDS = [
        "അനുച്ഛേദം",
        "വകുപ്പ്",
        "മുറി",
        "റൂം",
        "വാർഡ്",
        "ക്ലാസ്",
        "അധ്യായം",
        "ഭാഗം",
        "ഗേറ്റ്",
        "കൗണ്ടർ",
    ]
    MINUTE_WORDS = ["മിനിറ്റ്", "മിനുട്ട്"]
    SCALE_WORDS = {
        "കോടി": "crore",
        "കോടിയുടെ": "crore",
        "ലക്ഷം": "lakh",
        "ലക്ഷത്തി": "lakh",
        "ലക്ഷത്തിന്റെ": "lakh",
        "ആയിരം": "thousand",
        "ആയിരത്തി": "thousand",
    }
    CURRENCY_WORDS = {
        "രൂപ": "rupee",
        "രൂപാ": "rupee",
        "രൂപയുടെ": "rupee",
        "രൂപയ്ക്ക്": "rupee",
    }

    ORDINAL_EXCEPTIONS = {}
    ORDINAL_SUFFIX = "ാം"  # ഒന്ന് -> ഒന്നാം, പത്ത് -> പത്താം
    ORDINAL_INPUT_SUFFIXES = ["ാം", "ാമത്തെ", "ാമത്"]

    _VIRAMA = "്"  # ്
    _ANUSVARA = "ം"  # ം

    # ------------------------------------------------------------------
    # Dravidian composition overrides (fused hundreds/thousands + sandhi)
    # ------------------------------------------------------------------

    def hundreds(self, h, rest):
        if rest:
            return f"{self._H_COMB[h]} {self.two_digit(rest)}"
        return self._H_STAND[h]

    def _thousand(self, m, combining):
        """Fused '<m> thousand' word for m in 1..99."""
        if m in self._TH_STAND:
            word = self._TH_STAND[m]
        else:
            base = self.ONES[m]
            if base.endswith("ത" + self._VIRAMA):
                # ends in ...ത് (round tens, -ഒമ്പത്): ഇരുപത് -> ഇരുപതിനായിരം
                word = base[:-1] + "ിനായിരം"
            elif base.endswith(self._VIRAMA):
                # virama + ആ -> ാ : അമ്പത്തിനാല് -> അമ്പത്തിനാലായിരം,
                # തൊണ്ണൂറ് -> തൊണ്ണൂറായിരം
                word = base[:-1] + "ായിരം"
            else:
                word = base + " " + self.THOUSAND
        if combining:
            # ം -> ത്തി : ആയിരം -> ആയിരത്തി
            return word[:-1] + "ത്തി"
        return word

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 100:
            return self.ONES[n]
        if n < 1000:
            return self.hundreds(n // 100, n % 100)
        parts = []
        crore, n = divmod(n, 10**7)
        lakh, n = divmod(n, 10**5)
        thousand, n = divmod(n, 1000)
        if crore:
            mult = self._ONE_MULT if crore == 1 else self.cardinal(crore)
            parts.append(f"{mult} {self.CRORE}")
        if lakh:
            word = self.LAKH[:-1] + "ത്തി" if (thousand or n) else self.LAKH
            mult = self._ONE_MULT if lakh == 1 else self.two_digit(lakh)
            parts.append(f"{mult} {word}")
        if thousand:
            parts.append(self._thousand(thousand, combining=bool(n)))
        if n:
            parts.append(self.under_thousand(n))
        return " ".join(parts)

    def year(self, y):
        """Malayalam reads years as full cardinals (news/TTS convention):
        1947 -> ആയിരത്തി തൊള്ളായിരത്തി നാൽപ്പത്തിയേഴ്."""
        return self.cardinal(y)

    def decade(self, tens, full):
        # തൊണ്ണൂറ് + കൾ -> തൊണ്ണൂറുകൾ (final virama -> ു before കൾ)
        stem = self.two_digit(tens)
        if stem.endswith(self._VIRAMA):
            stem = stem[:-1] + "ു"
        tpl = self.DECADE_FULL if full else self.DECADE_SHORT
        return tpl.format(tens=stem)

    def _ordinal_compose(self, n, src_suffix):
        # രണ്ട് -> രണ്ടാം : strip final virama/anusvara, then attach suffix.
        stem = self.cardinal(n)
        if stem.endswith(self._VIRAMA) or stem.endswith(self._ANUSVARA):
            stem = stem[:-1]
        suffix = (
            src_suffix
            if src_suffix in self.ORDINAL_INPUT_SUFFIXES
            else self.ORDINAL_SUFFIX
        )
        return f"{stem}{suffix}"

    # Malayalam case markers (often hyphen-attached in source) attach with
    # sandhi: രൂപ+ന്റെ->രൂപയുടെ, ആറ്+ൽ->ആറിൽ, രൂപ+ൽ->രൂപയിൽ
    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if not last or not ("ഀ" <= last[-1] <= "ൿ"):
            return None
        base = rep[: len(rep) - len(last)]
        virama = last.endswith("്")
        anusvara = last.endswith("ം")
        ivow = last[-1] in ("ി", "ീ")
        if t in ("ന്റെ", "ുടെ", "യുടെ"):
            if virama:
                return base + last[:-1] + "ിന്റെ"
            if anusvara:
                return base + last[:-1] + "ത്തിന്റെ"
            return base + last + "യുടെ"
        if t in ("ൽ", "ിൽ", "ലെ", "ിലെ"):
            end = "ിലെ" if t in ("ലെ", "ിലെ") else "ിൽ"
            if virama:
                return base + last[:-1] + end
            if anusvara:
                return base + last[:-1] + "ത്ത" + end
            return base + last + "യ" + end
        if t in ("ലേക്ക്", "ിലേക്ക്"):
            if virama:
                return base + last[:-1] + "ിലേക്ക്"
            if anusvara:
                return base + last[:-1] + "ത്തിലേക്ക്"
            return base + last + "യിലേക്ക്"
        if t in ("ന്", "ിന്", "ക്ക്", "ിനാണ്"):
            tail = "ിനാണ്" if t == "ിനാണ്" else "ിന്"
            if virama:
                return base + last[:-1] + tail
            if anusvara:
                return base + last[:-1] + "ത്ത" + tail
            if ivow:
                return base + last + "ക്ക്"
            return base + last + "യ്ക്ക്"
        return None


class Marathi(LangBase):
    code = "mr"
    name = "Marathi"
    native_digits = "०१२३४५६७८९"
    SINGULAR_NOUNS = {"रुपये": "रुपया", "मिनिटे": "मिनिट"}

    ONES = [
        "शून्य",
        "एक",
        "दोन",
        "तीन",
        "चार",
        "पाच",
        "सहा",
        "सात",
        "आठ",
        "नऊ",
        "दहा",
        "अकरा",
        "बारा",
        "तेरा",
        "चौदा",
        "पंधरा",
        "सोळा",
        "सतरा",
        "अठरा",
        "एकोणीस",
        "वीस",
        "एकवीस",
        "बावीस",
        "तेवीस",
        "चोवीस",
        "पंचवीस",
        "सव्वीस",
        "सत्तावीस",
        "अठ्ठावीस",
        "एकोणतीस",
        "तीस",
        "एकतीस",
        "बत्तीस",
        "तेहतीस",
        "चौतीस",
        "पस्तीस",
        "छत्तीस",
        "सदतीस",
        "अडतीस",
        "एकोणचाळीस",
        "चाळीस",
        "एक्केचाळीस",
        "बेचाळीस",
        "त्रेचाळीस",
        "चव्वेचाळीस",
        "पंचेचाळीस",
        "सेहेचाळीस",
        "सत्तेचाळीस",
        "अठ्ठेचाळीस",
        "एकोणपन्नास",
        "पन्नास",
        "एक्कावन्न",
        "बावन्न",
        "त्रेपन्न",
        "चोपन्न",
        "पंचावन्न",
        "छप्पन्न",
        "सत्तावन्न",
        "अठ्ठावन्न",
        "एकोणसाठ",
        "साठ",
        "एकसष्ट",
        "बासष्ट",
        "त्रेसष्ट",
        "चौसष्ट",
        "पासष्ट",
        "सहासष्ट",
        "सदुसष्ट",
        "अडुसष्ट",
        "एकोणसत्तर",
        "सत्तर",
        "एक्काहत्तर",
        "बाहत्तर",
        "त्र्याहत्तर",
        "चौऱ्याहत्तर",
        "पंच्याहत्तर",
        "शहात्तर",
        "सत्त्याहत्तर",
        "अठ्ठ्याहत्तर",
        "एकोणऐंशी",
        "ऐंशी",
        "एक्क्याऐंशी",
        "ब्याऐंशी",
        "त्र्याऐंशी",
        "चौऱ्याऐंशी",
        "पंच्याऐंशी",
        "शहाऐंशी",
        "सत्त्याऐंशी",
        "अठ्ठ्याऐंशी",
        "एकोणनव्वद",
        "नव्वद",
        "एक्क्याण्णव",
        "ब्याण्णव",
        "त्र्याण्णव",
        "चौऱ्याण्णव",
        "पंच्याण्णव",
        "शहाण्णव",
        "सत्त्याण्णव",
        "अठ्ठ्याण्णव",
        "नव्व्याण्णव",
    ]
    HUNDRED = "शे"
    THOUSAND = "हजार"
    LAKH = "लाख"
    CRORE = "कोटी"

    LETTERS = {
        "A": "ए",
        "B": "बी",
        "C": "सी",
        "D": "डी",
        "E": "ई",
        "F": "एफ",
        "G": "जी",
        "H": "एच",
        "I": "आय",
        "J": "जे",
        "K": "के",
        "L": "एल",
        "M": "एम",
        "N": "एन",
        "O": "ओ",
        "P": "पी",
        "Q": "क्यू",
        "R": "आर",
        "S": "एस",
        "T": "टी",
        "U": "यू",
        "V": "व्ही",
        "W": "डब्ल्यू",
        "X": "एक्स",
        "Y": "वाय",
        "Z": "झेड",
    }

    MONTHS = [
        "जानेवारी",
        "फेब्रुवारी",
        "मार्च",
        "एप्रिल",
        "मे",
        "जून",
        "जुलै",
        "ऑगस्ट",
        "सप्टेंबर",
        "ऑक्टोबर",
        "नोव्हेंबर",
        "डिसेंबर",
    ]

    SYM = {
        "rupee": "रुपये",
        "dollar": "डॉलर",
        "euro": "युरो",
        "pound": "पाउंड",
        "cent": "सेंट",
        "percent": "टक्के",
        "point": "पूर्णांक",
        "minus": "उणे",
        "plus": "अधिक",
        "equals": "बरोबर",
        "times": "गुणिले",
        "divide": "भागिले",
        "squared": "चा वर्ग",
        "sqrt": "वर्गमूळ",
        "lt": "पेक्षा कमी",
        "gt": "पेक्षा जास्त",
        "plus_minus": "अधिक-उणे",
        "at": "ॲट द रेट",
        "underscore": "अंडरस्कोर",
        "dot": "डॉट",
        "slash": "स्लॅश",
        "hash": "हॅश",
        "amp": "आणि",
        "half": "अर्धा",
        "quarter": "पाव",
        "three_quarters": "पाऊण",
        "degree": "अंश",
        "celsius": "अंश सेल्सिअस",
        "fahrenheit": "अंश फॅरनहाइट",
        "minute": "मिनिटे",
        "second": "सेकंद",
    }

    TIME_ON_HOUR = "{h} वाजता"
    TIME_HM = "{h} वाजून {m} मिनिटे"
    RANGE = "{a} ते {b}"
    RANGE_TO = "{a} ते {b} पर्यंत"
    RATIO = "{a} गुणोत्तर {b}"
    POWER = "{base} चा घात {exp}"
    FRACTION = "{num} बटा {den}"
    DECADE_SHORT = "{tens} च्या"
    DECADE_FULL = "{tens} चे दशक"

    PERIODS = {
        "morning": "सकाळी",
        "afternoon": "दुपारी",
        "evening": "संध्याकाळी",
        "night": "रात्री",
    }

    TIME_MARKERS = ["वाजता", "वाजून", "वाजले", "वाजल्या", "वाजेपर्यंत"]
    PERIOD_DETECT = {
        "सकाळी": "morning",
        "सकाळ": "morning",
        "सकाळचे": "morning",
        "पहाटे": "morning",
        "दुपारी": "afternoon",
        "दुपार": "afternoon",
        "दुपारचे": "afternoon",
        "संध्याकाळी": "evening",
        "संध्याकाळ": "evening",
        "सायंकाळी": "evening",
        "रात्री": "night",
        "रात्र": "night",
        "रात्रीचे": "night",
    }
    YEAR_WORDS = ["साल", "साली", "सालच्या", "सन", "वर्ष", "वर्षी", "इसवी"]
    ID_KEYWORDS = [
        "नंबर",
        "क्रमांक",
        "खाते",
        "खात्याचा",
        "खात्यावर",
        "कोड",
        "ऑर्डर",
        "फोन",
        "मोबाईल",
        "मोबाइल",
        "अकाउंट",
        "अकाउंटसाठी",
        "अकाउंटमधून",
        "खात्यातून",
        "खात्यात",
        "खात्याची",
        "पिन",
        "आयडी",
        "पासवर्ड",
        "ट्रेन",
        "एक्सप्रेस",
        "पीएनआर",
        "कार्ड",
        "रेफरन्स",
        "ट्रांझॅक्शन",
        "तिकीट",
        "ओटीपी",
        "गाडी",
        "युजर",
        "यूजर",
        "हप्ता",
        "हप्त्याचा",
        "अंक",
        "व्यवहार",
        "ट्रान्झॅक्शन",
        "क्र",
        "क्रं",
        "चेक",
        "धनादेश",
    ]
    ID_AFTER_WORDS = ["डायल", "कॉल", "संपर्क"]
    NUMBER_WORD = "क्रमांक"
    ABBREV_UNITS = {
        "MINS": "मिनिटे",
        "MIN": "मिनिट",
        "HRS": "तास",
        "HR": "तास",
        "SECS": "सेकंद",
        "SEC": "सेकंद",
        "KMS": "किलोमीटर",
    }
    LABEL_WORDS = [
        "कलम",
        "अनुच्छेद",
        "खोली",
        "वर्ग",
        "इयत्ता",
        "वॉर्ड",
        "अध्याय",
        "भाग",
        "प्रकरण",
        "गट",
    ]
    MINUTE_WORDS = ["मिनिटे", "मिनिटं", "मिनिट", "मिनिटांनी"]
    SCALE_WORDS = {"कोटी": "crore", "लाख": "lakh", "हजार": "thousand", "अब्ज": "arab"}
    CURRENCY_WORDS = {
        "रुपये": "rupee",
        "रुपया": "rupee",
        "रुपयांचा": "rupee",
        "रुपयांची": "rupee",
        "रु": "rupee",
    }

    ORDINAL_EXCEPTIONS = {1: "पहिला", 2: "दुसरा", 3: "तिसरा", 4: "चौथा"}
    ORDINAL_SUFFIX = "वा"
    ORDINAL_INPUT_SUFFIXES = ["वा", "वी", "वे", "व्या", "वं"]

    # ------------------------------------------------------------------
    # Hundreds fuse: एक+शे -> एकशे, दोन+शे -> दोनशे (written solid).
    # Exactly 100 with no remainder is शंभर.
    # ------------------------------------------------------------------
    def hundreds(self, h, rest):
        if h == 1 and rest == 0:
            return "शंभर"
        out = f"{self.ONES[h]}{self.HUNDRED}"
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def year(self, y):
        """1947 -> एकोणीसशे सत्तेचाळीस (fused शे); 2026 -> दोन हजार सव्वीस."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            out = f"{self.two_digit(hi)}{self.HUNDRED}"
            if lo:
                out += f" {self.two_digit(lo)}"
            return out
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # 12वा -> बारावा, 12वी -> बारावी : cardinal + preserved suffix
        if src_suffix:
            return f"{self.cardinal(n)}{src_suffix}"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"

    # Marathi: -ए ending noun-words take oblique plural -यां before a case
    # marker (रुपये+चा->रुपयांचा, मिनिटे+ला->मिनिटांना)
    _ATTACH_NOUNS = {
        "रुपये",
        "पैसे",
        "डॉलर",
        "टक्के",
        "मिनिटे",
        "तास",
        "किलोमीटर",
        "किलो",
        "मीटर",
        "लिटर",
        "ग्रॅम",
        "अंश",
    }
    _CASE = {
        "चा": "चा",
        "ची": "ची",
        "चे": "चे",
        "चं": "चं",
        "च्या": "च्या",
        "ला": "ना",
        "ना": "ना",
        "नी": "नी",
        "त": "त",
        "तून": "तून",
        "हून": "हून",
        "पर्यंत": "पर्यंत",
        "मध्ये": "मध्ये",
        "साठी": "साठी",
    }

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-").lstrip("ांं")
        last = rep.rsplit(" ", 1)[-1]
        if last not in self._ATTACH_NOUNS or t not in self._CASE:
            return None
        base = rep[: len(rep) - len(last)]
        obl = last[:-1] + "ां" if last.endswith("े") else last
        return base + obl + self._CASE[t]


class Odia(LangBase):
    code = "or"
    name = "Odia"
    native_digits = "୦୧୨୩୪୫୬୭୮୯"

    ONES = [
        "ଶୂନ୍ୟ",
        "ଏକ",
        "ଦୁଇ",
        "ତିନି",
        "ଚାରି",
        "ପାଞ୍ଚ",
        "ଛଅ",
        "ସାତ",
        "ଆଠ",
        "ନଅ",
        "ଦଶ",
        "ଏଗାର",
        "ବାର",
        "ତେର",
        "ଚଉଦ",
        "ପନ୍ଦର",
        "ଷୋହଳ",
        "ସତର",
        "ଅଠର",
        "ଉଣେଇଶି",
        "କୋଡ଼ିଏ",
        "ଏକୋଇଶି",
        "ବାଇଶି",
        "ତେଇଶି",
        "ଚବିଶି",
        "ପଚିଶି",
        "ଛବିଶି",
        "ସତାଇଶି",
        "ଅଠେଇଶି",
        "ଅଣତିରିଶି",
        "ତିରିଶି",
        "ଏକତିରିଶି",
        "ବତିଶି",
        "ତେତିଶି",
        "ଚଉତିରିଶି",
        "ପଞ୍ଚତିରିଶି",
        "ଛତିଶି",
        "ସଇଁତିରିଶି",
        "ଅଠତିରିଶି",
        "ଅଣଚାଳିଶି",
        "ଚାଳିଶି",
        "ଏକଚାଳିଶି",
        "ବୟାଳିଶି",
        "ତେୟାଳିଶି",
        "ଚଉରାଳିଶି",
        "ପଞ୍ଚଚାଳିଶି",
        "ଛୟାଳିଶି",
        "ସତଚାଳିଶି",
        "ଅଠଚାଳିଶି",
        "ଅଣଚାଶ",
        "ପଚାଶ",
        "ଏକାବନ",
        "ବାଉନ",
        "ତେପନ",
        "ଚଉବନ",
        "ପଞ୍ଚାବନ",
        "ଛପନ",
        "ସତାବନ",
        "ଅଠାବନ",
        "ଅଣଷଠି",
        "ଷାଠିଏ",
        "ଏକଷଠି",
        "ବାଷଠି",
        "ତେଷଠି",
        "ଚଉଷଠି",
        "ପଞ୍ଚଷଠି",
        "ଛଅଷଠି",
        "ସତଷଠି",
        "ଅଠଷଠି",
        "ଅଣସ୍ତରି",
        "ସତୁରି",
        "ଏକସ୍ତରି",
        "ବାସ୍ତରି",
        "ତେସ୍ତରି",
        "ଚଉସ୍ତରି",
        "ପଞ୍ଚସ୍ତରି",
        "ଛଅସ୍ତରି",
        "ସତସ୍ତରି",
        "ଅଠସ୍ତରି",
        "ଅଣାଅଶୀ",
        "ଅଶୀ",
        "ଏକାଅଶୀ",
        "ବୟାଅଶୀ",
        "ତେୟାଅଶୀ",
        "ଚଉରାଅଶୀ",
        "ପଞ୍ଚାଅଶୀ",
        "ଛୟାଅଶୀ",
        "ସତାଅଶୀ",
        "ଅଠାଅଶୀ",
        "ଅଣାନବେ",
        "ନବେ",
        "ଏକାନବେ",
        "ବୟାନବେ",
        "ତେୟାନବେ",
        "ଚଉରାନବେ",
        "ପଞ୍ଚାନବେ",
        "ଛୟାନବେ",
        "ସତାନବେ",
        "ଅଠାନବେ",
        "ଅନେଶତ",
    ]
    HUNDRED = "ଶହ"
    THOUSAND = "ହଜାର"
    LAKH = "ଲକ୍ଷ"
    CRORE = "କୋଟି"

    LETTERS = {
        "A": "ଏ",
        "B": "ବି",
        "C": "ସି",
        "D": "ଡି",
        "E": "ଇ",
        "F": "ଏଫ୍",
        "G": "ଜି",
        "H": "ଏଚ୍",
        "I": "ଆଇ",
        "J": "ଜେ",
        "K": "କେ",
        "L": "ଏଲ୍",
        "M": "ଏମ୍",
        "N": "ଏନ୍",
        "O": "ଓ",
        "P": "ପି",
        "Q": "କ୍ୟୁ",
        "R": "ଆର୍",
        "S": "ଏସ୍",
        "T": "ଟି",
        "U": "ୟୁ",
        "V": "ଭି",
        "W": "ଡବ୍ଲ୍ୟୁ",
        "X": "ଏକ୍ସ",
        "Y": "ୱାଇ",
        "Z": "ଜେଡ୍",
    }

    MONTHS = [
        "ଜାନୁଆରୀ",
        "ଫେବୃଆରୀ",
        "ମାର୍ଚ୍ଚ",
        "ଅପ୍ରେଲ",
        "ମଇ",
        "ଜୁନ",
        "ଜୁଲାଇ",
        "ଅଗଷ୍ଟ",
        "ସେପ୍ଟେମ୍ବର",
        "ଅକ୍ଟୋବର",
        "ନଭେମ୍ବର",
        "ଡିସେମ୍ବର",
    ]

    SYM = {
        "rupee": "ଟଙ୍କା",
        "dollar": "ଡଲାର",
        "euro": "ୟୁରୋ",
        "pound": "ପାଉଣ୍ଡ",
        "cent": "ସେଣ୍ଟ",
        "percent": "ପ୍ରତିଶତ",
        "point": "ଦଶମିକ",
        "minus": "ମାଇନସ୍",
        "plus": "ପ୍ଲସ୍",
        "equals": "ସମାନ",
        "times": "ଗୁଣ",
        "divide": "ଭାଗ",
        "squared": "ର ବର୍ଗ",
        "sqrt": "ବର୍ଗମୂଳ",
        "lt": "ଠାରୁ କମ୍",
        "gt": "ଠାରୁ ଅଧିକ",
        "plus_minus": "ପ୍ଲସ୍-ମାଇନସ୍",
        "at": "ଆଟ୍ ଦି ରେଟ୍",
        "underscore": "ଅଣ୍ଡରସ୍କୋର",
        "dot": "ଡଟ୍",
        "slash": "ସ୍ଲାସ୍",
        "hash": "ହ୍ୟାସ୍",
        "amp": "ଆଣ୍ଡ",
        "half": "ଅଧା",
        "quarter": "ଏକ ଚତୁର୍ଥାଂଶ",
        "three_quarters": "ତିନି ଚତୁର୍ଥାଂଶ",
        "degree": "ଡିଗ୍ରୀ",
        "celsius": "ଡିଗ୍ରୀ ସେଲସିୟସ୍",
        "fahrenheit": "ଡିଗ୍ରୀ ଫାରେନହାଇଟ୍",
        "minute": "ମିନିଟ୍",
        "paisa": "ପଇସା",
        "per": "ପ୍ରତି",
        "second": "ସେକେଣ୍ଡ",
    }

    TIME_ON_HOUR = "{h}ଟା"
    TIME_HM = "{h}ଟା {m} ମିନିଟ୍"
    RANGE = "{a} ରୁ {b}"
    RANGE_TO = "{a} ରୁ {b} ପର୍ଯ୍ୟନ୍ତ"
    RATIO = "{a} ଅନୁପାତ {b}"
    POWER = "{base} ର ଘାତ {exp}"
    FRACTION = "{num} ବଟା {den}"
    DECADE_SHORT = "{tens} ଦଶକ"
    DECADE_FULL = "{tens} ଦଶକ"

    PERIODS = {
        "morning": "ସକାଳ",
        "afternoon": "ଦିନ",
        "evening": "ସନ୍ଧ୍ୟା",
        "night": "ରାତି",
    }

    TIME_MARKERS = ["ଟା", "ଟାରେ", "ଟାରୁ", "ବେଳେ", "ବଜେ"]
    PERIOD_DETECT = {
        "ସକାଳ": "morning",
        "ସକାଳେ": "morning",
        "ସକାଳୁ": "morning",
        "ଭୋର": "morning",
        "ଭୋରରୁ": "morning",
        "ଦିନ": "afternoon",
        "ଅପରାହ୍ନ": "afternoon",
        "ମଧ୍ୟାହ୍ନ": "afternoon",
        "ଦ୍ୱିପହର": "afternoon",
        "ସନ୍ଧ୍ୟା": "evening",
        "ସନ୍ଧ୍ୟାରେ": "evening",
        "ରାତି": "night",
        "ରାତିରେ": "night",
        "ରାତ୍ରି": "night",
    }
    YEAR_WORDS = ["ମସିହା", "ମସିହାରେ", "ସାଲ", "ସନ", "ବର୍ଷ", "ବର୍ଷରେ"]
    ID_KEYWORDS = [
        "ନମ୍ବର",
        "ନମ୍ବର୍",
        "ଖାତା",
        "ଆକାଉଣ୍ଟ",
        "କୋଡ୍",
        "ଅର୍ଡର",
        "ଫୋନ୍",
        "ଫୋନ",
        "ମୋବାଇଲ୍",
        "ମୋବାଇଲ",
        "ଓଟିପି",
        "ପିନ୍",
        "ଆଇଡି",
        "ପାସୱାର୍ଡ",
        "ଏକ୍ସପ୍ରେସ",
        "ପିଏନ୍‌ଆର୍",
        "କାର୍ଡ",
        "ରେଫରେନ୍ସ",
        "ଟିକେଟ",
        "ଆକାଉଣ୍ଟରୁ",
        "ଟ୍ରେନ୍",
        "ୟୁଜର",
        "କିସ୍ତି",
        "ଅଙ୍କ",
        "ଲାଣଦେଣ",
    ]
    ID_AFTER_WORDS = ["ଡାଏଲ", "କଲ", "ଯୋଗାଯୋଗ"]
    NUMBER_WORD = "ନମ୍ବର"
    ABBREV_UNITS = {
        "MINS": "ମିନିଟ୍",
        "MIN": "ମିନିଟ୍",
        "HRS": "ଘଣ୍ଟା",
        "HR": "ଘଣ୍ଟା",
        "SECS": "ସେକେଣ୍ଡ",
        "SEC": "ସେକେଣ୍ଡ",
        "KMS": "କିଲୋମିଟର",
    }
    LABEL_WORDS = [
        "ଅନୁଚ୍ଛେଦ",
        "ଧାରା",
        "ରୁମ୍",
        "କୋଠରି",
        "ୱାର୍ଡ",
        "ଅଧ୍ୟାୟ",
        "ଭାଗ",
        "ଗେଟ୍",
        "କାଉଣ୍ଟର",
        "ଶ୍ରେଣୀ",
    ]
    MINUTE_WORDS = ["ମିନିଟ୍", "ମିନିଟ"]
    SCALE_WORDS = {"କୋଟି": "crore", "ଲକ୍ଷ": "lakh", "ଲାଖ": "lakh", "ହଜାର": "thousand"}
    CURRENCY_WORDS = {
        "ଟଙ୍କା": "rupee",
        "ଟଙ୍କାର": "rupee",
        "ଟଙ୍କାରେ": "rupee",
        "ଟଂକା": "rupee",
        "ରୁପି": "rupee",
    }

    ORDINAL_EXCEPTIONS = {
        1: "ପ୍ରଥମ",
        2: "ଦ୍ୱିତୀୟ",
        3: "ତୃତୀୟ",
        4: "ଚତୁର୍ଥ",
        5: "ପଞ୍ଚମ",
        6: "ଷଷ୍ଠ",
        7: "ସପ୍ତମ",
        8: "ଅଷ୍ଟମ",
        9: "ନବମ",
        10: "ଦଶମ",
    }
    ORDINAL_SUFFIX = "ତମ"
    ORDINAL_INPUT_SUFFIXES = ["ତମ"]

    # Odia postpositions agglutinate onto noun-words (ଟଙ୍କା + ରୁ -> ଟଙ୍କାରୁ)
    _ATTACH_NOUNS = {
        "ଟଙ୍କା",
        "ପଇସା",
        "ଡଲାର",
        "ପ୍ରତିଶତ",
        "କିଲୋମିଟର",
        "କିଲୋଗ୍ରାମ",
        "ମିଟର",
        "ଲିଟର",
        "ଗ୍ରାମ",
        "ଡିଗ୍ରୀ",
        "ମିନିଟ୍",
        "ଘଣ୍ଟା",
        "ସେକେଣ୍ଡ",
    }
    _PP = {
        "ରେ",
        "ରୁ",
        "କୁ",
        "ର",
        "ଠାରୁ",
        "ପର୍ଯ୍ୟନ୍ତ",
        "ମଧ୍ୟରେ",
        "ପାଇଁ",
        "ଅପେକ୍ଷା",
        "ଭିତରେ",
    }

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if t == last and last in self._ATTACH_NOUNS:
            return rep
        if t in self._PP and last in self._ATTACH_NOUNS:
            return rep + t
        return None


class Punjabi(LangBase):
    code = "pa"
    name = "Punjabi"
    native_digits = "੦੧੨੩੪੫੬੭੮੯"
    SINGULAR_NOUNS = {"ਘੰਟੇ": "ਘੰਟਾ"}

    ONES = [
        "ਸਿਫ਼ਰ",
        "ਇੱਕ",
        "ਦੋ",
        "ਤਿੰਨ",
        "ਚਾਰ",
        "ਪੰਜ",
        "ਛੇ",
        "ਸੱਤ",
        "ਅੱਠ",
        "ਨੌਂ",
        "ਦਸ",
        "ਗਿਆਰਾਂ",
        "ਬਾਰਾਂ",
        "ਤੇਰਾਂ",
        "ਚੌਦਾਂ",
        "ਪੰਦਰਾਂ",
        "ਸੋਲਾਂ",
        "ਸਤਾਰਾਂ",
        "ਅਠਾਰਾਂ",
        "ਉੱਨੀ",
        "ਵੀਹ",
        "ਇੱਕੀ",
        "ਬਾਈ",
        "ਤੇਈ",
        "ਚੌਵੀ",
        "ਪੱਚੀ",
        "ਛੱਬੀ",
        "ਸਤਾਈ",
        "ਅਠਾਈ",
        "ਉਣੱਤੀ",
        "ਤੀਹ",
        "ਇਕੱਤੀ",
        "ਬੱਤੀ",
        "ਤੇਤੀ",
        "ਚੌਂਤੀ",
        "ਪੈਂਤੀ",
        "ਛੱਤੀ",
        "ਸੈਂਤੀ",
        "ਅਠੱਤੀ",
        "ਉਣਤਾਲੀ",
        "ਚਾਲੀ",
        "ਇਕਤਾਲੀ",
        "ਬਿਆਲੀ",
        "ਤਰਤਾਲੀ",
        "ਚੁਤਾਲੀ",
        "ਪੰਤਾਲੀ",
        "ਛਿਆਲੀ",
        "ਸੰਤਾਲੀ",
        "ਅਠਤਾਲੀ",
        "ਉਣੰਜਾ",
        "ਪੰਜਾਹ",
        "ਇਕਵੰਜਾ",
        "ਬਵੰਜਾ",
        "ਤਰਵੰਜਾ",
        "ਚੁਰੰਜਾ",
        "ਪਚਵੰਜਾ",
        "ਛਪੰਜਾ",
        "ਸਤਵੰਜਾ",
        "ਅਠਵੰਜਾ",
        "ਉਣਾਹਠ",
        "ਸੱਠ",
        "ਇਕਾਹਠ",
        "ਬਾਹਠ",
        "ਤ੍ਰੇਹਠ",
        "ਚੌਂਹਠ",
        "ਪੈਂਹਠ",
        "ਛਿਆਹਠ",
        "ਸਤਾਹਠ",
        "ਅਠਾਹਠ",
        "ਉਣੱਤਰ",
        "ਸੱਤਰ",
        "ਇਕਹੱਤਰ",
        "ਬਹੱਤਰ",
        "ਤਿਹੱਤਰ",
        "ਚੁਹੱਤਰ",
        "ਪੰਝੱਤਰ",
        "ਛਿਹੱਤਰ",
        "ਸਤੱਤਰ",
        "ਅਠੱਤਰ",
        "ਉਣਾਸੀ",
        "ਅੱਸੀ",
        "ਇਕਾਸੀ",
        "ਬਿਆਸੀ",
        "ਤਰਾਸੀ",
        "ਚੁਰਾਸੀ",
        "ਪਚਾਸੀ",
        "ਛਿਆਸੀ",
        "ਸਤਾਸੀ",
        "ਅਠਾਸੀ",
        "ਉਣਾਨਵੇਂ",
        "ਨੱਬੇ",
        "ਇਕਾਨਵੇਂ",
        "ਬਾਨਵੇਂ",
        "ਤਰਾਨਵੇਂ",
        "ਚੁਰਾਨਵੇਂ",
        "ਪਚਾਨਵੇਂ",
        "ਛਿਆਨਵੇਂ",
        "ਸਤਾਨਵੇਂ",
        "ਅਠਾਨਵੇਂ",
        "ਨੜਿੰਨਵੇਂ",
    ]
    HUNDRED = "ਸੌ"
    THOUSAND = "ਹਜ਼ਾਰ"
    LAKH = "ਲੱਖ"
    CRORE = "ਕਰੋੜ"

    LETTERS = {
        "A": "ਏ",
        "B": "ਬੀ",
        "C": "ਸੀ",
        "D": "ਡੀ",
        "E": "ਈ",
        "F": "ਐੱਫ",
        "G": "ਜੀ",
        "H": "ਐੱਚ",
        "I": "ਆਈ",
        "J": "ਜੇ",
        "K": "ਕੇ",
        "L": "ਐੱਲ",
        "M": "ਐੱਮ",
        "N": "ਐੱਨ",
        "O": "ਓ",
        "P": "ਪੀ",
        "Q": "ਕਿਊ",
        "R": "ਆਰ",
        "S": "ਐੱਸ",
        "T": "ਟੀ",
        "U": "ਯੂ",
        "V": "ਵੀ",
        "W": "ਡਬਲਿਊ",
        "X": "ਐਕਸ",
        "Y": "ਵਾਈ",
        "Z": "ਜ਼ੈੱਡ",
    }

    MONTHS = [
        "ਜਨਵਰੀ",
        "ਫ਼ਰਵਰੀ",
        "ਮਾਰਚ",
        "ਅਪ੍ਰੈਲ",
        "ਮਈ",
        "ਜੂਨ",
        "ਜੁਲਾਈ",
        "ਅਗਸਤ",
        "ਸਤੰਬਰ",
        "ਅਕਤੂਬਰ",
        "ਨਵੰਬਰ",
        "ਦਸੰਬਰ",
    ]

    SYM = {
        "rupee": "ਰੁਪਏ",
        "dollar": "ਡਾਲਰ",
        "euro": "ਯੂਰੋ",
        "pound": "ਪੌਂਡ",
        "cent": "ਸੈਂਟ",
        "percent": "ਫ਼ੀਸਦੀ",
        "point": "ਦਸ਼ਮਲਵ",
        "minus": "ਮਾਈਨਸ",
        "plus": "ਪਲੱਸ",
        "equals": "ਬਰਾਬਰ",
        "times": "ਗੁਣਾ",
        "divide": "ਭਾਗ",
        "squared": "ਦਾ ਵਰਗ",
        "sqrt": "ਵਰਗਮੂਲ",
        "lt": "ਤੋਂ ਘੱਟ",
        "gt": "ਤੋਂ ਵੱਧ",
        "plus_minus": "ਧਨ-ਰਿਣ",
        "at": "ਐਟ ਦ ਰੇਟ",
        "underscore": "ਅੰਡਰਸਕੋਰ",
        "dot": "ਡਾਟ",
        "slash": "ਸਲੈਸ਼",
        "hash": "ਹੈਸ਼",
        "amp": "ਅਤੇ",
        "half": "ਅੱਧਾ",
        "quarter": "ਚੌਥਾਈ",
        "three_quarters": "ਤਿੰਨ ਚੌਥਾਈ",
        "degree": "ਡਿਗਰੀ",
        "celsius": "ਡਿਗਰੀ ਸੈਲਸੀਅਸ",
        "fahrenheit": "ਡਿਗਰੀ ਫਾਰਨਹਾਈਟ",
        "minute": "ਮਿੰਟ",
        "second": "ਸਕਿੰਟ",
    }

    TIME_ON_HOUR = "{h} ਵਜੇ"
    TIME_HM = "{h} ਵੱਜ ਕੇ {m} ਮਿੰਟ"
    RANGE = "{a} ਤੋਂ {b}"
    RANGE_TO = "{a} ਤੋਂ {b} ਤੱਕ"
    RATIO = "{a} ਅਨੁਪਾਤ {b}"
    POWER = "{base} ਦੀ ਘਾਤ {exp}"
    FRACTION = "{num} ਬਟਾ {den}"
    DECADE_SHORT = "{tens} ਦੇ"
    DECADE_FULL = "{tens} ਦਾ ਦਹਾਕਾ"

    PERIODS = {
        "morning": "ਸਵੇਰੇ",
        "afternoon": "ਦੁਪਹਿਰ",
        "evening": "ਸ਼ਾਮ",
        "night": "ਰਾਤ",
    }

    TIME_MARKERS = ["ਵਜੇ", "ਵੱਜ ਕੇ", "ਵਜਕੇ", "ਵੱਜੇ"]
    PERIOD_DETECT = {
        "ਸਵੇਰੇ": "morning",
        "ਸਵੇਰ": "morning",
        "ਤੜਕੇ": "morning",
        "ਦੁਪਹਿਰ": "afternoon",
        "ਸ਼ਾਮ": "evening",
        "ਸ਼ਾਮੀਂ": "evening",
        "ਰਾਤ": "night",
        "ਰਾਤੀਂ": "night",
    }
    YEAR_WORDS = ["ਸਾਲ", "ਵਰ੍ਹਾ", "ਵਰ੍ਹੇ", "ਸੰਨ"]
    ID_KEYWORDS = [
        "ਨੰਬਰ",
        "ਖਾਤਾ",
        "ਖਾਤੇ",
        "ਕੋਡ",
        "ਆਰਡਰ",
        "ਫ਼ੋਨ",
        "ਫੋਨ",
        "ਮੋਬਾਈਲ",
        "ਮੋਬਾਇਲ",
        "ਅਕਾਊਂਟ",
        "ਅਕਾਊਂਟਵਿੱਚ",
        "ਪਿੰਨ",
        "ਆਈਡੀ",
        "ਪਾਸਵਰਡ",
        "ਟ੍ਰੇਨ",
        "ਐਕਸਪ੍ਰੈਸ",
        "ਪੀਐਨਆਰ",
        "ਕਾਰਡ",
        "ਰੈਫਰੈਂਸ",
        "ਟਿਕਟ",
        "ਓਟੀਪੀ",
        "ਯੂਜ਼ਰ",
        "ਕਿਸ਼ਤ",
        "ਅੰਕ",
        "ਲੈਣ-ਦੇਣ",
    ]
    ID_AFTER_WORDS = ["ਡਾਇਲ", "ਕਾਲ", "ਸੰਪਰਕ"]
    NUMBER_WORD = "ਨੰਬਰ"
    ABBREV_UNITS = {
        "MINS": "ਮਿੰਟ",
        "MIN": "ਮਿੰਟ",
        "HRS": "ਘੰਟੇ",
        "HR": "ਘੰਟਾ",
        "SECS": "ਸਕਿੰਟ",
        "SEC": "ਸਕਿੰਟ",
        "KMS": "ਕਿਲੋਮੀਟਰ",
    }
    LABEL_WORDS = ["ਧਾਰਾ", "ਅਨੁਛੇਦ", "ਕਮਰਾ", "ਕਲਾਸ", "ਵਾਰਡ", "ਅਧਿਆਇ", "ਭਾਗ", "ਜਮਾਤ"]
    MINUTE_WORDS = ["ਮਿੰਟ"]
    SCALE_WORDS = {
        "ਕਰੋੜ": "crore",
        "ਲੱਖ": "lakh",
        "ਹਜ਼ਾਰ": "thousand",
        "ਹਜਾਰ": "thousand",
        "ਅਰਬ": "arab",
    }
    CURRENCY_WORDS = {
        "ਰੁਪਏ": "rupee",
        "ਰੁਪਿਆ": "rupee",
        "ਰੁਪਈਏ": "rupee",
        "ਰੁਪਿਏ": "rupee",
    }

    ORDINAL_EXCEPTIONS = {1: "ਪਹਿਲਾ", 2: "ਦੂਜਾ", 3: "ਤੀਜਾ", 4: "ਚੌਥਾ", 6: "ਛੇਵਾਂ"}
    ORDINAL_SUFFIX = "ਵਾਂ"
    ORDINAL_INPUT_SUFFIXES = ["ਵੀਂ", "ਵਾਂ", "ਵੇਂ", "ਵਾ"]

    def _ordinal_compose(self, n, src_suffix):
        # 12ਵੀਂ -> ਬਾਰਾਂਵੀਂ : cardinal + preserved suffix
        if src_suffix:
            return f"{self.cardinal(n)}{src_suffix}"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"


class Tamil(LangBase):
    code = "ta"
    name = "Tamil"
    native_digits = "௦௧௨௩௪௫௬௭௮௯"
    COUNT_ONE = "ஒரு"

    ONES = [
        "பூஜ்ஜியம்",
        "ஒன்று",
        "இரண்டு",
        "மூன்று",
        "நான்கு",
        "ஐந்து",
        "ஆறு",
        "ஏழு",
        "எட்டு",
        "ஒன்பது",
        "பத்து",
        "பதினொன்று",
        "பன்னிரண்டு",
        "பதின்மூன்று",
        "பதினான்கு",
        "பதினைந்து",
        "பதினாறு",
        "பதினேழு",
        "பதினெட்டு",
        "பத்தொன்பது",
        "இருபது",
        "இருபத்தொன்று",
        "இருபத்திரண்டு",
        "இருபத்துமூன்று",
        "இருபத்துநான்கு",
        "இருபத்தைந்து",
        "இருபத்தாறு",
        "இருபத்தேழு",
        "இருபத்தெட்டு",
        "இருபத்தொன்பது",
        "முப்பது",
        "முப்பத்தொன்று",
        "முப்பத்திரண்டு",
        "முப்பத்துமூன்று",
        "முப்பத்துநான்கு",
        "முப்பத்தைந்து",
        "முப்பத்தாறு",
        "முப்பத்தேழு",
        "முப்பத்தெட்டு",
        "முப்பத்தொன்பது",
        "நாற்பது",
        "நாற்பத்தொன்று",
        "நாற்பத்திரண்டு",
        "நாற்பத்துமூன்று",
        "நாற்பத்துநான்கு",
        "நாற்பத்தைந்து",
        "நாற்பத்தாறு",
        "நாற்பத்தேழு",
        "நாற்பத்தெட்டு",
        "நாற்பத்தொன்பது",
        "ஐம்பது",
        "ஐம்பத்தொன்று",
        "ஐம்பத்திரண்டு",
        "ஐம்பத்துமூன்று",
        "ஐம்பத்துநான்கு",
        "ஐம்பத்தைந்து",
        "ஐம்பத்தாறு",
        "ஐம்பத்தேழு",
        "ஐம்பத்தெட்டு",
        "ஐம்பத்தொன்பது",
        "அறுபது",
        "அறுபத்தொன்று",
        "அறுபத்திரண்டு",
        "அறுபத்துமூன்று",
        "அறுபத்துநான்கு",
        "அறுபத்தைந்து",
        "அறுபத்தாறு",
        "அறுபத்தேழு",
        "அறுபத்தெட்டு",
        "அறுபத்தொன்பது",
        "எழுபது",
        "எழுபத்தொன்று",
        "எழுபத்திரண்டு",
        "எழுபத்துமூன்று",
        "எழுபத்துநான்கு",
        "எழுபத்தைந்து",
        "எழுபத்தாறு",
        "எழுபத்தேழு",
        "எழுபத்தெட்டு",
        "எழுபத்தொன்பது",
        "எண்பது",
        "எண்பத்தொன்று",
        "எண்பத்திரண்டு",
        "எண்பத்துமூன்று",
        "எண்பத்துநான்கு",
        "எண்பத்தைந்து",
        "எண்பத்தாறு",
        "எண்பத்தேழு",
        "எண்பத்தெட்டு",
        "எண்பத்தொன்பது",
        "தொண்ணூறு",
        "தொண்ணூற்றொன்று",
        "தொண்ணூற்றிரண்டு",
        "தொண்ணூற்றுமூன்று",
        "தொண்ணூற்றுநான்கு",
        "தொண்ணூற்றைந்து",
        "தொண்ணூற்றாறு",
        "தொண்ணூற்றேழு",
        "தொண்ணூற்றெட்டு",
        "தொண்ணூற்றொன்பது",
    ]
    HUNDRED = "நூறு"
    THOUSAND = "ஆயிரம்"
    LAKH = "லட்சம்"
    CRORE = "கோடி"

    # Fused hundreds (index 1..9) and their combining forms before a remainder.
    HUNDREDS = [
        "",
        "நூறு",
        "இருநூறு",
        "முந்நூறு",
        "நானூறு",
        "ஐந்நூறு",
        "அறுநூறு",
        "எழுநூறு",
        "எண்ணூறு",
        "தொள்ளாயிரம்",
    ]
    HUNDREDS_COMB = [
        "",
        "நூற்று",
        "இருநூற்று",
        "முந்நூற்று",
        "நானூற்று",
        "ஐந்நூற்று",
        "அறுநூற்று",
        "எழுநூற்று",
        "எண்ணூற்று",
        "தொள்ளாயிரத்து",
    ]

    LETTERS = {
        "A": "ஏ",
        "B": "பி",
        "C": "சி",
        "D": "டி",
        "E": "இ",
        "F": "எஃப்",
        "G": "ஜி",
        "H": "எச்",
        "I": "ஐ",
        "J": "ஜே",
        "K": "கே",
        "L": "எல்",
        "M": "எம்",
        "N": "என்",
        "O": "ஓ",
        "P": "பி",
        "Q": "கியூ",
        "R": "ஆர்",
        "S": "எஸ்",
        "T": "டி",
        "U": "யூ",
        "V": "வி",
        "W": "டபிள்யூ",
        "X": "எக்ஸ்",
        "Y": "ஒய்",
        "Z": "இசட்",
    }

    MONTHS = [
        "ஜனவரி",
        "பிப்ரவரி",
        "மார்ச்",
        "ஏப்ரல்",
        "மே",
        "ஜூன்",
        "ஜூலை",
        "ஆகஸ்ட்",
        "செப்டம்பர்",
        "அக்டோபர்",
        "நவம்பர்",
        "டிசம்பர்",
    ]

    SYM = {
        "rupee": "ரூபாய்",
        "dollar": "டாலர்",
        "euro": "யூரோ",
        "pound": "பவுண்ட்",
        "cent": "சென்ட்",
        "percent": "சதவீதம்",
        "point": "புள்ளி",
        "minus": "மைனஸ்",
        "plus": "பிளஸ்",
        "equals": "சமம்",
        "times": "பெருக்கல்",
        "divide": "வகுத்தல்",
        "squared": "இன் வர்க்கம்",
        "sqrt": "வர்க்கமூலம்",
        "lt": "விடக் குறைவு",
        "gt": "விட அதிகம்",
        "plus_minus": "பிளஸ் மைனஸ்",
        "at": "அட் தி ரேட்",
        "underscore": "அண்டர்ஸ்கோர்",
        "dot": "டாட்",
        "slash": "ஸ்லாஷ்",
        "hash": "ஹேஷ்",
        "amp": "மற்றும்",
        "half": "அரை",
        "quarter": "கால்",
        "three_quarters": "முக்கால்",
        "degree": "டிகிரி",
        "celsius": "டிகிரி செல்சியஸ்",
        "fahrenheit": "டிகிரி ஃபாரன்ஹீட்",
        "minute": "நிமிடம்",
        "second": "வினாடி",
    }

    TIME_ON_HOUR = "{h} மணி"
    TIME_HM = "{h} மணி {m} நிமிடம்"
    RANGE = "{a} முதல் {b}"
    RANGE_TO = "{a} முதல் {b} வரை"
    RATIO = "{a} விகிதம் {b}"
    POWER = "{base} இன் அடுக்கு {exp}"
    FRACTION = "{num} வகுத்தல் {den}"
    DECADE_SHORT = "{tens}கள்"
    DECADE_FULL = "{tens}களின் தசாப்தம்"

    PERIODS = {
        "morning": "காலை",
        "afternoon": "மதியம்",
        "evening": "மாலை",
        "night": "இரவு",
    }

    TIME_MARKERS = ["மணிக்கு", "மணி"]
    MARKER_SUFFIXES = {"மணிக்கு": "க்கு"}
    PERIOD_DETECT = {
        "காலை": "morning",
        "அதிகாலை": "morning",
        "மதியம்": "afternoon",
        "நண்பகல்": "afternoon",
        "பிற்பகல்": "afternoon",
        "மாலை": "evening",
        "இரவு": "night",
    }
    YEAR_WORDS = ["ஆண்டு", "ஆண்டில்", "வருடம்", "வருடத்தில்", "வருஷம்"]
    ID_KEYWORDS = [
        "எண்",
        "நம்பர்",
        "கணக்கு",
        "தொலைபேசி",
        "போன்",
        "மொபைல்",
        "ஆர்டர்",
        "ஓடிபி",
        "பின்",
        "ஐடி",
        "கடவுச்சொல்",
        "ரயில்",
        "எக்ஸ்பிரஸ்",
        "பிஎன்ஆர்",
        "கார்டு",
        "குறியீடு",
        "டிக்கெட்",
        "கணக்கில்",
        "யூசர்",
        "தவணை",
        "இலக்கம்",
        "பரிவர்த்தனை",
    ]
    ID_AFTER_WORDS = ["டயல்", "அழை", "தொடர்பு"]
    NUMBER_WORD = "எண்"
    ABBREV_UNITS = {
        "MINS": "நிமிடம்",
        "MIN": "நிமிடம்",
        "HRS": "மணி",
        "HR": "மணி",
        "SECS": "விநாடி",
        "SEC": "விநாடி",
        "KMS": "கிலோமீட்டர்",
    }
    LABEL_WORDS = [
        "பிரிவு",
        "சரத்து",
        "அறை",
        "வார்டு",
        "அத்தியாயம்",
        "வகுப்பு",
        "பாகம்",
        "கவுண்டர்",
        "கேட்",
    ]
    MINUTE_WORDS = ["நிமிடம்", "நிமிடங்கள்", "நிமிஷம்"]
    SCALE_WORDS = {
        "கோடி": "crore",
        "கோடியே": "crore",
        "லட்சம்": "lakh",
        "லட்சத்து": "lakh",
        "இலட்சம்": "lakh",
        "ஆயிரம்": "thousand",
        "ஆயிரத்து": "thousand",
    }
    CURRENCY_WORDS = {"ரூபாய்": "rupee", "ரூ": "rupee", "ருபாய்": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "முதலாவது"}
    ORDINAL_SUFFIX = "ஆவது"
    ORDINAL_INPUT_SUFFIXES = ["ஆவது", "வது", "ஆம்", "ம்"]

    # Consonants that double onto a preceding combining form (வல்லினம்).
    _DOUBLE = {"க": "க்", "ச": "ச்", "த": "த்", "ப": "ப்"}

    # ------------------------------------------------------------------
    # Dravidian composition: fused hundreds/thousands + sandhi joins.
    # ------------------------------------------------------------------
    def _sandhi_join(self, parts):
        """Join number parts; after a combining form ending -த்து/-ற்று a
        following க/ச/த/ப doubles onto it (ஆயிரத்துத் தொள்ளாயிரம்)."""
        out = [p for p in parts if p]
        for i in range(len(out) - 1):
            if out[i].endswith(("த்து", "ற்று")) and out[i + 1][0] in self._DOUBLE:
                out[i] += self._DOUBLE[out[i + 1][0]]
        return " ".join(out)

    def hundreds(self, h, rest):
        if rest:
            return self._sandhi_join([self.HUNDREDS_COMB[h], self.two_digit(rest)])
        return self.HUNDREDS[h]

    def under_thousand(self, n):
        if n < 100:
            return self.two_digit(n)
        return self.hundreds(n // 100, n % 100)

    def _thousands_word(self, t, combining):
        """1 -> ஆயிரம்/ஆயிரத்து; else fuse multiplier: final -ு drops before
        ஆ- (நான்கு+ஆயிரம் -> நான்காயிரம், ஐம்பத்துநான்கு -> ஐம்பத்துநான்காயிரம்)."""
        base = "ஆயிரத்து" if combining else "ஆயிரம்"
        if t == 1:
            return base
        return self.ONES[t][:-1] + "ா" + base[1:]

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n == 0:
            return self.ONES[0]
        crore, rest = divmod(n, 10**7)
        lakh, rest = divmod(rest, 10**5)
        thousand, low = divmod(rest, 1000)
        parts = []
        if crore:
            mult = "ஒரு" if crore == 1 else self.cardinal(crore)
            parts.append(
                f"{mult} " + ("கோடியே" if (lakh or thousand or low) else "கோடி")
            )
        if lakh:
            mult = "ஒரு" if lakh == 1 else self.two_digit(lakh)
            parts.append(f"{mult} " + ("லட்சத்து" if (thousand or low) else "லட்சம்"))
        if thousand:
            parts.append(self._thousands_word(thousand, bool(low)))
        if low:
            parts.append(self.under_thousand(low))
        return self._sandhi_join(parts)

    def year(self, y):
        """Tamil reads years as full cardinals:
        1947 -> ஆயிரத்துத் தொள்ளாயிரத்து நாற்பத்தேழு,
        2026 -> இரண்டாயிரத்து இருபத்தாறு."""
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # 12ஆவது/12வது -> பன்னிரண்டாவது; 12ஆம்/12ம் -> பன்னிரண்டாம்.
        suffix = "ஆம்" if src_suffix in ("ஆம்", "ம்") else "ஆவது"
        if n == 1:
            return "முதலாம்" if suffix == "ஆம்" else "முதலாவது"
        base = self.cardinal(n)
        if base[-1] in ("ு", "்"):
            # drop final -u / virama and merge the suffix vowel: -ாவது / -ாம்
            return base[:-1] + "ா" + suffix[1:]
        if base[-1] in ("ி", "ீ", "ை"):
            # glide insertion: கோடி -> கோடியாவது
            return base + "யா" + suffix[1:]
        return base + suffix

    # Tamil case markers attach with sandhi to noun-words
    _ATTACH_NOUNS = {
        "ரூபாய்",
        "டாலர்",
        "சதவீதம்",
        "மணி",
        "நிமிடம்",
        "விநாடி",
        "கிலோமீட்டர்",
        "கிலோ",
        "மீட்டர்",
        "லிட்டர்",
        "கிராம்",
        "டிகிரி",
        "மைல்",
    }

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        base = rep[: len(rep) - len(last)]
        num_obl = {
            "ஒன்று": "ஒன்ற",
            "இரண்டு": "இரண்ட",
            "மூன்று": "மூன்ற",
            "நான்கு": "நான்க",
            "ஐந்து": "ஐந்த",
            "ஆறு": "ஆற",
            "ஏழு": "ஏழ",
            "எட்டு": "எட்ட",
            "ஒன்பது": "ஒன்பத",
            "பத்து": "பத்த",
            "பூஜ்ஜியம்": "பூஜ்ஜியத்த",
        }
        if last in num_obl:  # number + case (hyphen markers)
            if t in ("இல்", "ல்"):
                return base + num_obl[last] + "ில்"
            if t in ("க்கு", "கு"):
                return base + num_obl[last] + "ிற்கு"
            if t in ("இலிருந்து", "ல்இருந்து"):
                return base + num_obl[last] + "ிலிருந்து"
            return None
        if last not in self._ATTACH_NOUNS:
            return None
        if t in ("க்கு", "கு"):
            if last == "நிமிடம்":
                return base + "நிமிடத்திற்கு"
            if last == "சதவீதம்":
                return base + "சதவீதத்திற்கு"
            return base + last + "க்கு"
        if t in ("இல்", "ல்", "இலிருந்து", "ல்இருந்து"):
            if last.endswith("ம்"):
                return base + last[:-2] + "த்தில்"
            return base + last + "இல்"
        return None


class Telugu(LangBase):
    code = "te"
    name = "Telugu"
    native_digits = "౦౧౨౩౪౫౬౭౮౯"

    ONES = [
        "సున్నా",
        "ఒకటి",
        "రెండు",
        "మూడు",
        "నాలుగు",
        "ఐదు",
        "ఆరు",
        "ఏడు",
        "ఎనిమిది",
        "తొమ్మిది",
        "పది",
        "పదకొండు",
        "పన్నెండు",
        "పదమూడు",
        "పద్నాలుగు",
        "పదిహేను",
        "పదహారు",
        "పదిహేడు",
        "పద్దెనిమిది",
        "పందొమ్మిది",
        "ఇరవై",
        "ఇరవై ఒకటి",
        "ఇరవై రెండు",
        "ఇరవై మూడు",
        "ఇరవై నాలుగు",
        "ఇరవై ఐదు",
        "ఇరవై ఆరు",
        "ఇరవై ఏడు",
        "ఇరవై ఎనిమిది",
        "ఇరవై తొమ్మిది",
        "ముప్పై",
        "ముప్పై ఒకటి",
        "ముప్పై రెండు",
        "ముప్పై మూడు",
        "ముప్పై నాలుగు",
        "ముప్పై ఐదు",
        "ముప్పై ఆరు",
        "ముప్పై ఏడు",
        "ముప్పై ఎనిమిది",
        "ముప్పై తొమ్మిది",
        "నలభై",
        "నలభై ఒకటి",
        "నలభై రెండు",
        "నలభై మూడు",
        "నలభై నాలుగు",
        "నలభై ఐదు",
        "నలభై ఆరు",
        "నలభై ఏడు",
        "నలభై ఎనిమిది",
        "నలభై తొమ్మిది",
        "యాభై",
        "యాభై ఒకటి",
        "యాభై రెండు",
        "యాభై మూడు",
        "యాభై నాలుగు",
        "యాభై ఐదు",
        "యాభై ఆరు",
        "యాభై ఏడు",
        "యాభై ఎనిమిది",
        "యాభై తొమ్మిది",
        "అరవై",
        "అరవై ఒకటి",
        "అరవై రెండు",
        "అరవై మూడు",
        "అరవై నాలుగు",
        "అరవై ఐదు",
        "అరవై ఆరు",
        "అరవై ఏడు",
        "అరవై ఎనిమిది",
        "అరవై తొమ్మిది",
        "డెబ్బై",
        "డెబ్బై ఒకటి",
        "డెబ్బై రెండు",
        "డెబ్బై మూడు",
        "డెబ్బై నాలుగు",
        "డెబ్బై ఐదు",
        "డెబ్బై ఆరు",
        "డెబ్బై ఏడు",
        "డెబ్బై ఎనిమిది",
        "డెబ్బై తొమ్మిది",
        "ఎనభై",
        "ఎనభై ఒకటి",
        "ఎనభై రెండు",
        "ఎనభై మూడు",
        "ఎనభై నాలుగు",
        "ఎనభై ఐదు",
        "ఎనభై ఆరు",
        "ఎనభై ఏడు",
        "ఎనభై ఎనిమిది",
        "ఎనభై తొమ్మిది",
        "తొంభై",
        "తొంభై ఒకటి",
        "తొంభై రెండు",
        "తొంభై మూడు",
        "తొంభై నాలుగు",
        "తొంభై ఐదు",
        "తొంభై ఆరు",
        "తొంభై ఏడు",
        "తొంభై ఎనిమిది",
        "తొంభై తొమ్మిది",
    ]
    # Indo-Aryan style fields kept for interface completeness; the fused
    # forms below (వంద/నూట/వందలు/వందల etc.) are what cardinal() actually uses.
    HUNDRED = "వంద"
    THOUSAND = "వెయ్యి"
    LAKH = "లక్ష"
    CRORE = "కోటి"

    LETTERS = {
        "A": "ఏ",
        "B": "బీ",
        "C": "సీ",
        "D": "డీ",
        "E": "ఈ",
        "F": "ఎఫ్",
        "G": "జీ",
        "H": "హెచ్",
        "I": "ఐ",
        "J": "జే",
        "K": "కే",
        "L": "ఎల్",
        "M": "ఎం",
        "N": "ఎన్",
        "O": "ఓ",
        "P": "పీ",
        "Q": "క్యూ",
        "R": "ఆర్",
        "S": "ఎస్",
        "T": "టీ",
        "U": "యూ",
        "V": "వీ",
        "W": "డబ్ల్యూ",
        "X": "ఎక్స్",
        "Y": "వై",
        "Z": "జెడ్",
    }

    MONTHS = [
        "జనవరి",
        "ఫిబ్రవరి",
        "మార్చి",
        "ఏప్రిల్",
        "మే",
        "జూన్",
        "జూలై",
        "ఆగస్టు",
        "సెప్టెంబర్",
        "అక్టోబర్",
        "నవంబర్",
        "డిసెంబర్",
    ]
    MONTH_ALIASES = {"సెప్టెంబరు": 9, "అక్టోబరు": 10, "నవంబరు": 11, "డిసెంబరు": 12}

    SYM = {
        "rupee": "రూపాయలు",
        "dollar": "డాలర్లు",
        "euro": "యూరోలు",
        "pound": "పౌండ్లు",
        "cent": "సెంట్లు",
        "percent": "శాతం",
        "paisa": "పైసలు",
        "point": "పాయింట్",
        "minus": "మైనస్",
        "plus": "ప్లస్",
        "equals": "సమానం",
        "times": "ఇంటూ",
        "divide": "భాగించగా",
        "squared": "వర్గం",
        "sqrt": "వర్గమూలం",
        "lt": "కంటే తక్కువ",
        "gt": "కంటే ఎక్కువ",
        "plus_minus": "ప్లస్ మైనస్",
        "at": "ఎట్ ద రేట్",
        "underscore": "అండర్ స్కోర్",
        "dot": "డాట్",
        "slash": "స్లాష్",
        "hash": "హ్యాష్",
        "amp": "మరియు",
        "half": "అర",
        "quarter": "పావు",
        "three_quarters": "ముప్పావు",
        "degree": "డిగ్రీలు",
        "celsius": "డిగ్రీల సెల్సియస్",
        "fahrenheit": "డిగ్రీల ఫారెన్‌హీట్",
        "minute": "నిమిషాలు",
        "per": "పర్",
        "second": "సెకన్లు",
    }

    TIME_ON_HOUR = "{h} గంటలు"
    TIME_HM = "{h} గంటల {m} నిమిషాలు"
    RANGE = "{a} నుంచి {b}"
    RANGE_TO = "{a} నుంచి {b} వరకు"
    RATIO = "{a} నిష్పత్తి {b}"
    POWER = "{base} పవర్ {exp}"
    FRACTION = "{num} బై {den}"
    DECADE_SHORT = "{tens}ల"
    DECADE_FULL = "{tens}ల దశకం"

    PERIODS = {
        "morning": "ఉదయం",
        "afternoon": "మధ్యాహ్నం",
        "evening": "సాయంత్రం",
        "night": "రాత్రి",
    }

    TIME_MARKERS = ["గంటలకు", "గంటలకి", "గంటకు", "గంటకి", "గంటల", "గంటలు", "గం"]
    MARKER_SUFFIXES = {
        "గంటలకు": "కు",
        "గంటలకి": "కి",
        "గంటకు": "కు",
        "గంటకి": "కి",
        "నిమిషాలకు": "కు",
        "నిమిషానికి": "కి",
    }
    PERIOD_DETECT = {
        "ఉదయం": "morning",
        "పొద్దున": "morning",
        "పొద్దున్న": "morning",
        "తెల్లవారుజామున": "morning",
        "మధ్యాహ్నం": "afternoon",
        "సాయంత్రం": "evening",
        "రాత్రి": "night",
    }
    YEAR_WORDS = ["సంవత్సరం", "సంవత్సరంలో", "ఏడాది", "సం"]
    ID_KEYWORDS = [
        "నంబర్",
        "నంబరు",
        "నెంబర్",
        "నెంబరు",
        "ఖాతా",
        "ఫోన్",
        "మొబైల్",
        "కోడ్",
        "ఆర్డర్",
        "ఎక్స్ప్రెస్",
        "express",
        "అకౌంట్",
        "టికెట్",
        "రిజిస్ట్రేషన్",
        "యూజర్",
        "వాయిదా",
        "అంకె",
        "లావాదేవీ",
        "పిన్",
    ]
    ID_AFTER_WORDS = [
        "ఎండ్",
        "ముగిసే",
        "ముగుస్తుంది",
        "చివర",
        "చివరి",
        "అంకెలతో",
        "డయల్",
        "పిన్",
        "పిన్‌కోడ్",
        "pin",
        "pincode",
        "కాల్",
        "సంప్రదించండి",
    ]
    LABEL_WORDS = [
        "గది",
        "వార్డు",
        "అధికరణ",
        "అధ్యాయం",
        "తరగతి",
        "భాగం",
        "సెక్షన్",
        "ప్లాట్ఫారమ్",
        "ప్లాట్‌ఫారమ్",
        "సీటు",
        "సీట్",
        "కోచ్",
        "గేట్",
        "రూమ్",
        "హాల్",
        "టెర్మినల్",
        "కౌంటర్",
    ]
    MINUTE_WORDS = ["నిమిషాలకు", "నిమిషానికి", "నిమిషాలు", "నిమిషాల", "నిమిషం"]
    SCALE_WORDS = {
        "కోటి": "crore",
        "కోటీ": "crore",
        "కోట్లు": "crore",
        "కోట్ల": "crore",
        "లక్ష": "lakh",
        "లక్షా": "lakh",
        "లక్షలు": "lakh",
        "లక్షల": "lakh",
        "వెయ్యి": "thousand",
        "వేలు": "thousand",
        "వేల": "thousand",
    }
    CURRENCY_WORDS = {
        "రూపాయలు": "rupee",
        "రూపాయల": "rupee",
        "రూపాయి": "rupee",
        "రూపాయ": "rupee",
        "రూ": "rupee",
    }

    ORDINAL_EXCEPTIONS = {1: "మొదటి"}
    ORDINAL_SUFFIX = "వ"
    ORDINAL_INPUT_SUFFIXES = ["వ", "వది", "ో"]

    # ==================================================================
    # Dravidian composition — fused/sandhi forms (overrides base).
    # ==================================================================

    def hundreds(self, h, rest):
        """100 వంద / నూట; 200+ వందలు / combining వందల."""
        if h == 1:
            if rest:
                return f"నూట {self.ONES[rest]}"
            return "వంద"
        if rest:
            return f"{self.ONES[h]} వందల {self.ONES[rest]}"
        return f"{self.ONES[h]} వందలు"

    def under_thousand(self, n):
        if n < 100:
            return self.ONES[n]
        return self.hundreds(n // 100, n % 100)

    def _thousands(self, t, has_rest):
        """t in 1..99. 1000 వెయ్యి (same before a remainder); 2+ వేలు/వేల."""
        if t == 1:
            return "వెయ్యి"
        return f"{self.ONES[t]} {'వేల' if has_rest else 'వేలు'}"

    def _lakhs(self, l, has_rest):
        """l in 1..99. లక్ష / sandhi లక్షా; 2+ లక్షలు/లక్షల."""
        if l == 1:
            return "లక్షా" if has_rest else "లక్ష"
        return f"{self.ONES[l]} {'లక్షల' if has_rest else 'లక్షలు'}"

    def _crores(self, c, has_rest):
        """కోటి / sandhi కోటీ; 2+ కోట్లు/కోట్ల."""
        if c == 1:
            return "కోటీ" if has_rest else "కోటి"
        return f"{self.cardinal(c)} {'కోట్ల' if has_rest else 'కోట్లు'}"

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 100:
            return self.ONES[n]
        if n < 1000:
            return self.under_thousand(n)
        crore, rest = divmod(n, 10**7)
        lakh, rest = divmod(rest, 10**5)
        thousand, sub = divmod(rest, 1000)
        parts = []
        if crore:
            parts.append(self._crores(crore, bool(lakh or thousand or sub)))
        if lakh:
            parts.append(self._lakhs(lakh, bool(thousand or sub)))
        if thousand:
            parts.append(self._thousands(thousand, bool(sub)))
        if sub:
            parts.append(self.under_thousand(sub))
        return " ".join(parts)

    def year(self, y):
        """Spoken Telugu reads 1100-1999 hundred-style — 1947 పందొమ్మిది
        వందల నలభై ఏడు (the "వెయ్యి తొమ్మిది వందల…" full cardinal sounds
        stiff; flagged by native review). 2000+ stays full cardinal:
        2026 రెండు వేల ఇరవై ఆరు."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            if lo:
                return f"{self.ONES[hi]} వందల {self.ONES[lo]}"
            return f"{self.ONES[hi]} వందలు"
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # Formal -వ ordinal with the regular Telugu stem change:
        # రెండు -> రెండవ, ఎనిమిది -> ఎనిమిదవ, ఇరవై -> ఇరవయ్యవ, వంద -> వందవ.
        if n in self.ORDINAL_EXCEPTIONS and not src_suffix:
            return self.ORDINAL_EXCEPTIONS[n]
        word = self.cardinal(n)
        if word.endswith("ు") or word.endswith("ి"):  # ు / ి matra
            stem = word[:-1]
        elif word.endswith("ై"):  # ై matra
            stem = word[:-1] + "య్య"
        else:
            stem = word
        return f"{stem}{self.ORDINAL_SUFFIX}"

    # ------------------------------------------------------------------
    # Inherited Telugu grammar hooks. Upstream cited an external review, not
    # independently verified human/native-speaker approval.
    # ------------------------------------------------------------------

    # metric units read as full words, not letter names
    UNIT_WORDS = {
        "KM": "కిలోమీటర్లు",
        "KG": "కిలోలు",
        "GM": "గ్రాములు",
        "MG": "మిల్లీగ్రాములు",
        "MCG": "మైక్రోగ్రాములు",
        "ML": "మిల్లీలీటర్లు",
        "CM": "సెంటీమీటర్లు",
        "MM": "మిల్లీమీటర్లు",
        "H": "అవర్",
        "HR": "అవర్",
    }

    NUMBER_WORD = "నంబర్"
    ABBREV_UNITS = {
        "MINS": "నిమిషాలు",
        "MIN": "నిమిషాలు",
        "HRS": "గంటలు",
        "HR": "గంట",
        "SECS": "సెకన్లు",
        "SEC": "సెకన్లు",
        "KMS": "కిలోమీటర్లు",
    }

    # singular noun forms used after "1" (ఒక రూపాయి, ఒక గంట)
    _NOUN_SINGULAR = {
        "రూపాయలు": "రూపాయి",
        "డాలర్లు": "డాలర్",
        "గంటలు": "గంట",
        "నిమిషాలు": "నిమిషం",
        "పైసలు": "పైసా",
        "సెకన్లు": "సెకను",
    }

    def _oblique_last(self, words):
        """Oblique stem for the final word before an appended noun:
        వందలు/వేలు/లక్షలు/కోట్లు -> వందల/వేల/లక్షల/కోట్ల ; ఒకటి -> ఒక."""
        last = words.rsplit(" ", 1)[-1]
        if last.endswith("లు"):
            return words[:-1]  # strip ు : వందలు -> వందల
        return words

    def decimal(self, int_part, frac_digits):
        frac = " ".join(self.digit(int(d)) for d in frac_digits)
        return (
            f"{self._oblique_last(self.cardinal(int_part))} "
            f"{self.SYM['point']} {frac}"
        )

    def join_amount(self, words, noun):
        if words == "ఒకటి":
            return f"ఒక {self._NOUN_SINGULAR.get(noun, noun)}"
        return f"{self._oblique_last(words)} {noun}"

    def time_phrase(self, h, m, period_key=None):
        h12 = h % 12 or 12
        hw = "ఒక" if h12 == 1 else self.two_digit(h12)
        hnoun = "గంట" if h12 == 1 else ("గంటలు" if m == 0 else "గంటల")
        if h12 == 1 and m != 0:
            hnoun = "గంట"
        if m == 0:
            core = f"{hw} {hnoun}"
        elif m == 1:
            core = f"{hw} {hnoun} ఒక నిమిషం"
        else:
            core = f"{hw} {hnoun} {self.two_digit(m)} నిమిషాలు"
        if period_key:
            return f"{self.PERIODS[period_key]} {core}"
        return core

    # postpositions appearing as a separate token right after a rendered
    # span get attached with correct sandhi ("గంటలు కి" -> "గంటలకు")
    _PP_SPACED = {
        "వరకు",
        "దాకా",
        "లోపు",
        "నుండి",
        "నుంచి",
        "పాటు",
        "క్రితం",
        "తర్వాత",
        "ముందు",
        "చొప్పున",
        "కంటే",
        "దగ్గర",
        "లోపల",
    }
    _PP_GLUE = {
        "కి",
        "కు",
        "ని",
        "న",
        "నే",
        "కే",
        "లో",
        "తో",
        "లోని",
        "లోకి",
        "లోనే",
        "కల్లా",
    }
    # గా keeps the nominative: రూపాయలు + గా -> రూపాయలుగా
    _PP_NOMINATIVE = {"గా", "గానే"}
    # common counted nouns: a plural-final cardinal right before these takes
    # the oblique stem (మూడు లక్షలు రూపాయలు -> మూడు లక్షల రూపాయలు)
    _COUNT_NOUNS = {
        "రూపాయలు",
        "మంది",
        "కిలోలు",
        "కిలోమీటర్లు",
        "గంటలు",
        "నిమిషాలు",
        "రోజులు",
        "సంవత్సరాలు",
        "ఏళ్లు",
        "సార్లు",
        "మార్కులు",
        "ఓట్లు",
        "సీట్లు",
        "పరుగులు",
        "వికెట్లు",
        "లీటర్లు",
        "గ్రాములు",
        "టన్నులు",
        "ఎకరాలు",
        "శాతం",
        "కోట్లు",
        "లక్షలు",
        "వడ్డీ",
        "జీతం",
        "ఖర్చు",
        "ధర",
        "ఆదాయం",
        "లాభం",
        "నష్టం",
        "పెట్టుబడి",
        "రుణం",
        "జరిమానా",
        "బడ్జెట్",
        "మోతాదు",
        "మాత్ర",
        "మాత్రలు",
        "టాబ్లెట్",
        "కోట్ల",
        "లక్షల",
        "దూరం",
        "బరువు",
        "వేగం",
        "మైళ్లు",
        "మైళ్ళు",
        "అడుగులు",
        "సెకన్లు",
        "పేజీలు",
        "సమయం",
        "సమయానికి",
    }

    def scale_noun(self, canon, numstr):
        one = numstr.replace(",", "") == "1"
        return {
            "crore": "కోటి" if one else "కోట్లు",
            "lakh": "లక్ష" if one else "లక్షలు",
            "thousand": "వెయ్యి" if one else "వేలు",
        }.get(canon, canon)

    _PP_ROMAN = {
        "ki": "కి",
        "ku": "కు",
        "ni": "ని",
        "lo": "లో",
        "ga": "గా",
        "ke": "కే",
        "varaku": "వరకు",
        "nunchi": "నుంచి",
        "nundi": "నుండి",
        "daggara": "దగ్గర",
    }

    def attach_postposition(self, rep, tok):
        tok = self._PP_ROMAN.get(tok.lower(), tok)
        last = rep.rsplit(" ", 1)[-1]
        if rep == "ఒకటి" and tok[:1].isascii() and tok[:1].isalpha():
            return "ఒక " + tok  # 1 litre -> ఒక litre
        if not last or not ("ఀ" <= last[-1] <= "౿"):
            return None  # last word not Telugu (Latin unit etc.)
        if last in self.LETTERS.values():
            return None  # bare letter names take no glued case
        if tok == last:
            return rep  # duplicate word (వరకు వరకు) — consume
        plural = last.endswith("లు")
        if tok in self._PP_NOMINATIVE:
            return rep + tok  # రూపాయలు గా -> రూపాయలుగా
        base_tok = tok.replace("\u200c", "")
        for end in ("ను", "ని", "కు", "కి", "లో", "గా", "తో", "పై"):
            if base_tok.endswith(end) and base_tok[: -len(end)] in self._COUNT_NOUNS:
                base_tok = base_tok[: -len(end)]
                break
        if plural and base_tok in self._COUNT_NOUNS:
            return rep[:-1] + " " + tok
        if rep == "ఒకటి" and tok in self._COUNT_NOUNS:
            return "ఒక " + self._NOUN_SINGULAR.get(tok, tok)
        if tok in self._PP_SPACED:
            return (rep[:-1] if plural else rep) + " " + tok
        if tok not in self._PP_GLUE:
            return None
        if last == "శాతం" and tok in ("కి", "కు"):
            return rep[:-1] + "ానికి"  # శాతం కి -> శాతానికి
        if plural:
            if tok in ("కి", "కు"):
                return rep[:-1] + tok  # keep the source marker: గంటలు కి -> గంటలకి
            if tok == "ని":
                return rep[:-1] + "ను"  # రూపాయలు ని -> రూపాయలను
            return rep[:-1] + tok  # గంటలు లో -> గంటలలో
        if tok in ("కి", "కు"):
            return rep + tok  # keep the source marker: ఐదు కి -> ఐదుకి
        return rep + tok


class Assamese(Bengali):
    """Assamese lexicon; Bengali grammar/entries remain the documented fallback.

    Number forms: AI4Bharat data; Gregorian months: Unicode CLDR as.xml.
    No new inference dependency and no translation of customer prose.
    """

    code = "as"
    name = "Assamese"
    if __package__:
        from .assamese_numbers import HUNDREDS, ONES
    else:
        from assamese_numbers import HUNDREDS, ONES
    HUNDRED = "শ"
    THOUSAND = "হাজাৰ"
    MONTHS = [
        "জানুৱাৰী",
        "ফেব্ৰুৱাৰী",
        "মাৰ্চ",
        "এপ্ৰিল",
        "মে’",
        "জুন",
        "জুলাই",
        "আগষ্ট",
        "ছেপ্তেম্বৰ",
        "অক্টোবৰ",
        "নৱেম্বৰ",
        "ডিচেম্বৰ",
    ]
    SYM = {
        **Bengali.SYM,
        "rupee": "টকা",
        "paisa": "পইচা",
        "dollar": "ডলাৰ",
        "euro": "ইউৰো",
        "pound": "পাউণ্ড",
    }
    CURRENCY_WORDS = {
        **Bengali.CURRENCY_WORDS,
        "টকা": "rupee",
        "টকাৰ": "rupee",
        "টকাত": "rupee",
        "ৰুপী": "rupee",
    }
    SCALE_WORDS = {**Bengali.SCALE_WORDS, "হাজাৰ": "thousand"}
    ID_KEYWORDS = Bengali.ID_KEYWORDS + ["নম্বৰ", "একাউণ্ট", "কাৰ্ড", "নম্বৰত", "অংক"]
    NUMBER_WORD = "নম্বৰ"
    YEAR_WORDS = Bengali.YEAR_WORDS + ["চন", "চনত", "বছৰ", "বছৰত"]
    TIME_HM = "{h} বাজি {m} মিনিট"
    RANGE = "{a}ৰ পৰা {b}"
    RANGE_TO = "{a}ৰ পৰা {b}লৈ"
    FRACTION = "{num} বাই {den}"
    SINGULAR_NOUNS = {}

    def hundreds(self, h, rest):
        return self.HUNDREDS[h] + (" " + self.two_digit(rest) if rest else "")

    def year(self, y):
        return self.cardinal(y)


LANGUAGES = {
    cls.code: cls()
    for cls in (
        Hindi,
        Bengali,
        Assamese,
        Gujarati,
        Kannada,
        Malayalam,
        Marathi,
        Odia,
        Punjabi,
        Tamil,
        Telugu,
    )
}
