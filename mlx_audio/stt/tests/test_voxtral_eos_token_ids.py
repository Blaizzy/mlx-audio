"""Unit tests for `_ensure_eos_token_ids_list` in
`mlx_audio.stt.models.voxtral.voxtral`.

`transformers.PreTrainedTokenizerBase` defines `__getattr__` and
`__setattr__` overrides that special-case attribute names ending in
`_id` / `_ids`:

  * `__getattr__` collapses any missing `*_id` / `*_ids` lookup to
    the corresponding single-token id, so `getattr(tok,
    "eos_token_ids", default)` returns the int `eos_token_id`
    instead of falling back to `default`.
  * `__setattr__` rejects non-string values for any attribute name
    ending in `_id` / `_ids`, raising
    `ValueError("Cannot set a non-string value as the eos_token")`.

`_ensure_eos_token_ids_list` bypasses both by writing through
`tokenizer.__dict__`. These tests exercise the production helper
against a real `PreTrainedTokenizerFast` instance (no model
download).
"""

import unittest

from mlx_audio.stt.models.voxtral.voxtral import (
    _VOXTRAL_EOS_TOKEN_IDS,
    _ensure_eos_token_ids_list,
)


def _make_tokenizer():
    """Build a minimal real `PreTrainedTokenizerFast` that exhibits
    `PreTrainedTokenizerBase`'s `__getattr__` / `__setattr__` magic.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    inner = Tokenizer(
        WordLevel(
            vocab={"<unk>": 0, "<s>": 1, "</s>": 2},
            unk_token="<unk>",
        )
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=inner,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )


class TestBaselineMagic(unittest.TestCase):
    """Demonstrates the underlying base-class behavior that
    `_ensure_eos_token_ids_list` exists to bypass.
    """

    def test_getattr_collapses_eos_token_ids_to_int(self):
        tok = _make_tokenizer()
        # The default list is NEVER returned, regardless of magic on
        # `_ids`-suffixed attribute access:
        result = getattr(tok, "eos_token_ids", [2, 4, 32000])
        self.assertIsInstance(result, int)
        self.assertEqual(result, tok.eos_token_id)
        self.assertNotEqual(result, [2, 4, 32000])

    def test_setattr_rejects_non_string_eos_token_ids(self):
        tok = _make_tokenizer()
        with self.assertRaises(ValueError):
            tok.eos_token_ids = [2, 4, 32000]


class TestEnsureEosTokenIdsList(unittest.TestCase):
    """Exercises the production helper directly."""

    def test_initializes_to_default_when_absent(self):
        tok = _make_tokenizer()
        result = _ensure_eos_token_ids_list(tok)
        self.assertEqual(result, _VOXTRAL_EOS_TOKEN_IDS)
        self.assertEqual(tok.__dict__["eos_token_ids"], _VOXTRAL_EOS_TOKEN_IDS)

    def test_readback_via_attribute_returns_list(self):
        tok = _make_tokenizer()
        _ensure_eos_token_ids_list(tok)
        # __getattr__ no longer fires (attribute now in __dict__),
        # so the read returns the list, not the int.
        self.assertIsInstance(tok.eos_token_ids, list)
        self.assertEqual(tok.eos_token_ids, _VOXTRAL_EOS_TOKEN_IDS)

    def test_iteration_works_for_generation_loop(self):
        # Downstream caller (`generate.py`) does
        #   if token in eos_token_ids:
        # which previously failed with TypeError: 'int' is not iterable.
        tok = _make_tokenizer()
        _ensure_eos_token_ids_list(tok)
        self.assertIn(2, tok.eos_token_ids)
        self.assertNotIn(99, tok.eos_token_ids)
        self.assertEqual(list(tok.eos_token_ids), _VOXTRAL_EOS_TOKEN_IDS)

    def test_preserves_existing_list(self):
        tok = _make_tokenizer()
        tok.__dict__["eos_token_ids"] = [10, 20]
        _ensure_eos_token_ids_list(tok)
        self.assertEqual(tok.eos_token_ids, [10, 20])

    def test_preserves_existing_tuple_as_list(self):
        tok = _make_tokenizer()
        tok.__dict__["eos_token_ids"] = (10, 20)
        _ensure_eos_token_ids_list(tok)
        self.assertEqual(tok.__dict__["eos_token_ids"], [10, 20])
        self.assertIsInstance(tok.__dict__["eos_token_ids"], list)

    def test_preserves_existing_set_sorted_as_list(self):
        tok = _make_tokenizer()
        tok.__dict__["eos_token_ids"] = {30, 10, 20}
        _ensure_eos_token_ids_list(tok)
        self.assertEqual(tok.__dict__["eos_token_ids"], [10, 20, 30])

    def test_custom_default_is_used(self):
        tok = _make_tokenizer()
        _ensure_eos_token_ids_list(tok, default=[5, 6, 7])
        self.assertEqual(tok.__dict__["eos_token_ids"], [5, 6, 7])


if __name__ == "__main__":
    unittest.main()
