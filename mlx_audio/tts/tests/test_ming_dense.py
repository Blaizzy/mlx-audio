import re
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.bailingmm.bailingmm import (
    _MING3_VOICE_RE,
    MING3_BASE_STEPS,
    MING3_MAX_CACHE_LEN,
    MING3_STEPS_PER_UNIT,
    Model,
    _add_period,
    _is_style_instruction,
    _semantic_length,
    _split_fragments,
    _spoken_units,
    _trim_trailing_silence,
    ming3_instruction,
)

SPECIAL = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|vision_start|>",
    "<|vision_pad|>",
    "<|vision_end|>",
    "<role>",
    "</role>",
    "<spk>",
    "</spk>",
    "<audio>",
    "<audioPatch>",
]
SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL) + ")")


class FakeTokenizer:
    """Specials get their own id; every other character is one token."""

    def __init__(self):
        self.vocab = {tok: i for i, tok in enumerate(SPECIAL)}

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in SPECIAL_RE.split(text):
            if part in self.vocab:
                ids.append(self.vocab[part])
            else:
                ids.extend(len(SPECIAL) + ord(c) for c in part)
        return ids

    def decode(self, ids):
        inv = {i: tok for tok, i in self.vocab.items()}
        return "".join(inv[i] if i in inv else chr(i - len(SPECIAL)) for i in ids)


def make_model(chat_layout, ming3, hidden=8, latent_dim=4):
    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.tokenizer = FakeTokenizer()
    model.chat_layout = chat_layout
    model.ming3 = ming3
    model.patch_size = 1 if ming3 else 4
    model.latent_dim = latent_dim
    model.llm_args = SimpleNamespace(hidden_size=hidden)
    model.model = SimpleNamespace(
        model=SimpleNamespace(word_embeddings=nn.Embedding(70000, hidden))
    )
    model.linear_proj_audio = lambda x: mx.zeros((x.shape[0], hidden))
    return model


PROMPT = "Please generate speech based on the following description.\n"


class TestMingPromptLayout(unittest.TestCase):
    def decode(self, model, **kwargs):
        ids, embeds = model._prepare_input_embed(PROMPT, "Hi.", **kwargs)
        self.assertEqual(embeds.shape[1], ids.shape[1])
        return model.tokenizer.decode(ids[0].tolist()), embeds

    def test_dense_uses_qwen_chat_template(self):
        text, _ = self.decode(make_model(chat_layout=True, ming3=True))
        self.assertEqual(
            text,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + PROMPT + " Text input:\nHi.<|im_end|>\n"
            "<|im_start|>assistant\n<audio>",
        )

    def test_moe_keeps_role_turns(self):
        text, _ = self.decode(make_model(chat_layout=False, ming3=False))
        self.assertTrue(text.startswith("<role>HUMAN</role>" + PROMPT))
        self.assertTrue(text.endswith("Hi.<role>ASSISTANT</role><audio>"))

    def test_instruction_terminator(self):
        ming3, _ = self.decode(
            make_model(chat_layout=True, ming3=True), instruction="Warm."
        )
        old, _ = self.decode(
            make_model(chat_layout=True, ming3=False), instruction="Warm."
        )
        self.assertTrue(ming3.endswith("assistant\nWarm.<|im_end|><audio>"))
        self.assertTrue(old.endswith("assistant\nWarm.<|endoftext|><audio>"))

    def test_speaker_slot_replaces_pad_row(self):
        model = make_model(chat_layout=True, ming3=True)
        emb = mx.full((8,), 7.0)
        text, embeds = self.decode(model, spk_emb=[emb])
        self.assertIn(
            PROMPT + "  speaker_1:<|vision_start|><|vision_pad|><|vision_end|>\n"
            " Text input:\n",
            text,
        )
        ids = model.tokenizer.encode(text)
        pos = ids.index(model.tokenizer.convert_tokens_to_ids("<|vision_pad|>"))
        self.assertTrue(mx.all(embeds[0, pos] == 7.0).item())
        self.assertFalse(mx.all(embeds[0, pos + 1] == 7.0).item())

    def test_prompt_latents_follow_audio_token(self):
        model = make_model(chat_layout=True, ming3=True)
        text, embeds = self.decode(
            model, prompt_text="Ref.", prompt_latent=mx.ones((1, 3, 4))
        )
        self.assertIn(" Text input:\nRef.Hi.<|im_end|>", text)
        self.assertTrue(text.endswith("<audio>" + "<audioPatch>" * 3))

    def test_zero_speaker_embedding_is_projected_zero(self):
        model = make_model(chat_layout=True, ming3=True)
        (emb,) = model._speaker_embeddings(None, use_zero_spk_emb=True)
        self.assertEqual(emb.shape, (8,))
        self.assertTrue(mx.all(emb == 0).item())
        self.assertIsNone(model._speaker_embeddings(None, use_zero_spk_emb=False))

    def test_prompt_padding_side(self):
        wav = mx.ones((1, 100))
        model = make_model(chat_layout=True, ming3=True)
        model.audio = SimpleNamespace(sample_rate=44100)
        padded = model._pad_prompt_waveform(wav)
        self.assertEqual(padded.shape[1], 3528)
        self.assertEqual(padded[0, 0].item(), 0.0)
        self.assertEqual(padded[0, -1].item(), 1.0)

        old = make_model(chat_layout=True, ming3=False)
        old.audio = SimpleNamespace(sample_rate=44100)
        padded = old._pad_prompt_waveform(wav)
        self.assertEqual(padded.shape[1], 14112)
        self.assertEqual(padded[0, 0].item(), 1.0)
        self.assertEqual(padded[0, -1].item(), 0.0)


class TestMing3Instruction(unittest.TestCase):
    def test_ip_caption_matches_reference(self):
        # Output of Ming-3.0-tts tts_tasks.build_task_instruction("IP", ip="US_James").
        self.assertEqual(
            ming3_instruction(ip="US_James"),
            '{"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": null, '
            '"风格": null, "语速": null, "基频": null, "音量": null, "情感": null, '
            '"BGM": {"Genre": null, "Mood": null, "Instrument": null, "Theme": null, '
            '"ENV": null, "SNR": null}, "IP": "US_James"}]}',
        )

    def test_dialect_uses_english_schema(self):
        caption = ming3_instruction(dialect="cantonese")
        self.assertTrue(caption.startswith('{"audio_sequence": [{"id": 1, '))
        self.assertIn('"dialect": "cantonese"', caption)
        self.assertTrue(caption.endswith('"ip": null}]}'))


class TestMingText(unittest.TestCase):
    def test_add_period(self):
        self.assertEqual(_add_period("Hello"), "Hello.")
        self.assertEqual(_add_period("你好"), "你好。")
        self.assertEqual(_add_period("Hello,"), "Hello.")
        self.assertEqual(_add_period("Really?"), "Really?")
        self.assertEqual(_add_period("[Laughter]"), "[Laughter]")

    def test_short_text_is_one_fragment(self):
        self.assertEqual(_split_fragments("One. Two.", 50), ["One. Two."])

    def test_fragments_respect_length_and_keep_text(self):
        text = " ".join(f"Sentence number {i} has five words." for i in range(30))
        fragments = _split_fragments(text, 50)
        self.assertGreater(len(fragments), 1)
        self.assertTrue(all(_semantic_length(f) <= 50 for f in fragments))
        self.assertEqual(" ".join(fragments), text)

    def test_long_sentence_splits_at_commas(self):
        text = ", ".join(["one two three four five"] * 15) + "."
        fragments = _split_fragments(text, 20)
        self.assertTrue(all(_semantic_length(f) <= 20 for f in fragments))
        self.assertEqual(" ".join(fragments), text)

    def test_long_clause_splits_at_words_without_dropping_text(self):
        text = " ".join(["hello"] * 600) + ". Goodbye."
        fragments = _split_fragments(text, 50)
        self.assertTrue(all(_semantic_length(f) <= 50 for f in fragments))
        self.assertEqual(" ".join(fragments), text)

    def test_long_chinese_clause_splits_at_characters(self):
        text = "我们今天去公园散步" * 20 + "。"
        fragments = _split_fragments(text, 20)
        self.assertTrue(all(_semantic_length(f) <= 20 for f in fragments))
        self.assertEqual("".join(fragments), text)

    def test_word_splitting_preserves_numbers_abbreviations_and_tags(self):
        text = " ".join(
            ["It costs 1,000 dollars or 3.14 pounds in the U.S. [Surprise-ah] Wait..."]
            * 10
        )
        fragments = _split_fragments(text, 7)
        self.assertTrue(all(_semantic_length(f) <= 7 for f in fragments))
        self.assertEqual(" ".join(fragments), text)

    def test_fragment_length_must_be_positive(self):
        for max_length in (0, -1):
            with self.subTest(max_length=max_length):
                with self.assertRaisesRegex(ValueError, "max_length"):
                    _split_fragments("Hello.", max_length)

    def test_chinese_fragments_join_without_spaces(self):
        text = "今天天气很好。" * 12
        fragments = _split_fragments(text, 20)
        self.assertGreater(len(fragments), 1)
        self.assertEqual("".join(fragments), text)

    def test_fragments_keep_numbers_abbreviations_and_ellipses(self):
        text = (
            "The ratio is 3.14 and it costs 1,000 dollars in the U.S. today. "
            "Wait... then we go. "
        ) * 6
        fragments = _split_fragments(text, 20)
        self.assertGreater(len(fragments), 1)
        self.assertEqual(" ".join(fragments), " ".join(text.split()))
        joined = " | ".join(fragments)
        self.assertNotIn("3. 14", joined)
        self.assertNotIn("1, 000", joined)
        self.assertNotIn(". . .", joined)

    def test_mixed_chinese_english_fragments_keep_spaces(self):
        text = "我们今天去公园散步。 Then we walked home together slowly. " * 5
        fragments = _split_fragments(text, 15)
        self.assertGreater(len(fragments), 1)
        self.assertEqual(" ".join(fragments), " ".join(text.split()))

    def test_style_instructions_are_not_split(self):
        self.assertFalse(_is_style_instruction(None))
        self.assertTrue(_is_style_instruction("Speak warmly and slowly."))
        self.assertTrue(_is_style_instruction(ming3_instruction(style="温柔的女声")))
        self.assertFalse(_is_style_instruction(ming3_instruction(emotion="高兴")))
        self.assertFalse(_is_style_instruction(ming3_instruction(ip="US_James")))

    def test_voice_names(self):
        for name in ("US_James", "UK_Charlotte", "灵小柔", "灵信远_高兴"):
            self.assertTrue(_MING3_VOICE_RE.match(name), name)
        for name in ("alloy", "af_heart", "conversational_a", "vivian", ""):
            self.assertFalse(_MING3_VOICE_RE.match(name), name)

    def test_trim_trailing_silence_keeps_tail(self):
        sr = 1000
        speech = mx.concatenate([mx.ones((500,)), mx.zeros((2000,))])
        trimmed = _trim_trailing_silence(speech, sr)
        self.assertEqual(trimmed.shape[0], 500 + 300)

    def test_spoken_units_count_kana_hangul_and_digits(self):
        self.assertEqual(_spoken_units("Yes."), 1)
        self.assertEqual(_spoken_units("好的。"), 2)
        self.assertEqual(_spoken_units("ありがとう"), 5)
        self.assertEqual(_spoken_units("안녕하세요"), 5)
        self.assertEqual(_spoken_units("Call 555."), 4)


class TestMing3GenerationBudget(unittest.TestCase):
    def test_full_or_oversized_prompt_is_rejected_before_inference(self):
        model = make_model(chat_layout=True, ming3=True)
        model.llm_args.num_hidden_layers = 0
        model._forward_llm_hidden = MagicMock()
        ids, _ = model._prepare_input_embed(PROMPT, "")
        for prompt_length in (MING3_MAX_CACHE_LEN, MING3_MAX_CACHE_LEN + 1):
            with self.subTest(prompt_length=prompt_length):
                text = "x" * (prompt_length - ids.shape[1])
                with self.assertRaisesRegex(ValueError, "512.*Shorten"):
                    list(model.sample(PROMPT, text))
        model._forward_llm_hidden.assert_not_called()

    def test_sampling_respects_remaining_cache_and_requested_steps(self):
        model = make_model(chat_layout=True, ming3=True)
        model.llm_args.num_hidden_layers = 0
        model.history_patch_size = 1
        model._forward_llm_hidden = MagicMock(return_value=mx.zeros((1, 1, 8)))
        model.flowloss = SimpleNamespace(
            sample=MagicMock(return_value=(mx.zeros((1, 1, 4)), None))
        )
        model.stop_head = lambda _: mx.zeros((1, 1, 2))
        ids, _ = model._prepare_input_embed(PROMPT, "")
        for remaining, requested in ((1, 20), (2, 20), (2, 1)):
            with self.subTest(remaining=remaining, requested=requested):
                text = "x" * (MING3_MAX_CACHE_LEN - remaining - ids.shape[1])
                latents = list(model.sample(PROMPT, text, max_decode_steps=requested))
                self.assertEqual(len(latents), min(remaining, requested))

    def test_generate_does_not_skip_an_empty_fragment(self):
        model = make_model(chat_layout=True, ming3=True)
        model.audio = SimpleNamespace(sample_rate=1000, decode=MagicMock())
        model.audio.decode.return_value = (mx.ones((1, 100)), None, None)
        latent = mx.zeros((1, 1, 4))
        model.sample = MagicMock(side_effect=[iter(()), iter([(latent, True)])])
        text = " ".join(["hello"] * 50) + ". Goodbye."
        with self.assertRaisesRegex(RuntimeError, "fragment"):
            next(model.generate(text))
        model.sample.assert_called_once()
        model.audio.decode.assert_not_called()

    def test_generate_caps_steps_by_text_length(self):
        model = make_model(chat_layout=True, ming3=True)
        model.audio = SimpleNamespace(sample_rate=1000, decode=MagicMock())
        model.audio.decode.return_value = (mx.ones((1, 100)), None, None)
        latent = mx.zeros((1, 1, 4))
        model.sample = MagicMock(side_effect=lambda **_: iter([(latent, True)]))

        def requested_steps(**kwargs):
            list(model.generate("Yes.", **kwargs))
            return model.sample.call_args.kwargs["max_decode_steps"]

        cap = MING3_BASE_STEPS + MING3_STEPS_PER_UNIT
        self.assertEqual(requested_steps(), cap)
        # Servers always send max_tokens; it only lowers the cap.
        self.assertEqual(requested_steps(max_tokens=1200), cap)
        self.assertEqual(requested_steps(max_tokens=10), 10)
        self.assertEqual(requested_steps(max_decode_steps=300), 300)

    def test_stream_yields_each_fragment_as_it_is_decoded(self):
        model = make_model(chat_layout=True, ming3=True)
        model.audio = SimpleNamespace(sample_rate=1000, decode=MagicMock())
        model.audio.decode.side_effect = [
            (mx.full((1, 100), 1.0), None, None),
            (mx.full((1, 200), 2.0), None, None),
        ]
        latent = mx.zeros((1, 1, 4))
        model.sample = MagicMock(side_effect=lambda **_: iter([(latent, True)]))
        text = " ".join(["hello"] * 50) + ". Goodbye."

        results = model.generate(text, stream=True)
        first = next(results)
        self.assertEqual(model.audio.decode.call_count, 1)
        self.assertEqual(first.samples, 100)
        self.assertEqual(first.segment_idx, 0)
        self.assertTrue(first.is_streaming_chunk)
        self.assertFalse(first.is_final_chunk)

        rest = list(results)
        self.assertEqual(len(rest), 1)
        self.assertEqual(rest[0].samples, 200)
        self.assertEqual(rest[0].segment_idx, 1)
        self.assertTrue(rest[0].is_final_chunk)

        model.audio.decode.side_effect = [
            (mx.full((1, 100), 1.0), None, None),
            (mx.full((1, 200), 2.0), None, None),
        ]
        (joined,) = list(model.generate(text))
        self.assertEqual(joined.samples, 300)
        self.assertFalse(joined.is_streaming_chunk)
        self.assertTrue(joined.is_final_chunk)


if __name__ == "__main__":
    unittest.main()
