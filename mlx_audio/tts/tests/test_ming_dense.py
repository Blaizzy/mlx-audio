import re
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.tts.models.bailingmm.bailingmm import (
    _MING3_VOICE_RE,
    CFM,
    MING3_MAX_CACHE_LEN,
    DiT,
    Encoder,
    FlowLoss,
    Model,
    RMSNorm,
    RotaryEmbedding,
    _add_period,
    _is_style_instruction,
    _semantic_length,
    _split_fragments,
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

    def test_prompt_projection_preserves_bfloat16_conditioning(self):
        model = make_model(chat_layout=True, ming3=True)
        model.model.model.word_embeddings.set_dtype(mx.bfloat16)
        model.linear_proj_audio = MagicMock(
            return_value=mx.zeros((1, 1, 8), dtype=mx.bfloat16)
        )
        _, embeds = model._prepare_input_embed(
            PROMPT, "Hi.", prompt_text="Ref.", prompt_latent=mx.ones((1, 1, 4))
        )
        self.assertEqual(model.linear_proj_audio.call_args.args[0].dtype, mx.bfloat16)
        self.assertEqual(embeds.dtype, mx.bfloat16)

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
    def test_ip_caption_format(self):
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

    def test_trim_drops_partial_frame(self):
        trimmed = _trim_trailing_silence(mx.ones((555,)), 1000)
        self.assertEqual(trimmed.shape[0], 500)


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

    def test_generate_defaults_and_explicit_budgets(self):
        model = make_model(chat_layout=True, ming3=True)
        model.audio = SimpleNamespace(sample_rate=1000, decode=MagicMock())
        model.audio.decode.return_value = (mx.ones((1, 100)), None, None)
        latent = mx.zeros((1, 1, 4))
        model.sample = MagicMock(side_effect=lambda **_: iter([(latent, True)]))

        list(model.generate("Yes."))
        options = model.sample.call_args.kwargs
        for name, value in {
            "max_decode_steps": 1000,
            "cfg": 2.0,
            "sigma": 0.25,
            "temperature": 0.0,
            "flow_steps": 10,
        }.items():
            self.assertEqual(options[name], value)

        for options, expected in (
            ({"max_tokens": 1200}, 1200),
            ({"max_tokens": 10}, 10),
            ({"max_tokens": 10, "max_decode_steps": 300}, 300),
        ):
            list(model.generate("Yes.", **options))
            self.assertEqual(
                model.sample.call_args.kwargs["max_decode_steps"], expected
            )

        model.sample.reset_mock()
        list(model.generate("one two three four five six seven eight. " * 4))
        self.assertEqual(model.sample.call_count, 2)
        self.assertTrue(
            all(
                _semantic_length(call.kwargs["text"]) <= 24
                for call in model.sample.call_args_list
            )
        )

    def test_raw_text_and_silence_trim_switches(self):
        model = make_model(chat_layout=True, ming3=True)
        model.audio = SimpleNamespace(sample_rate=1000, decode=MagicMock())
        model.audio.decode.return_value = (mx.ones((1, 555)), None, None)
        latent = mx.zeros((1, 1, 4))
        model.sample = MagicMock(side_effect=lambda **_: iter([(latent, True)]))
        (result,) = list(model.generate("Yes", raw_text=True, trim_silence=False))
        self.assertEqual(model.sample.call_args.kwargs["text"], "Yes")
        self.assertEqual(result.samples, 555)
        (result,) = list(model.generate("Yes"))
        self.assertEqual(model.sample.call_args.kwargs["text"], "Yes.")
        self.assertEqual(result.samples, 500)

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

        results = model.generate(text, stream=True, max_length=50)
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
        (joined,) = list(model.generate(text, max_length=50))
        self.assertEqual(joined.samples, 300)
        self.assertFalse(joined.is_streaming_chunk)
        self.assertTrue(joined.is_final_chunk)


class ConstantDiT(nn.Module):
    forward_with_cfg = DiT.forward_with_cfg

    def __call__(self, x, t, c, latent_history):
        prefix = mx.full((x.shape[0], 2, x.shape[-1]), 100.0, dtype=x.dtype)
        prediction = mx.broadcast_to(c.mean(axis=-1, keepdims=True), x.shape)
        return mx.concatenate([prefix, prediction], axis=1)


class TestMing3FlowSampling(unittest.TestCase):
    def test_reference_audio_encoder_preserves_model_dtype(self):
        encoder = Encoder(
            dict(
                model_type="qwen2",
                hidden_size=8,
                num_hidden_layers=1,
                intermediate_size=16,
                num_attention_heads=2,
                num_key_value_heads=2,
                rms_norm_eps=1e-6,
                vocab_size=1,
            ),
            input_dim=4,
            hop_size=4,
            latent_dim=2,
            patch_size=-1,
        )
        encoder.set_dtype(mx.bfloat16)
        output = encoder(mx.ones((1, 16)))
        self.assertEqual(output.dtype, mx.bfloat16)

    def test_bfloat16_rms_norm_values(self):
        norm = RMSNorm(4)
        norm.weight = mx.array([0.125, 1.75, 0.3, -0.5], dtype=mx.bfloat16)
        output = norm(mx.array([[0.123, 1.5, -2.1, 3.3]], dtype=mx.bfloat16))
        np.testing.assert_allclose(
            np.array(output.astype(mx.float32)),
            [[0.007354736328125, 1.2578125, -0.30078125, -0.7890625]],
            rtol=8e-3,  # Allow one BF16 ULP across CPU and Metal reductions.
            atol=0,
        )

    def test_bfloat16_dit_does_not_promote_the_autoregressive_state(self):
        dit = DiT(4, hidden_size=8, depth=1, num_heads=2, llm_cond_dim=8)
        dit.set_dtype(mx.bfloat16)
        dit.eval()
        output = dit.forward_with_cfg(
            x=mx.ones((1, 1, 4), dtype=mx.bfloat16),
            t=mx.array(0.25, dtype=mx.bfloat16),
            c=mx.ones((1, 1, 8), dtype=mx.bfloat16),
            latent_history=mx.zeros((1, 1, 4), dtype=mx.bfloat16),
            cfg_scale=2.0,
            patch_size=1,
        )
        self.assertEqual(output.dtype, mx.bfloat16)

    def test_rotary_preserves_query_and_key_dtype(self):
        rotary = RotaryEmbedding(4)
        query = mx.ones((1, 2, 3, 4), dtype=mx.bfloat16)
        q, k = rotary(query, query)
        self.assertEqual(q.dtype, query.dtype)
        self.assertEqual(k.dtype, query.dtype)

    def test_guidance_zero_one_and_two(self):
        for dtype in (mx.float32, mx.bfloat16):
            for cfg in (0.0, 1.0, 2.0):
                with self.subTest(dtype=dtype, cfg=cfg):
                    flow = CFM(ConstantDiT(), ming3_sampling=True)
                    latent, _ = flow.sample(
                        noise=mx.zeros((1, 2, 4), dtype=dtype),
                        c=mx.ones((1, 1, 8), dtype=dtype),
                        latent_history=mx.zeros((1, 1, 4), dtype=dtype),
                        cfg_scale=cfg,
                        patch_size=2,
                        steps=1,
                        temperature=0.0,
                        sde_noise=mx.zeros((1, 1, 2, 4), dtype=dtype),
                    )
                    self.assertEqual(latent.shape, (1, 2, 4))
                    self.assertEqual(latent.dtype, dtype)
                    np.testing.assert_array_equal(
                        np.array(latent.astype(mx.float32)), 1.0 + cfg
                    )

    def test_noise_is_drawn_up_front_in_latent_shape_and_dtype(self):
        flow = FlowLoss(4, 8, ming3_sampling=True, hidden_size=8, depth=0)
        flow.cfm.sample = MagicMock(return_value=(None, None))
        with patch(
            "mlx_audio.tts.models.bailingmm.bailingmm.mx.random.normal",
            side_effect=lambda shape, dtype: mx.zeros(shape, dtype=dtype),
        ) as normal:
            flow.sample(
                mx.zeros((1, 1, 8), dtype=mx.bfloat16),
                mx.zeros((1, 1, 4), dtype=mx.bfloat16),
                patch_size=2,
                steps=10,
                temperature=0.0,
            )
        self.assertEqual(
            normal.call_args_list,
            [
                unittest.mock.call((1, 2, 4), dtype=mx.bfloat16),
                unittest.mock.call((10, 1, 2, 4), dtype=mx.bfloat16),
            ],
        )

    def test_final_sde_noise_is_included(self):
        flow = CFM(ConstantDiT(), ming3_sampling=True)
        latent, _ = flow.sample(
            noise=mx.zeros((1, 1, 4)),
            c=mx.zeros((1, 1, 8)),
            latent_history=mx.zeros((1, 1, 4)),
            steps=1,
            cfg_scale=2.0,
            sigma=0.25,
            temperature=4.0,
            sde_noise=mx.ones((1, 1, 1, 4)),
        )
        np.testing.assert_array_equal(np.array(latent), 0.5)


class TestMing3StopHead(unittest.TestCase):
    def make_sampler(self, logits):
        model = make_model(chat_layout=True, ming3=True)
        model.llm_args.num_hidden_layers = 0
        model.history_patch_size = 1
        model._prepare_input_embed = MagicMock(
            return_value=(None, mx.zeros((1, 2, 8), dtype=mx.bfloat16))
        )
        model._forward_llm_hidden = MagicMock(
            return_value=mx.zeros((1, 1, 8), dtype=mx.bfloat16)
        )
        latent = mx.full((1, 1, 4), 7.0, dtype=mx.bfloat16)
        model.flowloss = SimpleNamespace(sample=MagicMock(return_value=(latent, None)))
        model.stop_head = MagicMock(side_effect=[mx.array([[v]]) for v in logits])
        return model

    def test_earliest_stop_is_fifth_patch_and_keeps_that_patch(self):
        model = self.make_sampler([[0.0, 1.0]] * 10)
        patches = list(model.sample(PROMPT, "Yes.", max_decode_steps=10))
        self.assertEqual([last for _, last in patches], [False] * 4 + [True])
        self.assertEqual(patches[-1][0][0, 0, 0].item(), 7.0)
        self.assertEqual(model.flowloss.sample.call_count, 5)

    def test_half_probability_does_not_stop(self):
        model = self.make_sampler([[0.0, 0.0]] * 6 + [[0.0, 1.0]])
        patches = list(model.sample(PROMPT, "Yes.", max_decode_steps=10))
        self.assertEqual([last for _, last in patches], [False] * 6 + [True])

    def test_direct_sampler_default_uses_remaining_cache_budget(self):
        model = self.make_sampler([[0.0, 0.0]] * MING3_MAX_CACHE_LEN)
        patches = list(model.sample(PROMPT, "Yes."))
        self.assertEqual(len(patches), MING3_MAX_CACHE_LEN - 2)
        self.assertFalse(any(last for _, last in patches))

    def test_history_uses_reference_tail_and_hidden_dtype(self):
        model = self.make_sampler([[1.0, 0.0]] * 2)
        reference = mx.arange(12, dtype=mx.float32).reshape(1, 3, 4)
        list(
            model.sample(
                PROMPT,
                "Yes.",
                prompt_text="Ref.",
                prompt_latent=reference,
                max_decode_steps=2,
            )
        )
        first, second = model.flowloss.sample.call_args_list
        self.assertEqual(first.args[1].dtype, mx.bfloat16)
        np.testing.assert_array_equal(
            np.array(first.args[1].astype(mx.float32)), [[[8, 9, 10, 11]]]
        )
        np.testing.assert_array_equal(
            np.array(second.args[1].astype(mx.float32)), [[[7, 7, 7, 7]]]
        )


if __name__ == "__main__":
    unittest.main()
