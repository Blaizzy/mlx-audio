import unittest


class TestIndexTTSTextFrontend(unittest.TestCase):
    def test_pronunciation_annotations(self):
        from mlx_audio.tts.models.indextts2.text_frontend import (
            apply_pronunciation_annotations,
        )

        text = "行<行|XING2>走 and <going|G OW1 . IH0 NG>"
        result = apply_pronunciation_annotations(text)
        self.assertIn("<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>", result)
        self.assertIn("<|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|>", result)

    def test_kana_passthrough(self):
        from mlx_audio.tts.models.indextts2.text_frontend import (
            apply_pronunciation_annotations,
        )

        self.assertEqual(apply_pronunciation_annotations("<今日|きょう>"), " きょう ")

    def test_language_ids_are_stable(self):
        from mlx_audio.tts.models.indextts2.text_frontend import (
            LANGUAGES,
            lang_to_token,
        )

        self.assertEqual(list(LANGUAGES)[:5], ["en", "zh", "de", "es", "ru"])
        self.assertEqual(lang_to_token("ZH"), lang_to_token("zh"))
        self.assertEqual(lang_to_token("unknown"), lang_to_token("common"))

    def test_split_keeps_pronunciation_token_intact(self):
        from mlx_audio.tts.models.indextts2.text_frontend import split_text_by_tokens

        text = "第一句。<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>第二句。"

        def token_len(value):
            return 1 if value.startswith("<|SPECIAL_TOKEN") else len(value)

        chunks = split_text_by_tokens(
            text,
            max_tokens=6,
            token_len=token_len,
            capacity=32,
        )
        self.assertTrue(
            any("<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>" in c for c in chunks)
        )
        self.assertEqual("".join(chunks), text)

    def test_case_normalization(self):
        from mlx_audio.tts.models.indextts2.text_frontend import normalize_case

        self.assertEqual(normalize_case("Hello", "en"), "hello")
        self.assertEqual(normalize_case("hola", "es"), "HOLA")
        self.assertEqual(normalize_case("Hallo", "de"), "Hallo")


class TestIndexTTSConfig(unittest.TestCase):
    def test_2_5_defaults_select_multilingual_mode(self):
        from mlx_audio.tts.models.indextts2.config import ModelConfig

        cfg = ModelConfig.from_dict({"model_type": "indextts2_5"})
        self.assertEqual(cfg.version, "2.5")
        self.assertTrue(cfg.is_2_5)
        self.assertEqual(cfg.spk_cond_mode, "campplus")
        self.assertEqual(cfg.tokenizer, "tiktoken")
        self.assertFalse(cfg.use_gpt_latent)

    def test_2_0_defaults_remain_compatible(self):
        from mlx_audio.tts.models.indextts2.config import ModelConfig

        cfg = ModelConfig.from_dict({"model_type": "indextts2"})
        self.assertEqual(cfg.version, "2.0")
        self.assertFalse(cfg.is_2_5)
        self.assertEqual(cfg.spk_cond_mode, "conformer")
        self.assertEqual(cfg.tokenizer, "sentencepiece")
        self.assertTrue(cfg.use_gpt_latent)


if __name__ == "__main__":
    unittest.main()
