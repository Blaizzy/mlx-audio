import importlib
import unittest
from unittest.mock import patch

from mlx_audio.tts.models.moss_tts.config import ModelConfig
from mlx_audio.tts.models.moss_tts.processor import MossTTSProcessor
from mlx_audio.tts.models.moss_tts.pronunciation import (
    PronunciationHelperUnavailableError,
    convert_text_to_ipa,
    convert_text_to_tone_numbered_pinyin,
    validate_pronunciation_input_contract,
)


def _tiny_delay_config_dict():
    return {
        "model_type": "moss_tts_delay",
        "language_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "vocab_size": 64,
            "tie_word_embeddings": True,
            "attention_bias": False,
        },
        "n_vq": 2,
        "audio_vocab_size": 16,
        "audio_user_slot_token_id": 40,
        "audio_assistant_gen_slot_token_id": 41,
        "audio_assistant_delay_slot_token_id": 42,
        "audio_start_token_id": 43,
        "audio_end_token_id": 44,
        "pad_token_id": 0,
        "im_start_token_id": 1,
        "im_end_token_id": 2,
        "audio_pad_code": 16,
        "sampling_rate": 24000,
        "additional_mlp_ffn_hidden_size": 24,
    }


class _DummyTokenizer:
    def __init__(self, special_id_to_token):
        self.special_id_to_token = dict(special_id_to_token)
        self.special_token_to_id = {v: k for k, v in special_id_to_token.items()}

    def convert_ids_to_tokens(self, token_id):
        return self.special_id_to_token.get(int(token_id), chr(97 + int(token_id) % 26))

    def encode(self, text):
        output = []
        for char in text:
            if char in self.special_token_to_id:
                output.append(self.special_token_to_id[char])
            else:
                output.append((ord(char) % 20) + 3)
        return output

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        joined = "".join(m["content"] for m in messages)
        if add_generation_prompt:
            joined = joined + "<assistant>"
        return joined

    def decode(self, token_ids):
        chars = [self.convert_ids_to_tokens(token_id) for token_id in token_ids]
        return "".join(chars)


def _build_dummy_processor(config: ModelConfig) -> MossTTSProcessor:
    token_map = {
        config.audio_user_slot_token_id: "§",
        config.audio_assistant_gen_slot_token_id: "¤",
        config.audio_assistant_delay_slot_token_id: "¦",
        config.audio_start_token_id: "†",
        config.audio_end_token_id: "‡",
    }
    tokenizer = _DummyTokenizer(token_map)
    return MossTTSProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=None,
        model_config=config,
    )


class TestMossPronunciationValidation(unittest.TestCase):
    def test_accepts_text_input_type_without_extra_constraints(self):
        self.assertEqual(
            validate_pronunciation_input_contract("hello world", "text"),
            "text",
        )

    def test_accepts_valid_tone_numbered_pinyin(self):
        self.assertEqual(
            validate_pronunciation_input_contract(
                "ni3 hao3 peng2 you3",
                "pinyin",
            ),
            "pinyin",
        )

    def test_rejects_pinyin_without_tone_numbers(self):
        with self.assertRaisesRegex(ValueError, "tone-numbered"):
            validate_pronunciation_input_contract("ni hao peng you", "pinyin")

    def test_rejects_pinyin_with_han_characters(self):
        with self.assertRaisesRegex(ValueError, "Han characters"):
            validate_pronunciation_input_contract("你好 朋友", "pinyin")

    def test_accepts_ipa_wrapped_spans_and_mixed_text(self):
        self.assertEqual(
            validate_pronunciation_input_contract(
                "Say /həˈloʊ/ to everyone.",
                "ipa",
            ),
            "ipa",
        )

    def test_rejects_unbalanced_ipa_slashes(self):
        with self.assertRaisesRegex(ValueError, "wrapped|unbalanced"):
            validate_pronunciation_input_contract("Say /həˈloʊ to everyone.", "ipa")

    def test_rejects_empty_ipa_span(self):
        with self.assertRaisesRegex(ValueError, "empty IPA span"):
            validate_pronunciation_input_contract("Say // now.", "ipa")


class TestMossProcessorPronunciationContract(unittest.TestCase):
    def test_processor_user_message_omits_input_type_prompt_field(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)

        msg = processor.build_user_message(
            text="ni3 hao3",
            input_type="pinyin",
        )

        self.assertNotIn("Input Type:", msg["content"])
        self.assertIn("- Text:\nni3 hao3", msg["content"])

    def test_processor_fails_fast_for_invalid_pinyin_payload(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)

        with self.assertRaisesRegex(ValueError, "tone-numbered"):
            processor.build_user_message(text="hello world", input_type="pinyin")

    def test_processor_fails_fast_for_invalid_ipa_payload(self):
        config = ModelConfig.from_dict(_tiny_delay_config_dict())
        processor = _build_dummy_processor(config)

        with self.assertRaisesRegex(ValueError, "/\\.\\.\\./"):
            processor.build_user_message(text="həˈloʊ", input_type="ipa")


class TestMossPronunciationHelpers(unittest.TestCase):
    def test_pinyin_helper_raises_clear_error_when_dependency_missing(self):
        def _fake_import(module_name: str):
            if module_name == "pypinyin":
                raise ImportError("pypinyin missing")
            return importlib.import_module(module_name)

        with patch(
            "mlx_audio.tts.models.moss_tts.pronunciation.import_module",
            side_effect=_fake_import,
        ):
            with self.assertRaises(PronunciationHelperUnavailableError):
                convert_text_to_tone_numbered_pinyin("你好")

    def test_ipa_helper_raises_clear_error_when_dependency_missing(self):
        def _fake_import(module_name: str):
            if module_name == "phonemizer":
                raise ImportError("phonemizer missing")
            return importlib.import_module(module_name)

        with patch(
            "mlx_audio.tts.models.moss_tts.pronunciation.import_module",
            side_effect=_fake_import,
        ):
            with self.assertRaises(PronunciationHelperUnavailableError):
                convert_text_to_ipa("Hello world")

    def test_deep_phonemizer_path_raises_clear_error_when_dependency_missing(self):
        def _fake_import(module_name: str):
            if module_name == "dp.phonemizer":
                raise ImportError("dp missing")
            return importlib.import_module(module_name)

        with patch(
            "mlx_audio.tts.models.moss_tts.pronunciation.import_module",
            side_effect=_fake_import,
        ):
            with self.assertRaises(PronunciationHelperUnavailableError):
                convert_text_to_ipa(
                    "Hello world",
                    deep_phonemizer_checkpoint="missing.ckpt",
                )


if __name__ == "__main__":
    unittest.main()
