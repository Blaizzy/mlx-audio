import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_audio.tts.models.voxtral_tts.voxtral_tts import Model


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [101, 102]


class TestVoxtralTTSPrompt(unittest.TestCase):
    def _make_model(self):
        model = Model.__new__(Model)
        model.tokenizer = FakeTokenizer()
        model.config = SimpleNamespace(
            bos_token_id=1,
            begin_audio_token_id=25,
            audio_token_id=24,
        )
        model._voice_embeddings = {}
        model._voice_embedding_files = {}
        model._voice_num_audio_tokens = {}
        model._text_to_audio_token_id = 36
        model._audio_to_text_token_id = 35
        return model

    def test_encode_text_fallback_matches_mistral_common_layout(self):
        model = self._make_model()
        model._voice_embeddings = {"casual_male": mx.zeros((147, 3072))}
        model._voice_num_audio_tokens = {"casual_male": 147}

        tokens = Model._encode_text(model, "Hello world.", "casual_male")

        self.assertEqual(tokens[:3], [1, 25, 24])
        self.assertEqual(tokens[1 + 1 + 147], 36)
        self.assertEqual(tokens[1 + 1 + 147 + 1 : 1 + 1 + 147 + 3], [101, 102])
        self.assertEqual(tokens[-2:], [35, 25])

    def test_encode_text_falls_back_to_voice_embedding_length(self):
        model = self._make_model()
        model._voice_embeddings = {"casual_male": mx.zeros((3, 3072))}

        tokens = Model._encode_text(model, "Hello world.", "casual_male")

        self.assertEqual(tokens, [1, 25, 24, 24, 24, 36, 101, 102, 35, 25])

    def test_encode_text_lazy_loads_requested_voice_embedding(self):
        model = self._make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            voice_file = Path(tmpdir) / "casual_male.safetensors"
            voice_file.touch()
            model._voice_embedding_files = {"casual_male": voice_file}

            with patch(
                "mlx_audio.tts.models.voxtral_tts.voxtral_tts.mx.load",
                return_value={"embedding": mx.zeros((3, 3072))},
            ) as mock_load:
                tokens = Model._encode_text(model, "Hello world.", "casual_male")

        self.assertEqual(tokens, [1, 25, 24, 24, 24, 36, 101, 102, 35, 25])
        mock_load.assert_called_once_with(str(voice_file))
        self.assertIn("casual_male", model._voice_embeddings)

    def test_get_voice_embedding_loads_once_and_caches(self):
        model = self._make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            voice_file = Path(tmpdir) / "casual_male.safetensors"
            voice_file.touch()
            model._voice_embedding_files = {"casual_male": voice_file}

            with patch(
                "mlx_audio.tts.models.voxtral_tts.voxtral_tts.mx.load",
                return_value={"embedding": mx.zeros((3, 3072))},
            ) as mock_load:
                first = Model._get_voice_embedding(model, "casual_male")
                second = Model._get_voice_embedding(model, "casual_male")

        self.assertEqual(first.shape, (3, 3072))
        self.assertIs(first, second)
        mock_load.assert_called_once_with(str(voice_file))

    def test_post_load_hook_registers_voice_embeddings_without_loading(self):
        model = self._make_model()
        model._voice_embeddings = {}
        model._voice_embedding_files = {}
        model._voice_num_audio_tokens = {}
        model._text_to_audio_token_id = None
        model._audio_to_text_token_id = None
        model.tokenizer = None

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            voice_dir = model_path / "voice_embedding"
            voice_dir.mkdir()
            (voice_dir / "casual_male.safetensors").touch()
            (voice_dir / "fr_female.safetensors").touch()

            with (
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=FakeTokenizer(),
                ),
                patch(
                    "mlx_audio.tts.models.voxtral_tts.voxtral_tts.mx.load",
                    side_effect=AssertionError("voice embeddings should load lazily"),
                ),
            ):
                Model.post_load_hook(model, model_path)

        self.assertEqual(
            set(model._voice_embedding_files), {"casual_male", "fr_female"}
        )
        self.assertEqual(model._voice_embeddings, {})


if __name__ == "__main__":
    unittest.main()
