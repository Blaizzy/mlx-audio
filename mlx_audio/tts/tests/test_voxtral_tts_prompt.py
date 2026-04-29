import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_audio.tts.models.voxtral_tts.voxtral_tts import Model


class FakeTokenizer:
    def __init__(self, tokens=None):
        self.tokens = tokens or [201, 202, 203]
        self.requests = []

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [101, 102]

    def encode_speech_request(self, request):
        self.requests.append(request)
        return SimpleNamespace(tokens=list(self.tokens))


def patch_fake_speech_request():
    request_module = ModuleType("mistral_common.protocol.speech.request")

    class FakeSpeechRequest:
        def __init__(self, input, voice):
            self.input = input
            self.voice = voice

    request_module.SpeechRequest = FakeSpeechRequest

    speech_module = ModuleType("mistral_common.protocol.speech")
    speech_module.request = request_module

    protocol_module = ModuleType("mistral_common.protocol")
    protocol_module.speech = speech_module

    mistral_common_module = ModuleType("mistral_common")
    mistral_common_module.protocol = protocol_module

    return patch.dict(
        sys.modules,
        {
            "mistral_common": mistral_common_module,
            "mistral_common.protocol": protocol_module,
            "mistral_common.protocol.speech": speech_module,
            "mistral_common.protocol.speech.request": request_module,
        },
    )


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

    def test_encode_text_uses_speech_request_tokens(self):
        model = self._make_model()
        model.tokenizer = FakeTokenizer(tokens=[7, 8, 9])

        with patch_fake_speech_request():
            tokens = Model._encode_text(model, "Hello world.", "casual_male")

        self.assertEqual(tokens, [7, 8, 9])

    def test_encode_text_passes_text_and_voice_to_speech_request(self):
        model = self._make_model()

        with patch_fake_speech_request():
            tokens = Model._encode_text(model, "Hello world.", "casual_male")

        self.assertEqual(tokens, [201, 202, 203])
        self.assertEqual(len(model.tokenizer.requests), 1)
        request = model.tokenizer.requests[0]
        self.assertEqual(request.input, "Hello world.")
        self.assertEqual(request.voice, "casual_male")

    def test_encode_text_does_not_lazy_load_voice_embedding_when_supported(self):
        model = self._make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            voice_file = Path(tmpdir) / "casual_male.safetensors"
            voice_file.touch()
            model._voice_embedding_files = {"casual_male": voice_file}

            with patch(
                "mlx_audio.tts.models.voxtral_tts.voxtral_tts.mx.load",
                side_effect=AssertionError(
                    "speech-request path should not load voices"
                ),
            ) as mock_load:
                with patch_fake_speech_request():
                    tokens = Model._encode_text(model, "Hello world.", "casual_male")

        self.assertEqual(tokens, [201, 202, 203])
        mock_load.assert_not_called()
        self.assertNotIn("casual_male", model._voice_embeddings)

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
