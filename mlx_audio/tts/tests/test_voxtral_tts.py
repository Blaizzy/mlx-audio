import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


class _DummyConfig:
    bos_token_id = 1
    begin_audio_token_id = 25
    audio_token_id = 24


class _DummyTokenizer:
    pass


class TestVoxtralDependencyContract(unittest.TestCase):
    def test_tts_extra_includes_mistral_common_audio(self):
        pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
        pyproject = tomllib.loads(pyproject_path.read_text())

        tts_extra = pyproject["project"]["optional-dependencies"]["tts"]
        self.assertIn("mistral-common[audio]", tts_extra)

    def test_encode_text_requires_speech_tokenizer_support(self):
        from mlx_audio.tts.models.voxtral_tts.voxtral_tts import Model

        model = Model.__new__(Model)
        model.config = _DummyConfig()
        model.tokenizer = _DummyTokenizer()
        model._voice_embeddings = {}
        model._text_to_audio_token_id = 100
        model._audio_to_text_token_id = 101

        with self.assertRaisesRegex(RuntimeError, "mistral-common\\[audio\\]"):
            model._encode_text("hello world", "casual_male")
