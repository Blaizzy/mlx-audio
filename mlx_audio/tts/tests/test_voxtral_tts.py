import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


class _RecordingBackbone:
    """Returns hidden states that identify which LM call produced them."""

    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, cache=None, input_embeddings=None):
        import mlx.core as mx

        self.calls += 1
        return mx.full((1, input_ids.shape[1], 4), float(self.calls))


class TestVoxtralGenerationStart(unittest.TestCase):
    def test_first_frame_is_decoded_from_the_prompt_state(self):
        import mlx.core as mx

        from mlx_audio.tts.models.voxtral_tts.voxtral_tts import Model

        model = Model.__new__(Model)
        model.config = SimpleNamespace(audio_token_id=24, sample_rate=24000)
        model.tokenizer = _DummyTokenizer()
        model._encode_text = lambda text, voice: [1, 25, 24, 36, 7, 35, 25]
        model._build_input_embeddings = lambda ids, voice: mx.zeros(
            (1, ids.shape[1], 4)
        )
        backbone = _RecordingBackbone()
        model.language_model = SimpleNamespace(
            model=SimpleNamespace(model=backbone),
            embed_tokens=lambda ids: mx.zeros((1, ids.shape[1], 4)),
        )
        decoded_from = []

        def decode_one_frame(hidden):
            decoded_from.append(hidden[0, 0].item())
            semantic = 1 if len(decoded_from) > 2 else 5
            return mx.array([[semantic] + [2] * 36])

        model.acoustic_transformer = SimpleNamespace(decode_one_frame=decode_one_frame)
        model._codes_to_global_indices = lambda codes: codes
        model.audio_codebook_embeddings = {
            "embeddings": lambda codes: mx.zeros((1, codes.shape[1], 4))
        }
        model.audio_tokenizer = SimpleNamespace(
            decode=lambda codes: mx.zeros((1, codes.shape[1] * 1920))
        )

        with patch("mlx_audio.lm.models.cache.make_prompt_cache", return_value=None):
            results = list(model.generate("Done.", voice="casual_male"))

        # Frame 1 comes from the prefill (call 1), frame 2 from feeding frame 1
        # back (call 2); no extra [AUDIO] step sits in between.
        self.assertEqual(decoded_from, [1.0, 2.0, 3.0])
        self.assertEqual(backbone.calls, 3)
        self.assertEqual(results[-1].samples, 2 * 1920)
