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


class TestVoxtralStreamingContext(unittest.TestCase):
    def test_streamed_chunks_match_offline_decode(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        from mlx_audio.tts.models.voxtral_tts.audio_tokenizer import (
            AudioTokenizerArgs,
            VoxtralTTSAudioTokenizer,
        )
        from mlx_audio.tts.models.voxtral_tts.voxtral_tts import Model

        mx.random.seed(0)
        args = AudioTokenizerArgs(
            semantic_codebook_size=16,
            semantic_dim=8,
            dim=16,
            hidden_dim=32,
            head_dim=4,
            n_heads=4,
            n_kv_heads=4,
        )
        tokenizer = VoxtralTTSAudioTokenizer(args)
        flat = tree_flatten(tokenizer.parameters())
        keep_ones = ("cluster_usage", "original0")
        tokenizer.update(
            tree_unflatten(
                [
                    (
                        k,
                        (
                            mx.ones(v.shape)
                            if k.endswith(keep_ones)
                            else mx.random.normal(v.shape) * 0.3
                        ),
                    )
                    for k, v in flat
                ]
            )
        )
        for block in tokenizer.decoder_blocks[1::2]:
            for layer in block.layers:
                layer.attention_scale = mx.full((args.dim,), 1.0)
                layer.ffn_scale = mx.full((args.dim,), 1.0)
        mx.eval(tokenizer.parameters())

        n_frames = 80
        frame_codes = mx.concatenate(
            [
                mx.random.randint(2, 2 + 16, (n_frames, 1)),
                mx.random.randint(2, 2 + 21, (n_frames, 36)),
            ],
            axis=1,
        ).astype(mx.int32)
        mx.eval(frame_codes)

        class _Backbone:
            def __call__(self, input_ids, cache=None, input_embeddings=None):
                return mx.zeros((1, input_ids.shape[1], 4))

        class _Head:
            def __init__(self):
                self.i = 0

            def decode_one_frame(self, hidden):
                i, self.i = self.i, self.i + 1
                if i >= n_frames:
                    return mx.array([[1] + [2] * 36])
                return frame_codes[i][None, :]

        def run(**kwargs):
            model = Model.__new__(Model)
            model.config = SimpleNamespace(audio_token_id=24, sample_rate=24000)
            model.tokenizer = _DummyTokenizer()
            model._encode_text = lambda text, voice: [1, 25, 24, 36, 7, 35, 25]
            model._build_input_embeddings = lambda ids, voice: mx.zeros(
                (1, ids.shape[1], 4)
            )
            model.language_model = SimpleNamespace(
                model=SimpleNamespace(model=_Backbone())
            )
            model.acoustic_transformer = _Head()
            model._codes_to_global_indices = lambda codes: codes
            model.audio_codebook_embeddings = {
                "embeddings": lambda codes: mx.zeros((1, codes.shape[1], 4))
            }
            model.audio_tokenizer = tokenizer
            with patch(
                "mlx_audio.lm.models.cache.make_prompt_cache", return_value=None
            ):
                return list(model.generate("Done.", voice="casual_male", **kwargs))

        offline = run()[-1].audio
        chunks = run(stream=True, streaming_interval=2.0)
        self.assertEqual(len(chunks), 4)
        streamed = mx.concatenate([c.audio for c in chunks])

        self.assertEqual(streamed.shape, offline.shape)
        self.assertLess(mx.abs(streamed - offline).max().item(), 2e-5)
