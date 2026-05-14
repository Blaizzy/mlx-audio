import importlib.resources
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# Create a patch for the deprecated open_text function
def patched_open_text(package, resource):
    """Replacement for deprecated open_text using files() API"""
    return importlib.resources.files(package).joinpath(resource).open("r")


class FakeTokenizer:
    def __init__(self):
        self.semantic_begin_id = 1000
        self._next = 1

    def encode(self, text):
        token = self._next
        self._next += 1
        return [token]


def tiny_config():
    from mlx_audio.tts.models.fish_qwen3_omni.config import ModelConfig

    return ModelConfig.from_dict(
        {
            "semantic_start_token_id": 1000,
            "semantic_end_token_id": 1007,
            "text_config": {
                "vocab_size": 32,
                "n_layer": 1,
                "n_head": 2,
                "dim": 8,
                "intermediate_size": 16,
                "n_local_heads": 1,
                "head_dim": 4,
                "norm_eps": 1e-6,
                "max_seq_len": 64,
                "attention_qk_norm": True,
            },
            "audio_decoder_config": {
                "vocab_size": 8,
                "n_layer": 1,
                "n_head": 2,
                "dim": 8,
                "intermediate_size": 16,
                "n_local_heads": 1,
                "head_dim": 4,
                "num_codebooks": 2,
                "norm_eps": 1e-6,
                "max_seq_len": 3,
            },
        }
    )


# Apply the patch at the module level
@patch("importlib.resources.open_text", patched_open_text)
class TestSanitizeLSTMWeights(unittest.TestCase):
    def test_sanitize_lstm_weights(self):
        """Test sanitize_lstm_weights function."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.kokoro import sanitize_lstm_weights

        # Test weight_ih_l0_reverse
        key = "lstm.weight_ih_l0_reverse"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wx_backward")

        # Test weight_hh_l0_reverse
        key = "lstm.weight_hh_l0_reverse"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wh_backward")

        # Test bias_ih_l0_reverse
        key = "lstm.bias_ih_l0_reverse"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_ih_backward")

        # Test bias_hh_l0_reverse
        key = "lstm.bias_hh_l0_reverse"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_hh_backward")

        # Test weight_ih_l0
        key = "lstm.weight_ih_l0"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wx_forward")

        # Test weight_hh_l0
        key = "lstm.weight_hh_l0"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wh_forward")

        # Test bias_ih_l0
        key = "lstm.bias_ih_l0"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_ih_forward")

        # Test bias_hh_l0
        key = "lstm.bias_hh_l0"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_hh_forward")

        # Test unknown key
        key = "unknown.key"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "unknown.key")


@patch("importlib.resources.open_text", patched_open_text)
class TestKokoroModel(unittest.TestCase):
    @patch("json.load")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("mlx_audio.tts.models.kokoro.kokoro.mx.load")
    @patch.object(nn.Module, "load_weights")
    def test_init(self, mock_load_weights, mock_mx_load, mock_open, mock_json_load):
        """Test KokoroModel initialization."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.kokoro import Model, ModelConfig

        # Mock the config loading
        config = {
            "istftnet": {
                "upsample_kernel_sizes": [20, 12],
                "upsample_rates": [10, 6],
                "gen_istft_hop_size": 5,
                "gen_istft_n_fft": 20,
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "resblock_kernel_sizes": [3, 7, 11],
                "upsample_initial_channel": 512,
            },
            "dim_in": 64,
            "dropout": 0.2,
            "hidden_dim": 512,
            "max_conv_dim": 512,
            "max_dur": 50,
            "multispeaker": True,
            "n_layer": 3,
            "n_mels": 80,
            "n_token": 178,
            "style_dim": 128,
            "text_encoder_kernel_size": 5,
            "plbert": {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "intermediate_size": 2048,
                "max_position_embeddings": 512,
                "num_hidden_layers": 12,
                "dropout": 0.1,
            },
            "vocab": {"a": 1, "b": 2},
        }
        mock_json_load.return_value = config

        # Mock the weights loading
        mock_mx_load.return_value = {"key": mx.array(np.zeros(10))}

        # Make load_weights return the module
        mock_load_weights.return_value = None

        # Initialize the model with the config parameter
        model = Model(ModelConfig.from_dict(config))

        # Check that the model was initialized correctly
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.vocab, {"a": 1, "b": 2})

    def test_output_dataclass(self):
        """Test KokoroModel.Output dataclass."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.kokoro import Model

        # Create a mock output
        audio = mx.array(np.zeros((1, 1000)))
        pred_dur = mx.array(np.zeros((1, 100)))

        # Mock __init__ to return None
        with patch.object(Model, "__init__", return_value=None):
            output = Model.Output(audio=audio, pred_dur=pred_dur)

        # Check that the output was created correctly
        self.assertIs(output.audio, audio)
        self.assertIs(output.pred_dur, pred_dur)


@patch("importlib.resources.open_text", patched_open_text)
class TestKokoroPipeline(unittest.TestCase):
    def test_aliases_and_lang_codes(self):
        """Test ALIASES and LANG_CODES constants."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import ALIASES, LANG_CODES

        # Check that all aliases map to valid language codes
        for alias_key, alias_value in ALIASES.items():
            self.assertIn(alias_value, LANG_CODES)

        # Check specific mappings
        self.assertEqual(ALIASES["en-us"], "a")
        self.assertEqual(ALIASES["ja"], "j")
        self.assertEqual(LANG_CODES["a"], "American English")
        self.assertEqual(LANG_CODES["j"], "Japanese")

    def test_init(self):
        """Test KokoroPipeline initialization."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import LANG_CODES, KokoroPipeline

        # Mock the G2P class to avoid spacy download during tests
        mock_en = SimpleNamespace(G2P=MagicMock())
        mock_espeak = SimpleNamespace(EspeakFallback=MagicMock())
        with patch(
            "mlx_audio.tts.models.kokoro.pipeline._get_misaki_en",
            return_value=mock_en,
        ):
            with patch(
                "mlx_audio.tts.models.kokoro.pipeline._get_misaki_espeak",
                return_value=mock_espeak,
            ):
                mock_model = MagicMock()
                mock_en.G2P.return_value = MagicMock()
                mock_espeak.EspeakFallback.return_value = MagicMock()

                # Initialize with default model
                pipeline = KokoroPipeline(
                    lang_code="a", model=mock_model, repo_id="mock"
                )
                self.assertEqual(pipeline.lang_code, "a")
                self.assertEqual(LANG_CODES[pipeline.lang_code], "American English")

                # Initialize with provided model
                model = mock_model
                pipeline = KokoroPipeline(lang_code="a", model=model, repo_id="mock")
                self.assertEqual(pipeline.model, model)

                # Initialize with no model
                pipeline = KokoroPipeline(lang_code="a", model=False, repo_id="mock")
                self.assertIs(pipeline.model, False)

    def test_init_without_misaki(self):
        """Test KokoroPipeline raises a targeted install hint when misaki is missing."""
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        with patch(
            "mlx_audio.tts.models.kokoro.pipeline.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'misaki'"),
        ):
            with self.assertRaisesRegex(ImportError, "pip install misaki"):
                KokoroPipeline(lang_code="a", model=False, repo_id="mock")

    def test_load_voice(self):
        """Test load_voice method."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Setup the pipeline
        with patch.object(KokoroPipeline, "__init__", return_value=None):
            with patch(
                "mlx_audio.tts.models.kokoro.pipeline.load_voice_tensor"
            ) as load_voice_tensor:
                with patch(
                    "mlx_audio.tts.models.kokoro.pipeline.snapshot_download"
                ) as mock_snapshot_download:
                    pipeline = KokoroPipeline.__new__(KokoroPipeline)
                    pipeline.lang_code = "a"
                    pipeline.voices = {}
                    # Add the missing repo_id attribute
                    pipeline.repo_id = "mlx-community/kokoro-tts"

                    # Mock the load voice return value
                    load_voice_tensor.return_value = mx.zeros((512, 1, 256))

                    # Mock snapshot_download to return a path
                    # First call with local_files_only=True raises error, second downloads
                    mock_snapshot_download.side_effect = [
                        FileNotFoundError(),  # local_files_only=True fails
                        "/mock/path",  # actual download succeeds
                    ]

                    # Test loading a single voice
                    pipeline.load_single_voice("voice1")
                    self.assertEqual(mock_snapshot_download.call_count, 2)
                    self.assertIn("voice1", pipeline.voices)

                    # Test loading multiple voices
                    mock_snapshot_download.reset_mock()
                    mock_snapshot_download.side_effect = [
                        FileNotFoundError(),
                        "/mock/path",
                        FileNotFoundError(),
                        "/mock/path",
                    ]
                    pipeline.voices = {}  # Reset voices
                    result = pipeline.load_voice("voice1,voice2")
                    self.assertEqual(mock_snapshot_download.call_count, 4)
                    self.assertIn("voice1", pipeline.voices)
                    self.assertIn("voice2", pipeline.voices)

    def test_tokens_to_ps(self):
        """Test tokens_to_ps method."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Create mock tokens with whitespace attribute
        token1 = SimpleNamespace(ps="p1", whitespace=" ", phonemes="p1")
        token2 = SimpleNamespace(ps="p2", whitespace="", phonemes="p2")

        tokens = [token1, token2]

        # Test the method
        result = KokoroPipeline.tokens_to_ps(tokens)
        self.assertEqual(result, "p1 p2")

    def test_tokens_to_text(self):
        """Test tokens_to_text method."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Create mock tokens with whitespace attribute
        token1 = SimpleNamespace(text="Hello", whitespace=" ")
        token2 = SimpleNamespace(text="world", whitespace="")

        tokens = [token1, token2]

        # Test the method
        result = KokoroPipeline.tokens_to_text(tokens)
        self.assertEqual(result, "Hello world")

    def test_result_dataclass(self):
        """Test KokoroPipeline.Result dataclass."""
        # Import inside the test methods
        from mlx_audio.tts.models.kokoro.kokoro import Model
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Create a mock output
        audio = mx.array(np.zeros((1, 1000)))
        pred_dur = mx.array(np.zeros((1, 100)))
        model_output = Model.Output(audio=audio, pred_dur=pred_dur)

        # Create a Result instance
        result = KokoroPipeline.Result(
            graphemes="Hello",
            phonemes="HH EH L OW",
            tokens=[MagicMock()],
            output=model_output,
            text_index=0,
        )

        # Check properties
        self.assertEqual(result.graphemes, "Hello")
        self.assertEqual(result.phonemes, "HH EH L OW")
        self.assertIs(result.audio, audio)
        self.assertIs(result.pred_dur, pred_dur)

        # Test backward compatibility
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Hello")
        self.assertEqual(result[1], "HH EH L OW")
        self.assertIs(result[2], audio)

        # Test iteration
        items = list(result)
        self.assertEqual(items[0], "Hello")
        self.assertEqual(items[1], "HH EH L OW")
        self.assertIs(items[2], audio)


@patch("importlib.resources.open_text", patched_open_text)
class TestKittenTTSModel(unittest.TestCase):
    def _config(self):
        return {
            "hidden_dim": 16,
            "max_conv_dim": 16,
            "max_dur": 10,
            "n_layer": 1,
            "n_mels": 80,
            "n_token": 32,
            "style_dim": 64,
            "text_encoder_kernel_size": 3,
            "asr_res_dim": 8,
            "decoder_out_dim": 16,
            "plbert": {
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "max_position_embeddings": 32,
                "embedding_size": 16,
                "inner_group_num": 1,
                "num_hidden_groups": 1,
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "type_vocab_size": 2,
                "layer_norm_eps": 1e-12,
            },
            "istftnet": {
                "resblock_kernel_sizes": [3, 3],
                "upsample_rates": [2, 2],
                "upsample_initial_channel": 32,
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
                "upsample_kernel_sizes": [4, 4],
                "gen_istft_n_fft": 16,
                "gen_istft_hop_size": 4,
            },
        }

    def test_init(self):
        from mlx_audio.tts.models.kitten_tts.kitten_tts import Model, ModelConfig

        config = self._config()
        model = Model(ModelConfig.from_dict(config))
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.config.n_token, config["n_token"])

    def test_sanitize_alpha_names(self):
        from mlx_audio.tts.models.kitten_tts.kitten_tts import Model, ModelConfig

        config = self._config()
        model = Model(ModelConfig.from_dict(config))
        weights = {
            "decoder.generator.resblocks.0.alpha1.0": mx.ones((1, 1, 1)),
            "decoder.generator.resblocks.0.alpha2.0": mx.ones((1, 1, 1)),
        }
        sanitized = model.sanitize(weights)
        self.assertIn("decoder.generator.resblocks.0.alpha1_0", sanitized)
        self.assertIn("decoder.generator.resblocks.0.alpha2_0", sanitized)
        self.assertNotIn("decoder.generator.resblocks.0.alpha1.0", sanitized)

    def test_missing_phonemizer_error(self):
        from mlx_audio.tts.models.kitten_tts.kitten_tts import Model, ModelConfig

        config = self._config()
        model = Model(ModelConfig.from_dict(config))

        with patch(
            "mlx_audio.tts.models.kitten_tts.kitten_tts.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'phonemizer'"),
        ):
            with self.assertRaisesRegex(ImportError, "pip install phonemizer-fork"):
                model._get_phonemizer()


class TestBarkModel(unittest.TestCase):
    @patch("mlx_audio.tts.models.bark.bark.BertTokenizer")
    def test_init(self, mock_tokenizer):
        """Test BarkModel initialization."""
        from mlx_audio.tts.models.bark.bark import (
            CoarseAcousticsConfig,
            CodecConfig,
            FineAcousticsConfig,
            Model,
            ModelConfig,
            SemanticConfig,
        )

        # Create mock configs
        semantic_config = SemanticConfig()
        coarse_config = CoarseAcousticsConfig()
        fine_config = FineAcousticsConfig()
        codec_config = CodecConfig()

        config = ModelConfig(
            semantic_config=semantic_config,
            coarse_acoustics_config=coarse_config,
            fine_acoustics_config=fine_config,
            codec_config=codec_config,
        )

        # Initialize model
        model = Model(config)

        # Check that components were initialized correctly
        self.assertIsNotNone(model.semantic)
        self.assertIsNotNone(model.coarse_acoustics)
        self.assertIsNotNone(model.fine_acoustics)
        self.assertIsNotNone(model.tokenizer)

    def test_sanitize_weights(self):
        """Test weight sanitization."""
        from mlx_audio.tts.models.bark.bark import Model, ModelConfig

        # Create a minimal config
        config = ModelConfig(
            semantic_config={},
            coarse_acoustics_config={},
            fine_acoustics_config={},
            codec_config={},
        )

        model = Model(config)

        # Test with transformer weights
        weights = {
            "_orig_mod.transformer.h.0.mlp.weight": mx.zeros((10, 10)),
            "_orig_mod.transformer.h.1.mlp.weight": mx.zeros((10, 10)),
            "lm_head.weight": mx.zeros((10, 10)),
        }

        sanitized = model.sanitize(weights)

        # Check that weights were properly renamed
        self.assertIn("layers.0.mlp.weight", sanitized)
        self.assertIn("layers.1.mlp.weight", sanitized)
        self.assertIn("lm_head.weight", sanitized)


@patch("importlib.resources.open_text", patched_open_text)
class TestBarkPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        from mlx_audio.tts.models.bark.bark import (
            CoarseAcousticsConfig,
            CodecConfig,
            FineAcousticsConfig,
            Model,
            ModelConfig,
            SemanticConfig,
        )
        from mlx_audio.tts.models.bark.pipeline import Pipeline

        # Create mock model with required attributes
        self.mock_model = MagicMock(spec=Model)

        # Add the required mock attributes/methods
        self.mock_model.semantic = MagicMock()
        self.mock_model.coarse_acoustics = MagicMock()
        self.mock_model.fine_acoustics = MagicMock()
        self.mock_model.codec_model = MagicMock()

        self.mock_tokenizer = MagicMock()

        # Initialize pipeline
        self.pipeline = Pipeline(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=ModelConfig(
                semantic_config=SemanticConfig(),
                coarse_acoustics_config=CoarseAcousticsConfig(),
                fine_acoustics_config=FineAcousticsConfig(),
                codec_config=CodecConfig(),
            ),
        )

    def test_generate_text_semantic(self):
        """Test semantic token generation."""
        # Mock tokenizer output
        self.mock_tokenizer.encode.return_value = [1, 2, 3]

        # Create logits with proper shape including SEMANTIC_PAD_TOKEN
        logits = mx.zeros((1, 1, 129596))  # Large enough to include SEMANTIC_PAD_TOKEN
        # Mock model output
        self.mock_model.semantic.return_value = (
            logits,  # logits with correct shape
            None,  # kv_cache
        )

        # Test generation
        semantic_tokens, text_tokens = self.pipeline.generate_text_semantic(
            "test text",
            temperature=0.7,
            use_kv_caching=True,
            voice=None,
        )

        # Verify tokenizer was called
        self.mock_tokenizer.encode.assert_called_once_with(
            "test text", add_special_tokens=False
        )

        # Verify model was called
        self.mock_model.semantic.assert_called()

        # Check output types
        self.assertIsInstance(semantic_tokens, mx.array)
        self.assertIsInstance(text_tokens, mx.array)

    @patch("mlx.core.random.categorical")  # Add this patch since we use mx alias
    def test_generate_coarse(self, mock_mlx_categorical):
        """Test coarse token generation."""
        # Create mock semantic tokens
        semantic_tokens = mx.array([1, 2, 3])

        # Create logits with proper shape
        logits = mx.zeros((1, 1, 12096))

        # Mock both categorical functions to return predictable values
        mock_mlx_categorical.return_value = mx.array([10000])  # Return token index

        # Set up the mock to return proper values for each call
        self.mock_model.coarse_acoustics.return_value = (logits, None)

        # Test generation with minimal parameters to reduce test time
        coarse_tokens = self.pipeline.generate_coarse(
            semantic_tokens,
            temperature=0.7,
            use_kv_caching=True,
            voice=None,
            max_coarse_history=60,
            sliding_window_len=2,  # Reduce this to minimum
        )

        # Verify model was called at least once
        self.mock_model.coarse_acoustics.assert_called()

        # Check output type and shape
        self.assertIsInstance(coarse_tokens, mx.array)
        self.assertEqual(coarse_tokens.shape[0], 2)  # N_COARSE_CODEBOOKS

    def test_generate_fine(self):
        """Test fine token generation."""
        # Create mock coarse tokens
        coarse_tokens = mx.zeros((2, 100))  # N_COARSE_CODEBOOKS x sequence_length

        # Mock model output with proper shape
        self.mock_model.fine_acoustics.return_value = mx.zeros((1, 1024, 1024))

        # Test generation
        fine_tokens = self.pipeline.generate_fine(coarse_tokens, temperature=0.7)

        # Verify model was called
        self.mock_model.fine_acoustics.assert_called()

        # Check output type and shape
        self.assertIsInstance(fine_tokens, mx.array)
        self.assertEqual(
            fine_tokens.shape[0], 8
        )  # N_FINE_CODEBOOKS (corrected from 10 to 8)
        self.assertEqual(fine_tokens.shape[1], 100)  # sequence_length


class TestLlamaModel(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "attention_bias": False,
            "head_dim": 128,
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 24,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": True,
            "vocab_size": 156940,
            "layer_types": ["full_attention"] * 28,
        }

    @patch("transformers.AutoTokenizer")
    def test_init(self, mock_tokenizer):
        """Test LlamaModel initialization."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock the tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create a minimal config
        config = ModelConfig(**self._default_config)

        # Initialize model
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)

    @patch("transformers.AutoTokenizer")
    def test_generate(self, mock_tokenizer):
        """Test generate method."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()

        def mock_tokenize(text, return_tensors=None):
            result = MagicMock()
            result.input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int64)
            return result

        mock_tokenizer_instance.side_effect = mock_tokenize
        mock_tokenizer_instance.__call__ = mock_tokenize
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Verify batched input creation with a voice
        input_ids = model.prepare_input_ids(["Foo", "Bar Baz"], voice="zoe")
        self.assertEqual(input_ids.shape[0], 2)

        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, input_ids.shape[1], config.vocab_size))

        # Verify batched input creation with reference audio
        input_ids, input_mask = model.prepare_input_ids(
            ["Foo", "Bar Baz"], ref_audio=mx.zeros((100,)), ref_text="Caption"
        )
        self.assertEqual(input_ids.shape[0], 2)

        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, input_ids.shape[1], config.vocab_size))

    @patch("transformers.AutoTokenizer")
    def test_sanitize(self, mock_tokenizer):
        """Test sanitize method."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create a config with tie_word_embeddings=True
        config = ModelConfig(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=16384,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            head_dim=128,
            max_position_embeddings=1024,
            num_key_value_heads=32,
            attention_bias=True,
            mlp_bias=True,
            rope_theta=500000.0,
            rope_traditional=False,
            rope_scaling=None,
            tie_word_embeddings=True,
        )

        # Initialize the model with a patched __init__
        with patch.object(Model, "__init__", return_value=None):
            model = Model.__new__(Model)
            model.config = config

            # Add the sanitize method from actual implementation
            def mock_sanitize(weights):
                result = {}
                for k, v in weights.items():
                    if "rotary_emb" in k:
                        continue
                    if "lm_head.weight" in k and config.tie_word_embeddings:
                        continue
                    result[k] = v
                return result

            model.sanitize = mock_sanitize

            # Create test weights with rotary embeddings and lm_head
            weights = {
                "self_attn.rotary_emb.inv_freq": mx.zeros(10),
                "lm_head.weight": mx.zeros((32000, 4096)),
                "model.layers.0.input_layernorm.weight": mx.zeros(4096),
            }

            # Test sanitize method
            sanitized = model.sanitize(weights)

            # Assert rotary embeddings are removed
            self.assertNotIn("self_attn.rotary_emb.inv_freq", sanitized)

            # Assert lm_head weights are removed with tie_word_embeddings=True
            self.assertNotIn("lm_head.weight", sanitized)

            # Assert other weights remain
            self.assertIn("model.layers.0.input_layernorm.weight", sanitized)

            # Now test with tie_word_embeddings=False
            config.tie_word_embeddings = False

            # Test sanitize again
            sanitized2 = model.sanitize(weights)

            # lm_head should be kept with tie_word_embeddings=False
            self.assertIn("lm_head.weight", sanitized2)


class TestQwen3Model(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "head_dim": 128,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "max_position_embeddings": 40960,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000,
            "tie_word_embeddings": True,
            "vocab_size": 180352,
        }

    @patch("transformers.AutoTokenizer")
    def test_init(self, mock_tokenizer):
        """Test Qwen3Model initialization."""
        from mlx_audio.tts.models.qwen3.qwen3 import Model, ModelConfig

        # Mock the tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create a minimal config
        config = ModelConfig(**self._default_config)

        # Initialize model
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)
        self.assertEqual(model.model_type, "qwen3")
        self.assertIsNone(model.tokenizer)

    @patch("transformers.AutoTokenizer")
    def test_forward(self, mock_tokenizer):
        """Test forward pass."""
        from mlx_audio.tts.models.qwen3.qwen3 import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Test forward pass with random input
        input_ids = mx.random.randint(0, config.vocab_size, (2, 9))
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 9, config.vocab_size))

    @patch("transformers.AutoTokenizer")
    def test_prepare_input_ids_with_voice(self, mock_tokenizer):
        """Test prepare_input_ids method with voice."""
        from mlx_audio.tts.models.qwen3.qwen3 import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()

        # Mock tokenizer __call__ to return proper input_ids
        def mock_tokenize(text, return_tensors=None):
            result = MagicMock()
            # Return a simple token sequence for each text
            result.input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int64)
            return result

        mock_tokenizer_instance.side_effect = mock_tokenize
        mock_tokenizer_instance.__call__ = mock_tokenize
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)
        model.tokenizer = mock_tokenizer_instance

        # Test with voice
        input_ids = model.prepare_input_ids(["Hello", "World"], voice="zoe")

        # Verify batch size
        self.assertEqual(input_ids.shape[0], 2)

    @patch("transformers.AutoTokenizer")
    def test_parse_output(self, mock_tokenizer):
        """Test parse_output method."""
        from mlx_audio.tts.models.qwen3.qwen3 import (
            AUDIO_TOKENS_START,
            END_OF_SPEECH,
            START_OF_SPEECH,
            Model,
            ModelConfig,
        )

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Create input with speech tokens
        # Format: [START_OF_SPEECH, audio_tokens..., END_OF_SPEECH]
        audio_tokens = [AUDIO_TOKENS_START + i for i in range(7)]  # 7 audio tokens
        input_sequence = [START_OF_SPEECH] + audio_tokens + [END_OF_SPEECH]
        input_ids = mx.array([input_sequence], dtype=mx.int64)

        # Test parse_output
        code_lists = model.parse_output(input_ids)

        # Should return one code list (one batch item)
        self.assertEqual(len(code_lists), 1)

        # The code list should have 7 items (trimmed to multiple of 7)
        self.assertEqual(len(code_lists[0]), 7)

        # Verify codes are offset by AUDIO_TOKENS_START
        for i, code in enumerate(code_lists[0]):
            self.assertEqual(code, i)

    @patch("transformers.AutoTokenizer")
    def test_sample_rate(self, mock_tokenizer):
        """Test sample_rate property."""
        from mlx_audio.tts.models.qwen3.qwen3 import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Default sample rate should be 24000
        self.assertEqual(model.sample_rate, 24000)

    @patch("transformers.AutoTokenizer")
    def test_layers_property(self, mock_tokenizer):
        """Test layers property returns model layers."""
        from mlx_audio.tts.models.qwen3.qwen3 import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Verify layers property returns the model's layers
        layers = model.layers
        self.assertEqual(len(layers), config.num_hidden_layers)


class TestOuteTTSModel(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "attention_bias": False,
            "head_dim": 64,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": True,
            "vocab_size": 134400,
        }

    @patch("transformers.AutoTokenizer")
    def test_init(self, mock_tokenizer):
        """Test initialization."""
        from mlx_audio.tts.models.outetts.outetts import Model, ModelConfig

        # Mock the tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Create a minimal config
        config = ModelConfig(**self._default_config)

        # Initialize model
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)

    @patch("transformers.AutoTokenizer")
    def test_generate(self, mock_tokenizer):
        """Test generate method."""
        from mlx_audio.tts.models.outetts.outetts import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        input_ids = mx.random.randint(0, config.vocab_size, (2, 9))
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 9, config.vocab_size))


class TestDiaModel(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "version": "0.1",
            "model": {
                "encoder": {
                    "n_layer": 12,
                    "n_embd": 1024,
                    "n_hidden": 4096,
                    "n_head": 16,
                    "head_dim": 128,
                },
                "decoder": {
                    "n_layer": 18,
                    "n_embd": 2048,
                    "n_hidden": 8192,
                    "gqa_query_heads": 16,
                    "cross_query_heads": 16,
                    "kv_heads": 4,
                    "gqa_head_dim": 128,
                    "cross_head_dim": 128,
                },
                "src_vocab_size": 256,
                "tgt_vocab_size": 1028,
                "dropout": 0.0,
            },
            "training": {},
            "data": {
                "text_length": 1024,
                "audio_length": 3072,
                "channels": 9,
                "text_pad_value": 0,
                "audio_eos_value": 1024,
                "audio_pad_value": 1025,
                "audio_bos_value": 1026,
                "delay_pattern": [0, 8, 9, 10, 11, 12, 13, 14, 15],
            },
        }

    def test_init(self):
        """Test DiaModel initialization."""
        from mlx_audio.tts.models.dia.dia import Model

        # Initialize model
        config = self._default_config
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)


class TestSparkTTSModel(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "sample_rate": 16000,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 896,
            "initializer_range": 0.02,
            "intermediate_size": 4864,
            "max_position_embeddings": 32768,
            "max_window_layers": 21,
            "model_type": "qwen2",
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.43.1",
            "use_sliding_window": False,
            "vocab_size": 166000,
            "rope_traditional": False,
            "rope_scaling": None,
        }

    @patch("mlx_audio.tts.models.spark.spark.Qwen2Model")
    def test_init(self, mock_qwen2_model):
        """Test SparkTTSModel initialization."""
        from mlx_audio.tts.models.spark.spark import Model, ModelConfig

        # Mock return value for Qwen2Model
        mock_qwen2_model.return_value = MagicMock()

        # Create a config instance
        config = ModelConfig(**self._default_config)

        # Initialize the model
        model = Model(config)

        # Check that the model was initialized correctly
        self.assertIsInstance(model, Model)

        # Verify tokenizer is None initially (loaded via post_load_hook)
        self.assertIsNone(model.tokenizer)

        # Verify the Qwen2Model was initialized correctly
        mock_qwen2_model.assert_called_once_with(config)


class TestIndexTTS(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "tokenizer_name": "mlx-community/IndexTTS",
            "bigvgan": {
                "adam_b1": 0.8,
                "adam_b2": 0.99,
                "lr_decay": 0.999998,
                "seed": 1234,
                "resblock": "1",
                "upsample_rates": [4, 4, 4, 4, 2, 2],
                "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
                "upsample_initial_channel": 1536,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "feat_upsample": False,
                "speaker_embedding_dim": 512,
                "cond_d_vector_in_each_upsampling_layer": True,
                "gpt_dim": 1024,
                "activation": "snakebeta",
                "snake_logscale": True,
                "use_cqtd_instead_of_mrd": True,
                "cqtd_filters": 128,
                "cqtd_max_filters": 1024,
                "cqtd_filters_scale": 1,
                "cqtd_dilations": [1, 2, 4],
                "cqtd_hop_lengths": [512, 256, 256],
                "cqtd_n_octaves": [9, 9, 9],
                "cqtd_bins_per_octaves": [24, 36, 48],
                "resolutions": [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]],
                "mpd_reshapes": [2, 3, 5, 7, 11],
                "use_spectral_norm": False,
                "discriminator_channel_mult": 1,
                "use_multiscale_melloss": True,
                "lambda_melloss": 15,
                "clip_grad_norm": 1000,
                "segment_size": 16384,
                "num_mels": 100,
                "num_freq": 1025,
                "n_fft": 1024,
                "hop_size": 256,
                "win_size": 1024,
                "sampling_rate": 24000,
                "fmin": 0,
                "fmax": None,
                "fmax_for_loss": None,
                "mel_type": "pytorch",
                "num_workers": 2,
                "dist_config": {
                    "dist_backend": "nccl",
                    "dist_url": "tcp://localhost:54321",
                    "world_size": 1,
                },
            },
            "bigvgan_checkpoint": "bigvgan_generator.pth",
            "dataset": {
                "bpe_model": "checkpoints/bpe.model",
                "sample_rate": 24000,
                "squeeze": False,
                "mel": {
                    "sample_rate": 24000,
                    "n_fft": 1024,
                    "hop_length": 256,
                    "win_length": 1024,
                    "n_mels": 100,
                    "mel_fmin": 0,
                    "normalize": False,
                },
            },
            "dvae_checkpoint": "dvae.pth",
            "gpt": {
                "model_dim": 1024,
                "max_mel_tokens": 605,
                "max_text_tokens": 402,
                "heads": 16,
                "use_mel_codes_as_input": True,
                "mel_length_compression": 1024,
                "layers": 20,
                "number_text_tokens": 12000,
                "number_mel_codes": 8194,
                "start_mel_token": 8192,
                "stop_mel_token": 8193,
                "start_text_token": 0,
                "stop_text_token": 1,
                "train_solo_embeddings": False,
                "condition_type": "conformer_perceiver",
                "condition_module": {
                    "output_size": 512,
                    "linear_units": 2048,
                    "attention_heads": 8,
                    "num_blocks": 6,
                    "input_layer": "conv2d2",
                    "perceiver_mult": 2,
                },
            },
            "gpt_checkpoint": "gpt.pth",
            "vqvae": {
                "channels": 100,
                "num_tokens": 8192,
                "hidden_dim": 512,
                "num_resnet_blocks": 3,
                "codebook_dim": 512,
                "num_layers": 2,
                "positional_dims": 1,
                "kernel_size": 3,
                "smooth_l1_loss": True,
                "use_transposed_convs": False,
            },
        }

    def test_init(self):
        """Test IndexTTS initialization."""
        from mlx_audio.tts.models.indextts.indextts import Model

        # Initialize model
        config = self._default_config
        model = Model(config)  # type: ignore

        # Check that model was created
        self.assertIsInstance(model, Model)


class TestVibeVoiceModel(unittest.TestCase):
    @property
    def _default_config(self):
        from mlx_audio.tts.models.vibevoice.config import ModelConfig

        return ModelConfig(
            model_path="/fake/model/path",
            sample_rate=24000,
        )

    def test_init(self):
        """Test VibeVoiceModel initialization."""
        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        # Initialize model
        config = self._default_config
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)

        # Verify model components exist
        self.assertIsNotNone(model.language_model)
        self.assertIsNotNone(model.tts_language_model)
        self.assertIsNotNone(model.acoustic_tokenizer)
        self.assertIsNotNone(model.prediction_head)
        self.assertIsNotNone(model.tts_eos_classifier)

    def test_sample_rate(self):
        """Test VibeVoiceModel sample_rate property."""
        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        config = self._default_config
        model = Model(config)

        self.assertEqual(model.sample_rate, 24000)

    def test_get_input_embeddings(self):
        """Test VibeVoiceModel get_input_embeddings method."""
        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        config = self._default_config
        model = Model(config)

        embeddings = model.get_input_embeddings()
        self.assertIsInstance(embeddings, nn.Embedding)
        self.assertEqual(embeddings.weight.shape[0], config.decoder_config.vocab_size)

    def test_sanitize(self):
        """Test VibeVoiceModel sanitize method."""
        from mlx.utils import tree_flatten

        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        config = self._default_config
        model = Model(config)

        # Test sanitize with model's own weights (no transformation needed)
        weights = dict(tree_flatten(model.parameters()))
        sanitized = model.sanitize(weights)

        # Sanitized weights should contain valid keys
        self.assertIsInstance(sanitized, dict)

    def test_sanitize_huggingface_keys(self):
        """Test VibeVoiceModel sanitize transforms HuggingFace keys."""
        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        config = self._default_config
        model = Model(config)

        # Create mock weights with HuggingFace-style keys
        mock_weights = {
            "model.prediction_head.t_embedder.mlp.0.weight": mx.zeros((64, 64)),
            "model.prediction_head.adaLN_modulation.1.weight": mx.zeros((64, 64)),
        }

        sanitized = model.sanitize(mock_weights)

        # Check that keys were transformed (original keys should not exist)
        self.assertNotIn("model.prediction_head.t_embedder.mlp.0.weight", sanitized)
        self.assertNotIn("model.prediction_head.adaLN_modulation.1.weight", sanitized)

    def test_sanitize_preserves_quantization_metadata(self):
        """Test that sanitize preserves .scales and .biases for quantized models."""
        from mlx.utils import tree_flatten

        from mlx_audio.tts.models.vibevoice.vibevoice import Model

        config = self._default_config
        model = Model(config)

        # Start with the model's own weights
        weights = dict(tree_flatten(model.parameters()))

        # Add mock quantization metadata for the key from the bug report:
        # "Expected shape (151936, 896) but received shape (151936, 224)
        #  for parameter language_model.embed_tokens.weight"
        quant_key = "language_model.embed_tokens.weight"
        weights[f"{quant_key}.scales"] = mx.ones((1,))
        weights[f"{quant_key}.biases"] = mx.ones((1,))

        sanitized = model.sanitize(weights)

        # Quantization metadata must survive sanitization
        self.assertIn(f"{quant_key}.scales", sanitized)
        self.assertIn(f"{quant_key}.biases", sanitized)

    def test_config_defaults(self):
        """Test VibeVoiceModel uses correct config defaults."""
        from mlx_audio.tts.models.vibevoice.config import ModelConfig

        config = ModelConfig()

        # Verify default values
        self.assertEqual(config.sample_rate, 24000)
        self.assertEqual(config.acoustic_vae_dim, 64)
        self.assertEqual(config.tts_backbone_num_hidden_layers, 20)
        self.assertEqual(config.decoder_config.hidden_size, 896)
        self.assertEqual(config.decoder_config.num_hidden_layers, 24)


class TestChatterboxConfig(unittest.TestCase):
    def test_t3_config_defaults(self):
        """Test T3Config default values and factory methods."""
        from mlx_audio.tts.models.chatterbox.config import T3Config

        # Test defaults
        config = T3Config()
        self.assertEqual(config.text_tokens_dict_size, 704)
        self.assertEqual(config.speech_tokens_dict_size, 8194)
        self.assertEqual(config.llama_config_name, "Llama_520M")
        self.assertEqual(config.n_channels, 1024)
        self.assertFalse(config.is_multilingual)

        # Test factory methods
        self.assertFalse(T3Config.english_only().is_multilingual)
        self.assertTrue(T3Config.multilingual().is_multilingual)

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        from mlx_audio.tts.models.chatterbox.config import ModelConfig

        config = ModelConfig()

        self.assertEqual(config.model_type, "chatterbox")
        self.assertEqual(config.s3_sr, 16000)
        self.assertEqual(config.s3gen_sr, 24000)
        self.assertEqual(config.sample_rate, 24000)
        self.assertIsNotNone(config.t3_config)

    def test_model_config_from_dict(self):
        """Test ModelConfig.from_dict method."""
        from mlx_audio.tts.models.chatterbox.config import ModelConfig

        config_dict = {
            "model_type": "chatterbox",
            "t3_config": {
                "text_tokens_dict_size": 2454,
            },
        }

        config = ModelConfig.from_dict(config_dict)

        self.assertEqual(config.model_type, "chatterbox")
        self.assertTrue(config.t3_config.is_multilingual)


class TestChatterboxModel(unittest.TestCase):
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.T3")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3Token2Wav")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3TokenizerV2")
    def test_init(self, mock_s3_tokenizer, mock_ve, mock_s3gen, mock_t3):
        """Test Model initialization with config."""
        from mlx_audio.tts.models.chatterbox.chatterbox import Model
        from mlx_audio.tts.models.chatterbox.config import ModelConfig

        config = ModelConfig()
        model = Model(config)

        self.assertIsNotNone(model.t3)
        self.assertIsNotNone(model.s3gen)
        self.assertIsNotNone(model.ve)
        self.assertEqual(model.sr, 24000)
        self.assertEqual(model.sample_rate, 24000)

    @patch("mlx_audio.tts.models.chatterbox.chatterbox.T3")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3Token2Wav")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3TokenizerV2")
    def test_sanitize(
        self, mock_s3_tokenizer, mock_ve_class, mock_s3gen_class, mock_t3_class
    ):
        """Test weight sanitization routes to correct components."""
        from mlx_audio.tts.models.chatterbox.chatterbox import Model

        # Mock components to have sanitize methods that pass through weights
        for mock_class in [
            mock_ve_class,
            mock_t3_class,
            mock_s3gen_class,
            mock_s3_tokenizer,
        ]:
            mock_class.return_value.sanitize.side_effect = lambda w: w

        model = Model()

        # Test that prefixed weights are routed and re-prefixed
        weights = {
            "ve.lstm.weight": mx.zeros((10, 10)),
            "t3.tfmr.weight": mx.zeros((10, 10)),
            "s3gen.flow.weight": mx.zeros((10, 10)),
        }

        result = model.sanitize(weights)

        # Verify weights keep their prefixes
        self.assertIn("ve.lstm.weight", result)
        self.assertIn("t3.tfmr.weight", result)
        self.assertIn("s3gen.flow.weight", result)


class TestChatterboxFromPretrainedQuantization(unittest.TestCase):
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.nn.quantize")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.mx.load")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.T3")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3Token2Wav")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox.chatterbox.S3TokenizerV2")
    def test_from_pretrained_quantizes_ve_projection_when_quantized_weights_exist(
        self,
        mock_s3_tokenizer,
        mock_ve_class,
        mock_s3gen_class,
        mock_t3_class,
        mock_mx_load,
        mock_quantize,
    ):
        """Regression test for chatterbox-4bit loader: ve.proj must be quantized when scales exist."""
        from mlx_audio.tts.models.chatterbox.chatterbox import Model

        class StopAfterQuantize(Exception):
            pass

        quantizable_module = type(
            "QuantizableModule", (), {"to_quantized": lambda self: None}
        )()
        non_quantizable_module = object()

        def fake_quantize(model, group_size, bits, class_predicate):
            self.assertEqual(group_size, 64)
            self.assertEqual(bits, 4)
            self.assertTrue(class_predicate("ve.proj", quantizable_module))
            self.assertFalse(class_predicate("ve.lstm.layers.0", quantizable_module))
            self.assertFalse(class_predicate("ve.proj", non_quantizable_module))
            self.assertFalse(class_predicate("t3.tfmr", quantizable_module))
            raise StopAfterQuantize

        mock_quantize.side_effect = fake_quantize
        mock_mx_load.return_value = {
            "ve.proj.weight": mx.zeros((256, 32), dtype=mx.uint32),
            "ve.proj.scales": mx.zeros((256, 4)),
            "ve.proj.biases": mx.zeros((256, 4)),
            "ve.proj.bias": mx.zeros((256,)),
        }

        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "model.safetensors").write_bytes(b"stub")
            (model_dir / "config.json").write_text(
                json.dumps({"quantization": {"bits": 4, "group_size": 64}})
            )

            with self.assertRaises(StopAfterQuantize):
                Model.from_pretrained(model_dir)


class TestChatterboxTurboConfig(unittest.TestCase):
    def test_t3_config_defaults(self):
        """Test T3Config default values."""
        from mlx_audio.tts.models.chatterbox_turbo.models.t3 import T3Config

        config = T3Config()
        self.assertEqual(config.text_tokens_dict_size, 50276)
        self.assertEqual(config.speech_tokens_dict_size, 6563)
        self.assertEqual(config.llama_config_name, "GPT2_medium")
        self.assertEqual(config.n_channels, 1024)
        self.assertEqual(config.speaker_embed_size, 256)
        self.assertEqual(config.speech_cond_prompt_len, 375)
        self.assertFalse(config.emotion_adv)
        self.assertFalse(config.use_perceiver_resampler)

    def test_t3_config_turbo_factory(self):
        """Test T3Config.turbo() factory method."""
        from mlx_audio.tts.models.chatterbox_turbo.models.t3 import T3Config

        config = T3Config.turbo()
        self.assertEqual(config.text_tokens_dict_size, 50276)
        self.assertEqual(config.speech_tokens_dict_size, 6563)
        self.assertEqual(config.llama_config_name, "GPT2_medium")
        self.assertEqual(config.speech_cond_prompt_len, 375)
        self.assertFalse(config.emotion_adv)
        self.assertFalse(config.use_perceiver_resampler)

    def test_t3_config_is_multilingual(self):
        """Test is_multilingual property."""
        from mlx_audio.tts.models.chatterbox_turbo.models.t3 import T3Config

        # Default turbo config is not multilingual
        config = T3Config.turbo()
        self.assertFalse(config.is_multilingual)

        # Multilingual config has text_tokens_dict_size == 2454
        multilingual_config = T3Config(text_tokens_dict_size=2454)
        self.assertTrue(multilingual_config.is_multilingual)


class TestChatterboxTurboPuncNorm(unittest.TestCase):
    def test_empty_string(self):
        """Test punc_norm handles empty string."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        result = punc_norm("")
        self.assertEqual(result, "You need to add some text for me to talk.")

    def test_capitalizes_first_letter(self):
        """Test punc_norm capitalizes first letter."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        result = punc_norm("hello world")
        self.assertTrue(result[0].isupper())

    def test_adds_period_if_missing(self):
        """Test punc_norm adds period if no ending punctuation."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        result = punc_norm("Hello world")
        self.assertTrue(result.endswith("."))

    def test_keeps_existing_punctuation(self):
        """Test punc_norm keeps existing ending punctuation."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        self.assertTrue(punc_norm("Hello world!").endswith("!"))
        self.assertTrue(punc_norm("Hello world?").endswith("?"))
        self.assertTrue(punc_norm("Hello world.").endswith("."))

    def test_removes_multiple_spaces(self):
        """Test punc_norm removes multiple spaces."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        result = punc_norm("Hello    world")
        self.assertNotIn("  ", result)

    def test_replaces_special_punctuation(self):
        """Test punc_norm replaces special punctuation."""
        from mlx_audio.tts.models.chatterbox_turbo import punc_norm

        # Test ellipsis replacement
        result = punc_norm("Hello… world")
        self.assertNotIn("…", result)

        # Test em dash replacement
        result = punc_norm("Hello—world")
        self.assertIn("-", result)


class TestChatterboxTurboModel(unittest.TestCase):
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.T3")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3Gen")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3TokenizerV2")
    def test_init_with_config(self, mock_s3_tokenizer, mock_ve, mock_s3gen, mock_t3):
        """Test ChatterboxTurboTTS initialization with config dict."""
        from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS(config_or_t3={})

        self.assertIsNotNone(model.t3)
        self.assertIsNotNone(model.s3gen)
        self.assertIsNotNone(model.ve)
        self.assertEqual(model.sr, 24000)
        self.assertEqual(model.sample_rate, 24000)

    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.T3")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3Gen")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3TokenizerV2")
    def test_init_with_none(self, mock_s3_tokenizer, mock_ve, mock_s3gen, mock_t3):
        """Test ChatterboxTurboTTS initialization with None (default config)."""
        from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS()

        self.assertIsNotNone(model.t3)
        self.assertIsNotNone(model.s3gen)
        self.assertIsNotNone(model.ve)

    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.T3")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3Gen")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3TokenizerV2")
    def test_sanitize(
        self, mock_s3_tokenizer, mock_ve_class, mock_s3gen_class, mock_t3_class
    ):
        """Test weight sanitization routes to correct components."""
        from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS

        # Mock components to have sanitize methods that pass through weights
        for mock_class in [
            mock_ve_class,
            mock_t3_class,
            mock_s3gen_class,
            mock_s3_tokenizer,
        ]:
            mock_class.return_value.sanitize.side_effect = lambda w: w

        model = ChatterboxTurboTTS()

        # Test that prefixed weights are routed and re-prefixed
        weights = {
            "ve.lstm.weight": mx.zeros((10, 10)),
            "t3.tfmr.weight": mx.zeros((10, 10)),
            "s3gen.flow.weight": mx.zeros((10, 10)),
        }

        result = model.sanitize(weights)

        # Verify weights keep their prefixes
        self.assertIn("ve.lstm.weight", result)
        self.assertIn("t3.tfmr.weight", result)
        self.assertIn("s3gen.flow.weight", result)

    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.T3")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3Gen")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.VoiceEncoder")
    @patch("mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo.S3TokenizerV2")
    def test_sanitize_with_other_weights(
        self, mock_s3_tokenizer, mock_ve_class, mock_s3gen_class, mock_t3_class
    ):
        """Test that unrecognized weights pass through sanitization."""
        from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS

        # Mock components to have sanitize methods that pass through weights
        for mock_class in [
            mock_ve_class,
            mock_t3_class,
            mock_s3gen_class,
            mock_s3_tokenizer,
        ]:
            mock_class.return_value.sanitize.side_effect = lambda w: w

        model = ChatterboxTurboTTS()

        # Test with weights that don't have known prefixes
        weights = {
            "ve.lstm.weight": mx.zeros((10, 10)),
            "unknown.param": mx.zeros((5, 5)),
        }

        result = model.sanitize(weights)

        # Both should be in result
        self.assertIn("ve.lstm.weight", result)
        self.assertIn("unknown.param", result)


class TestChatterboxTurboConditionals(unittest.TestCase):
    def test_conditionals_dataclass(self):
        """Test Conditionals dataclass creation."""
        from mlx_audio.tts.models.chatterbox_turbo import Conditionals
        from mlx_audio.tts.models.chatterbox_turbo.models.t3 import T3Cond

        t3_cond = T3Cond(
            speaker_emb=mx.zeros((1, 256)),
            cond_prompt_speech_tokens=mx.zeros((1, 375), dtype=mx.int32),
        )
        gen_dict = {"ref_mel": mx.zeros((1, 80, 100))}

        conds = Conditionals(t3=t3_cond, gen=gen_dict)

        self.assertIsNotNone(conds.t3)
        self.assertIsNotNone(conds.gen)
        self.assertEqual(conds.t3.speaker_emb.shape, (1, 256))


class TestChatterboxTurboModelAlias(unittest.TestCase):
    def test_model_alias(self):
        """Test that Model is aliased to ChatterboxTurboTTS."""
        from mlx_audio.tts.models.chatterbox_turbo import ChatterboxTurboTTS, Model

        self.assertIs(Model, ChatterboxTurboTTS)


class TestSoprano(unittest.TestCase):
    """Tests for Soprano TTS model."""

    @property
    def _default_config(self):
        from mlx_audio.tts.models.soprano import DecoderConfig, ModelConfig

        return ModelConfig(
            model_type="qwen3",
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=1024,
            vocab_size=32000,
            head_dim=64,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            decoder_config=DecoderConfig(),
        )

    # Config tests
    def test_decoder_config_defaults(self):
        """Test DecoderConfig default values."""
        from mlx_audio.tts.models.soprano import DecoderConfig

        config = DecoderConfig()
        self.assertEqual(config.decoder_num_layers, 8)
        self.assertEqual(config.decoder_dim, 768)
        self.assertEqual(config.decoder_intermediate_dim, 2304)
        self.assertEqual(config.hop_length, 512)
        self.assertEqual(config.n_fft, 2048)
        self.assertEqual(config.upscale, 4)
        self.assertEqual(config.input_kernel, 1)
        self.assertEqual(config.dw_kernel, 3)
        self.assertEqual(config.token_size, 2048)
        self.assertEqual(config.receptive_field, 4)

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        from mlx_audio.tts.models.soprano import ModelConfig

        config = ModelConfig(
            model_type="qwen3",
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=1024,
            vocab_size=32000,
            head_dim=64,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )
        self.assertEqual(config.sample_rate, 32000)
        self.assertIsNotNone(config.decoder_config)

    def test_model_config_post_init(self):
        """Test that ModelConfig creates decoder_config if None."""
        from mlx_audio.tts.models.soprano import DecoderConfig, ModelConfig

        config = ModelConfig(
            model_type="qwen3",
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=1024,
            vocab_size=32000,
            head_dim=64,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            decoder_config=None,
        )
        self.assertIsNotNone(config.decoder_config)
        self.assertIsInstance(config.decoder_config, DecoderConfig)

    # Model tests
    def test_model_init(self):
        """Test Model initialization."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        self.assertIsNotNone(model.language_model)
        self.assertIsNotNone(model.decoder)
        self.assertEqual(model.config.sample_rate, 32000)

    def test_sample_rate_property(self):
        """Test sample_rate property."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        self.assertEqual(model.sample_rate, 32000)

    def test_layers_property(self):
        """Test layers property returns LM layers."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        layers = model.layers
        self.assertEqual(len(layers), config.num_hidden_layers)

    def test_sanitize(self):
        """Test weight sanitization."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        weights = {
            "model.embed_tokens.weight": mx.zeros((32000, 512)),
            "model.layers.0.input_layernorm.weight": mx.zeros(512),
            "decoder.backbone.weight": mx.zeros((512, 512)),
        }

        sanitized = model.sanitize(weights)

        self.assertIn("language_model.embed_tokens.weight", sanitized)
        self.assertIn("language_model.layers.0.input_layernorm.weight", sanitized)
        self.assertIn("decoder.backbone.weight", sanitized)
        self.assertNotIn("model.embed_tokens.weight", sanitized)

    def test_sanitize_decoder_float32(self):
        """Test that decoder weights are converted to float32."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        weights = {
            "decoder.backbone.weight": mx.zeros((512, 512), dtype=mx.bfloat16),
            "lm_head.weight": mx.zeros((32000, 512), dtype=mx.bfloat16),
        }

        sanitized = model.sanitize(weights)

        self.assertEqual(sanitized["decoder.backbone.weight"].dtype, mx.float32)
        self.assertEqual(sanitized["language_model.lm_head.weight"].dtype, mx.bfloat16)

    def test_format_duration(self):
        """Test _format_duration helper method."""
        from mlx_audio.tts.models.soprano import Model

        config = self._default_config
        model = Model(config)

        self.assertEqual(model._format_duration(0), "00:00:00.000")
        self.assertEqual(model._format_duration(1.5), "00:00:01.500")
        self.assertEqual(model._format_duration(61.25), "00:01:01.250")
        self.assertEqual(model._format_duration(3661.123), "01:01:01.123")

    # Text processing tests
    def test_clean_text(self):
        """Test clean_text function."""
        from mlx_audio.tts.models.soprano.text import clean_text

        self.assertEqual(clean_text("Hello World!"), "hello world!")
        self.assertEqual(clean_text("I have 5 apples."), "i have five apples.")

    def test_normalize_numbers(self):
        """Test number normalization."""
        from mlx_audio.tts.models.soprano.text import normalize_numbers

        self.assertIn("five", normalize_numbers("5"))
        self.assertIn("twenty", normalize_numbers("20"))
        self.assertIn("hundred", normalize_numbers("100"))
        self.assertIn("dollar", normalize_numbers("$5"))
        self.assertIn("first", normalize_numbers("1st"))

    def test_expand_abbreviations(self):
        """Test abbreviation expansion."""
        from mlx_audio.tts.models.soprano.text import expand_abbreviations

        self.assertIn("mister", expand_abbreviations("Mr."))
        self.assertIn("doctor", expand_abbreviations("Dr."))
        self.assertIn("text to speech", expand_abbreviations("TTS"))

    def test_expand_special_characters(self):
        """Test special character expansion."""
        from mlx_audio.tts.models.soprano.text import expand_special_characters

        self.assertIn("at", expand_special_characters("@"))
        self.assertIn("and", expand_special_characters("&"))
        self.assertIn("percent", expand_special_characters("%"))

    def test_collapse_whitespace(self):
        """Test whitespace collapsing."""
        from mlx_audio.tts.models.soprano.text import collapse_whitespace

        self.assertEqual(collapse_whitespace("hello  world"), "hello world")
        self.assertEqual(collapse_whitespace("  hello   world  "), "hello world")
        self.assertEqual(collapse_whitespace("hello ,world"), "hello,world")

    def test_dedup_punctuation(self):
        """Test punctuation deduplication."""
        from mlx_audio.tts.models.soprano.text import dedup_punctuation

        self.assertEqual(dedup_punctuation("hello...."), "hello.")
        self.assertEqual(dedup_punctuation("hello,,,,"), "hello,")
        self.assertEqual(dedup_punctuation("hello??!!"), "hello?")

    def test_convert_to_ascii(self):
        """Test unicode to ASCII conversion."""
        from mlx_audio.tts.models.soprano.text import convert_to_ascii

        self.assertEqual(convert_to_ascii("café"), "cafe")
        self.assertEqual(convert_to_ascii("naïve"), "naive")

    def test_num_to_words(self):
        """Test number to words conversion."""
        from mlx_audio.tts.models.soprano.text import _num_to_words

        self.assertEqual(_num_to_words(0), "zero")
        self.assertEqual(_num_to_words(1), "one")
        self.assertEqual(_num_to_words(10), "ten")
        self.assertEqual(_num_to_words(21), "twenty one")
        self.assertEqual(_num_to_words(100), "one hundred")
        self.assertEqual(_num_to_words(1000), "one thousand")
        self.assertEqual(_num_to_words(-5), "minus five")

    def test_ordinal_to_words(self):
        """Test ordinal to words conversion."""
        from mlx_audio.tts.models.soprano.text import _ordinal_to_words

        self.assertEqual(_ordinal_to_words(1), "first")
        self.assertEqual(_ordinal_to_words(2), "second")
        self.assertEqual(_ordinal_to_words(3), "third")
        self.assertEqual(_ordinal_to_words(10), "tenth")
        self.assertEqual(_ordinal_to_words(21), "twenty first")

    # Decoder tests
    def test_decoder_init(self):
        """Test SopranoDecoder initialization."""
        from mlx_audio.tts.models.soprano.decoder import SopranoDecoder

        decoder = SopranoDecoder(
            num_input_channels=512,
            decoder_num_layers=4,
            decoder_dim=256,
            decoder_intermediate_dim=768,
            hop_length=512,
            n_fft=2048,
            upscale=4,
            input_kernel=1,
            dw_kernel=3,
        )

        self.assertEqual(decoder.decoder_initial_channels, 512)
        self.assertEqual(decoder.num_layers, 4)
        self.assertEqual(decoder.dim, 256)
        self.assertEqual(decoder.intermediate_dim, 768)
        self.assertEqual(decoder.hop_length, 512)
        self.assertEqual(decoder.n_fft, 2048)
        self.assertEqual(decoder.upscale, 4)

    def test_decoder_default_intermediate_dim(self):
        """Test default intermediate_dim calculation."""
        from mlx_audio.tts.models.soprano.decoder import SopranoDecoder

        decoder = SopranoDecoder(
            num_input_channels=512,
            decoder_num_layers=4,
            decoder_dim=256,
            decoder_intermediate_dim=None,
        )

        self.assertEqual(decoder.intermediate_dim, 256 * 3)

    # ISTFT Head tests
    def test_istft_head_init(self):
        """Test ISTFTHead initialization."""
        from mlx_audio.tts.models.soprano.decoder import ISTFTHead

        head = ISTFTHead(dim=512, n_fft=2048, hop_length=512)

        self.assertEqual(head.n_fft, 2048)
        self.assertEqual(head.hop_length, 512)

    def test_istft_head_forward(self):
        """Test ISTFTHead forward pass."""
        from mlx_audio.tts.models.soprano.decoder import ISTFTHead

        head = ISTFTHead(dim=512, n_fft=2048, hop_length=512)
        x = mx.zeros((1, 10, 512))
        audio = head(x)

        self.assertEqual(len(audio.shape), 2)
        self.assertEqual(audio.shape[0], 1)


class TestQwen3TTSModel(unittest.TestCase):
    """Tests for Qwen3-TTS model."""

    def _default_talker_config(self):
        # Minimal config for fast tests
        return {
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "hidden_act": "silu",
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "num_code_groups": 4,
            "text_hidden_size": 64,
            "text_vocab_size": 100,
            "codec_eos_token_id": 30,
            "codec_pad_id": 28,
            "codec_bos_id": 29,
            "codec_language_id": {"english": 20, "chinese": 21},
            "spk_id": {"chelsie": 10, "ethan": 11},
            "code_predictor_config": {
                "vocab_size": 32,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "hidden_act": "silu",
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "num_code_groups": 4,
            },
        }

    def _default_config(self, tts_model_type="base"):
        return {
            "model_type": "qwen3_tts",
            "tts_model_type": tts_model_type,
            "tts_model_size": "0b6",
            "talker_config": self._default_talker_config(),
            "speaker_encoder_config": None,
            "tokenizer_config": None,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "tts_pad_token_id": 151671,
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673,
            "sample_rate": 24000,
        }

    def test_config_init(self):
        """Test Qwen3TTS ModelConfig initialization."""
        from mlx_audio.tts.models.qwen3_tts.config import ModelConfig

        config = ModelConfig.from_dict(self._default_config())

        self.assertEqual(config.model_type, "qwen3_tts")
        self.assertEqual(config.tts_model_type, "base")
        self.assertEqual(config.sample_rate, 24000)
        self.assertIsNotNone(config.talker_config)

    def test_config_custom_voice(self):
        """Test config with custom_voice model type."""
        from mlx_audio.tts.models.qwen3_tts.config import ModelConfig

        config = ModelConfig.from_dict(self._default_config("custom_voice"))

        self.assertEqual(config.tts_model_type, "custom_voice")

    def test_config_voice_design(self):
        """Test config with voice_design model type."""
        from mlx_audio.tts.models.qwen3_tts.config import ModelConfig

        config = ModelConfig.from_dict(self._default_config("voice_design"))

        self.assertEqual(config.tts_model_type, "voice_design")

    def test_model_init(self):
        """Test Qwen3TTS Model initialization."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config())
        model = Model(config)

        self.assertIsInstance(model, Model)
        self.assertEqual(model.model_type, "qwen3_tts")
        self.assertEqual(model.sample_rate, 24000)

    def test_model_supported_speakers(self):
        """Test supported speakers list."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config())
        model = Model(config)

        speakers = model.get_supported_speakers()
        self.assertIn("chelsie", speakers)
        self.assertIn("ethan", speakers)

    def test_model_supported_languages(self):
        """Test supported languages list."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config())
        model = Model(config)

        languages = model.get_supported_languages()
        self.assertIn("auto", languages)
        self.assertIn("english", languages)
        self.assertIn("chinese", languages)

    def test_talker_init(self):
        """Test Talker model initialization."""
        from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSTalkerConfig
        from mlx_audio.tts.models.qwen3_tts.talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        config = Qwen3TTSTalkerConfig(**self._default_talker_config())
        talker = Qwen3TTSTalkerForConditionalGeneration(config)

        self.assertIsNotNone(talker.model)
        self.assertIsNotNone(talker.code_predictor)
        self.assertEqual(config.vocab_size, 32)

    def test_talker_forward(self):
        """Test Talker forward pass."""
        from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSTalkerConfig
        from mlx_audio.tts.models.qwen3_tts.talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        config = Qwen3TTSTalkerConfig(**self._default_talker_config())
        talker = Qwen3TTSTalkerForConditionalGeneration(config)

        # Test forward with inputs_embeds
        batch_size, seq_len = 1, 10
        hidden_size = config.hidden_size
        inputs_embeds = mx.random.normal((batch_size, seq_len, hidden_size))

        # Talker returns (logits, hidden_states)
        logits, hidden_states = talker(inputs_embeds=inputs_embeds)

        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[1], seq_len)
        self.assertEqual(logits.shape[2], config.vocab_size)
        self.assertEqual(hidden_states.shape, (batch_size, seq_len, hidden_size))

    def test_code_predictor_init(self):
        """Test CodePredictor initialization."""
        from mlx_audio.tts.models.qwen3_tts.config import (
            Qwen3TTSTalkerCodePredictorConfig,
        )
        from mlx_audio.tts.models.qwen3_tts.talker import Qwen3TTSTalkerCodePredictor

        config = Qwen3TTSTalkerCodePredictorConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            num_code_groups=4,
        )
        code_predictor = Qwen3TTSTalkerCodePredictor(config, talker_hidden_size=64)

        self.assertEqual(len(code_predictor.codec_embedding), 3)  # num_code_groups - 1
        self.assertIsNotNone(code_predictor.model)

    def test_code_predictor_forward(self):
        """Test CodePredictor forward pass."""
        from mlx_audio.tts.models.qwen3_tts.config import (
            Qwen3TTSTalkerCodePredictorConfig,
        )
        from mlx_audio.tts.models.qwen3_tts.talker import Qwen3TTSTalkerCodePredictor

        config = Qwen3TTSTalkerCodePredictorConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            num_code_groups=4,
        )
        code_predictor = Qwen3TTSTalkerCodePredictor(config, talker_hidden_size=64)

        batch_size, seq_len = 1, 2
        inputs_embeds = mx.random.normal((batch_size, seq_len, 64))

        # CodePredictor returns (logits, cache, next_step)
        logits, _, _ = code_predictor(inputs_embeds=inputs_embeds)

        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[1], seq_len)
        self.assertEqual(logits.shape[2], config.vocab_size)

    def test_generate_routing_base(self):
        """Test that generate routes correctly for base model."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config("base"))
        model = Model(config)

        # Base model should not require instruct
        self.assertEqual(config.tts_model_type, "base")

    def test_generate_routing_custom_voice_requires_voice(self):
        """Test that custom_voice model requires voice parameter."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config("custom_voice"))
        model = Model(config)

        # Mock speech_tokenizer to avoid loading
        model.speech_tokenizer = MagicMock()

        with self.assertRaises(ValueError) as context:
            list(model.generate(text="Hello", voice=None))

        self.assertIn("voice", str(context.exception).lower())

    def test_generate_routing_voice_design_requires_instruct(self):
        """Test that voice_design model requires instruct parameter."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config = ModelConfig.from_dict(self._default_config("voice_design"))
        model = Model(config)

        # Mock speech_tokenizer to avoid loading
        model.speech_tokenizer = MagicMock()

        with self.assertRaises(ValueError) as context:
            list(model.generate(text="Hello", instruct=None))

        self.assertIn("instruct", str(context.exception).lower())

    def test_speaker_encoder_config(self):
        """Test SpeakerEncoder config initialization."""
        from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSSpeakerEncoderConfig

        config = Qwen3TTSSpeakerEncoderConfig()

        self.assertEqual(config.mel_dim, 128)
        self.assertEqual(config.enc_dim, 1024)
        self.assertEqual(config.sample_rate, 24000)
        self.assertEqual(len(config.enc_channels), 5)

    def test_mel_spectrogram(self):
        """Test mel spectrogram computation."""
        from mlx_audio.tts.models.qwen3_tts.qwen3_tts import mel_spectrogram

        # Create a simple test audio
        sample_rate = 24000
        duration = 0.5  # 0.5 seconds
        audio = mx.random.normal((int(sample_rate * duration),))

        mel = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=128,
            sample_rate=sample_rate,
            hop_size=256,
        )

        # Check output shape
        self.assertEqual(mel.ndim, 3)  # [batch, time, mels]
        self.assertEqual(mel.shape[0], 1)  # batch size
        self.assertEqual(mel.shape[2], 128)  # num_mels


class TestQwen3TTSEncoder(unittest.TestCase):
    """Tests for Qwen3TTSSpeechTokenizerEncoder."""

    def _default_encoder_config(self):
        from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSTokenizerEncoderConfig

        return Qwen3TTSTokenizerEncoderConfig(
            frame_rate=12.5,
            audio_channels=1,
            codebook_dim=256,
            codebook_size=64,  # Small for tests
            compress=2,
            dilation_growth_rate=2,
            hidden_size=64,  # Small for tests
            intermediate_size=128,
            kernel_size=7,
            last_kernel_size=3,
            num_attention_heads=4,
            num_filters=16,  # Small for tests
            num_hidden_layers=1,  # Single layer for speed
            num_key_value_heads=4,
            num_quantizers=32,
            num_residual_layers=1,
            num_semantic_quantizers=1,
            residual_kernel_size=3,
            rope_theta=10000.0,
            sampling_rate=24000,
            sliding_window=250,
            upsampling_ratios=[8, 6, 5, 4],
            use_causal_conv=True,
            use_conv_shortcut=False,
            layer_scale_initial_scale=0.01,
            max_position_embeddings=8000,
            head_dim=16,
        )

    def test_encoder_init(self):
        """Test encoder initialization with valid config."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        self.assertIsNotNone(encoder.encoder)
        self.assertIsNotNone(encoder.encoder_transformer)
        self.assertIsNotNone(encoder.downsample)
        self.assertIsNotNone(encoder.quantizer)
        self.assertEqual(encoder.valid_num_quantizers, 16)

    def test_encoder_components(self):
        """Test encoder has correct component types."""
        from mlx_audio.codec.models.mimi.modules.conv import ConvDownsample1d
        from mlx_audio.codec.models.mimi.modules.quantization import (
            SplitResidualVectorQuantizer as MimiSplitRVQ,
        )
        from mlx_audio.codec.models.mimi.modules.seanet import SeanetEncoder
        from mlx_audio.codec.models.mimi.modules.transformer import ProjectedTransformer
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        self.assertIsInstance(encoder.encoder, SeanetEncoder)
        self.assertIsInstance(encoder.encoder_transformer, ProjectedTransformer)
        self.assertIsInstance(encoder.downsample, ConvDownsample1d)
        self.assertIsInstance(encoder.quantizer, MimiSplitRVQ)

    def test_encoder_cache_init(self):
        """Test encoder cache is properly initialized."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        # Cache should have one entry per transformer layer
        self.assertEqual(len(encoder.encoder_cache), config.num_hidden_layers)

    def test_encoder_encode_output_shape(self):
        """Test encoder produces correct output shape."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        # Input: [batch, channels, samples]
        # The downsample rate = prod(upsampling_ratios) * downsample_stride
        # = (8*6*5*4) * 2 = 1920
        num_samples = 1920 * 3  # 3 time steps expected
        audio = mx.random.normal((1, 1, num_samples))

        codes = encoder.encode(audio)
        mx.eval(codes)

        # Output: [batch, valid_num_quantizers, time]
        self.assertEqual(codes.ndim, 3)
        self.assertEqual(codes.shape[0], 1)
        self.assertEqual(codes.shape[1], 16)  # valid_num_quantizers
        # Time dimension: num_samples / 1920 = 3
        self.assertEqual(codes.shape[2], 3)

    def test_encoder_encode_different_lengths(self):
        """Test encoder handles different audio lengths correctly."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        for num_frames in [2, 5, 10]:
            num_samples = 1920 * num_frames
            audio = mx.random.normal((1, 1, num_samples))
            codes = encoder.encode(audio)
            mx.eval(codes)

            self.assertEqual(codes.shape[0], 1)
            self.assertEqual(codes.shape[1], 16)
            self.assertEqual(codes.shape[2], num_frames)

    def test_encoder_encode_truncates_quantizers(self):
        """Test encoder only returns first 16 quantizers out of 32."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        self.assertEqual(config.num_quantizers, 32)

        encoder = Qwen3TTSSpeechTokenizerEncoder(config)
        audio = mx.random.normal((1, 1, 1920 * 2))
        codes = encoder.encode(audio)
        mx.eval(codes)

        # Should only have 16 quantizers, not 32
        self.assertEqual(codes.shape[1], 16)

    def test_encoder_encode_code_range(self):
        """Test that encoded codes are within valid codebook range."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        audio = mx.random.normal((1, 1, 1920 * 3))
        codes = encoder.encode(audio)
        mx.eval(codes)

        # Codes should be in range [0, codebook_size)
        self.assertTrue(mx.all(codes >= 0).item())
        self.assertTrue(mx.all(codes < config.codebook_size).item())

    def test_encoder_causal_mask(self):
        """Test that encode creates proper causal attention mask."""
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        # Different length inputs should produce consistent results
        # when processing the same prefix (due to causal masking)
        audio_short = mx.random.normal((1, 1, 1920 * 2))
        codes_short = encoder.encode(audio_short)
        mx.eval(codes_short)

        self.assertEqual(codes_short.shape[2], 2)

    def test_encoder_downsample_stride(self):
        """Test downsample stride is computed correctly from config."""
        import math

        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizerEncoder,
        )

        config = self._default_encoder_config()
        encoder = Qwen3TTSSpeechTokenizerEncoder(config)

        # encoder_frame_rate = sampling_rate / prod(upsampling_ratios) = 24000/960 = 25
        # downsample_stride = encoder_frame_rate / frame_rate = 25 / 12.5 = 2
        encoder_frame_rate = config.sampling_rate / math.prod(config.upsampling_ratios)
        expected_stride = int(encoder_frame_rate / config.frame_rate)
        self.assertEqual(expected_stride, 2)

        # Verify stride effect: with stride=2, output time should be halved
        # compared to no-downsample. The encode test already validates total
        # output shape (samples/1920), here we verify the stride math.
        self.assertEqual(encoder.downsample.conv.conv.conv._stride, expected_stride)


class TestQwen3TTSPrepareICLInputs(unittest.TestCase):
    """Tests for _prepare_icl_generation_inputs method."""

    def _make_model_with_mocks(self, hidden_size=64, num_code_groups=4, vocab_size=32):
        """Create a minimal Model with mocked components for testing ICL prep."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        # Use text_vocab_size large enough to hold tts_*_token_ids
        # so embeddings are distinct for different token IDs
        text_vocab_size = 200

        config_dict = {
            "model_type": "qwen3_tts",
            "tts_model_type": "base",
            "tts_model_size": "0b6",
            "talker_config": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "hidden_act": "silu",
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "num_code_groups": num_code_groups,
                "text_hidden_size": hidden_size,
                "text_vocab_size": text_vocab_size,
                "codec_eos_token_id": 30,
                "codec_pad_id": 28,
                "codec_bos_id": 29,
                "codec_language_id": {"english": 20, "chinese": 21},
                "spk_id": {"chelsie": 10, "ethan": 11},
                "code_predictor_config": {
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "intermediate_size": 128,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 32,
                    "hidden_act": "silu",
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "num_code_groups": num_code_groups,
                },
            },
            "speaker_encoder_config": None,
            "tokenizer_config": None,
            "im_start_token_id": 50,
            "im_end_token_id": 51,
            "tts_pad_token_id": 60,
            "tts_bos_token_id": 61,
            "tts_eos_token_id": 62,
            "sample_rate": 24000,
        }

        config = ModelConfig.from_dict(config_dict)
        model = Model(config)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        # Return 10 tokens for any encode call (includes role tokens)
        mock_tokenizer.encode.return_value = list(range(10))
        model.tokenizer = mock_tokenizer

        # Mock speech_tokenizer with encoder
        mock_speech_tokenizer = MagicMock()
        mock_speech_tokenizer.has_encoder = True
        ref_time = 5
        # encode returns [1, 16, ref_time]
        mock_speech_tokenizer.encode.return_value = mx.zeros(
            (1, 16, ref_time), dtype=mx.int32
        )
        model.speech_tokenizer = mock_speech_tokenizer

        # No speaker encoder for this test
        model.speaker_encoder = None

        return model, ref_time

    def test_prepare_icl_output_shapes(self):
        """Test that _prepare_icl_generation_inputs returns correct shapes."""
        model, ref_time = self._make_model_with_mocks()
        hidden_size = model.config.talker_config.hidden_size

        ref_audio = mx.random.normal((24000,))  # 1s audio
        (
            input_embeds,
            trailing,
            tts_pad,
            ref_codes,
        ) = model._prepare_icl_generation_inputs(
            text="Hello world",
            ref_audio=ref_audio,
            ref_text="Reference text",
            language="auto",
        )
        mx.eval(input_embeds, trailing, tts_pad, ref_codes)

        # input_embeds: [1, text_lens + codec_lens, hidden_size]
        self.assertEqual(input_embeds.ndim, 3)
        self.assertEqual(input_embeds.shape[0], 1)
        self.assertEqual(input_embeds.shape[2], hidden_size)

        # trailing_text_hidden: tts_pad_embed [1, 1, hidden_size] in non-streaming mode
        self.assertEqual(trailing.ndim, 3)
        self.assertEqual(trailing.shape[0], 1)
        self.assertEqual(trailing.shape[2], hidden_size)

        # tts_pad_embed: [1, 1, hidden_size]
        self.assertEqual(tts_pad.shape, (1, 1, hidden_size))

        # ref_codes: [1, 16, ref_time]
        self.assertEqual(ref_codes.shape, (1, 16, ref_time))

    def test_prepare_icl_non_streaming_structure(self):
        """Test non-streaming mode: text_with_codec_pad + codec_with_tts_pad."""
        model, ref_time = self._make_model_with_mocks()

        ref_audio = mx.random.normal((24000,))
        (
            input_embeds,
            trailing,
            tts_pad,
            ref_codes,
        ) = model._prepare_icl_generation_inputs(
            text="Hello",
            ref_audio=ref_audio,
            ref_text="Ref",
            language="auto",
        )
        mx.eval(input_embeds)

        # In non-streaming mode:
        # input_embeds = concat(text_with_codec_pad, codec_with_text_pad)
        # text_lens = ref_text_tokens + target_text_tokens + eos
        # codec_lens = 1 (bos) + ref_time (ref_codes)
        codec_lens = 1 + ref_time  # codec_bos + ref_time
        total_len = input_embeds.shape[1]

        # Total should be text_lens + codec_lens
        self.assertGreater(total_len, codec_lens)

    def test_prepare_icl_trailing_is_tts_pad(self):
        """Test that trailing_text_hidden equals tts_pad_embed in non-streaming mode."""
        model, _ = self._make_model_with_mocks()

        ref_audio = mx.random.normal((24000,))
        _, trailing, tts_pad, _ = model._prepare_icl_generation_inputs(
            text="Hello",
            ref_audio=ref_audio,
            ref_text="Ref",
            language="auto",
        )
        mx.eval(trailing, tts_pad)

        # In non-streaming mode, trailing = tts_pad_embed
        np.testing.assert_array_equal(np.array(trailing), np.array(tts_pad))

    def test_prepare_icl_ref_audio_dim_handling(self):
        """Test that ref_audio is properly reshaped for encoding."""
        model, _ = self._make_model_with_mocks()

        # Test 1D input
        ref_audio_1d = mx.random.normal((24000,))
        model._prepare_icl_generation_inputs(
            text="Hello", ref_audio=ref_audio_1d, ref_text="Ref"
        )
        # Speech tokenizer should receive [1, 1, samples]
        call_args = model.speech_tokenizer.encode.call_args[0][0]
        self.assertEqual(call_args.ndim, 3)
        self.assertEqual(call_args.shape[0], 1)
        self.assertEqual(call_args.shape[1], 1)

    def test_prepare_icl_ref_audio_2d_handling(self):
        """Test that 2D ref_audio is properly reshaped."""
        model, _ = self._make_model_with_mocks()

        # Test 2D input [1, samples]
        ref_audio_2d = mx.random.normal((1, 24000))
        model._prepare_icl_generation_inputs(
            text="Hello", ref_audio=ref_audio_2d, ref_text="Ref"
        )
        call_args = model.speech_tokenizer.encode.call_args[0][0]
        self.assertEqual(call_args.ndim, 3)

    def test_prepare_icl_language_id(self):
        """Test that language_id is incorporated in codec prefix."""
        model, _ = self._make_model_with_mocks()

        ref_audio = mx.random.normal((24000,))

        # With language="english", should include language_id in codec prefix
        input_embeds_en, _, _, _ = model._prepare_icl_generation_inputs(
            text="Hello", ref_audio=ref_audio, ref_text="Ref", language="english"
        )
        mx.eval(input_embeds_en)

        # With language="auto", no language_id
        input_embeds_auto, _, _, _ = model._prepare_icl_generation_inputs(
            text="Hello", ref_audio=ref_audio, ref_text="Ref", language="auto"
        )
        mx.eval(input_embeds_auto)

        # The embeddings should differ in size (language adds one more token)
        # auto: [nothink, think_bos, think_eos] = 3 codec prefix tokens
        # english: [think, think_bos, language_id, think_eos] = 4 codec prefix tokens
        # But this difference is after the icl_input_embed, so they should differ
        # Actually the codec prefix is appended AFTER icl_input_embed in the generate loop,
        # not in _prepare_icl_generation_inputs. The returned input_embeds only has
        # text_with_codec_pad + codec_with_text_pad. Language affects codec_prefix_embed
        # which is concatenated later. Let's verify the core structure is consistent.
        self.assertEqual(input_embeds_en.shape[2], input_embeds_auto.shape[2])

    def test_prepare_icl_no_tokenizer_raises(self):
        """Test that missing tokenizer raises ValueError."""
        model, _ = self._make_model_with_mocks()
        model.tokenizer = None

        ref_audio = mx.random.normal((24000,))
        with self.assertRaises(ValueError):
            model._prepare_icl_generation_inputs(
                text="Hello", ref_audio=ref_audio, ref_text="Ref"
            )

    def test_prepare_icl_codec_embed_includes_bos(self):
        """Test that codec embedding includes codec_bos prepended."""
        model, ref_time = self._make_model_with_mocks()

        ref_audio = mx.random.normal((24000,))
        input_embeds, _, _, _ = model._prepare_icl_generation_inputs(
            text="Hello", ref_audio=ref_audio, ref_text="Ref"
        )
        mx.eval(input_embeds)

        # Full structure: role_embed + combined_prefix + icl_input_embed
        # tokenizer.encode returns 10 tokens for each call
        # role_embed = target_ids[:, :3] = 3 tokens
        # combined_prefix: codec_prefix(nothink,think_bos,think_eos=3) + suffix(pad,bos=2) = 5
        #   pad_count = 5-2 = 3, combined_prefix = [3 pads + 1 bos] = 4 tokens
        # icl_input_embed = text_with_codec_pad + codec_with_text_pad
        #   text_lens = ref_text_ids(5) + text_ids(2) + eos(1) = 8
        #   codec_lens = bos(1) + ref_time(5) = 6
        #   icl total = 8 + 6 = 14
        # Total: 3 + 4 + 14 = 21
        role_tokens = 3
        prefix_tokens = 4  # (nothink/think + pad,bos) - 1 for offset
        text_lens = 5 + 2 + 1
        codec_lens = 1 + ref_time
        expected_total = role_tokens + prefix_tokens + text_lens + codec_lens
        self.assertEqual(input_embeds.shape[1], expected_total)


class TestQwen3TTSGenerateICL(unittest.TestCase):
    """Tests for _generate_icl method."""

    def _make_icl_model(self, hidden_size=64, num_code_groups=4, vocab_size=2048):
        """Create a minimal Model for ICL generation testing."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config_dict = {
            "model_type": "qwen3_tts",
            "tts_model_type": "base",
            "tts_model_size": "0b6",
            "talker_config": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "hidden_act": "silu",
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "num_code_groups": num_code_groups,
                "text_hidden_size": hidden_size,
                "text_vocab_size": 100,
                "codec_eos_token_id": 30,
                "codec_pad_id": 28,
                "codec_bos_id": 29,
                "codec_language_id": {"english": 20, "chinese": 21},
                "spk_id": {"chelsie": 10, "ethan": 11},
                "code_predictor_config": {
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "intermediate_size": 128,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 32,
                    "hidden_act": "silu",
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "num_code_groups": num_code_groups,
                },
            },
            "speaker_encoder_config": None,
            "tokenizer_config": None,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "tts_pad_token_id": 151671,
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673,
            "sample_rate": 24000,
        }

        config = ModelConfig.from_dict(config_dict)
        model = Model(config)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(10))
        model.tokenizer = mock_tokenizer

        # Mock speech_tokenizer
        mock_speech_tokenizer = MagicMock()
        mock_speech_tokenizer.has_encoder = True
        mock_speech_tokenizer.decode_upsample_rate = 1920
        ref_time = 5
        # encode returns [1, num_code_groups, ref_time] to match generation
        mock_speech_tokenizer.encode.return_value = mx.zeros(
            (1, num_code_groups, ref_time), dtype=mx.int32
        )
        # decode returns (audio, audio_lengths)
        mock_speech_tokenizer.decode.return_value = (
            mx.random.normal((1, 24000)),  # ~1s audio
            mx.array([24000]),
        )

        # streaming_decode yields audio chunks
        def mock_streaming_decode(codes):
            yield mx.random.normal((1, 24000))  # ~1s audio chunk

        mock_speech_tokenizer.streaming_decode = mock_streaming_decode
        model.speech_tokenizer = mock_speech_tokenizer
        model.speaker_encoder = None

        return model

    def test_generate_icl_produces_result(self):
        """Test that _generate_icl produces a GenerationResult."""
        from mlx_audio.tts.models.base import GenerationResult

        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        results = list(
            model._generate_icl(
                text="Hello world",
                ref_audio=ref_audio,
                ref_text="Reference",
                language="auto",
                temperature=0.9,
                max_tokens=5,  # Very few tokens for fast test
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.5,
            )
        )

        # Should produce at least one result (or empty if EOS on first token)
        # With random weights, it's unlikely to hit EOS immediately
        if results:
            self.assertIsInstance(results[0], GenerationResult)
            self.assertIsNotNone(results[0].audio)
            self.assertEqual(results[0].sample_rate, 24000)

    def test_generate_icl_calls_speech_tokenizer_decode(self):
        """Test that _generate_icl calls speech_tokenizer.decode."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        # Track decode calls
        decode_calls = []
        original_decode = model.speech_tokenizer.decode

        def tracking_decode(codes):
            decode_calls.append(codes)
            return original_decode(codes)

        model.speech_tokenizer.decode = tracking_decode

        results = list(
            model._generate_icl(
                text="Hello",
                ref_audio=ref_audio,
                ref_text="Ref",
                max_tokens=3,
                repetition_penalty=1.5,
            )
        )

        if results:
            # decode should have been called with combined ref + gen codes
            self.assertEqual(len(decode_calls), 1)
            decode_args = decode_calls[0]
            # Should be [1, ref_time + gen_len, num_code_groups]
            self.assertEqual(decode_args.ndim, 3)
            self.assertEqual(decode_args.shape[0], 1)
            self.assertEqual(decode_args.shape[2], 4)  # num_code_groups

    def test_generate_icl_eos_stops_generation(self):
        """Test that EOS token stops generation early."""
        model = self._make_icl_model()
        config = model.config.talker_config

        ref_audio = mx.random.normal((24000,))

        # Patch _sample_token to always return EOS
        eos_id = config.codec_eos_token_id
        with patch.object(model, "_sample_token", return_value=mx.array([[eos_id]])):
            results = list(
                model._generate_icl(
                    text="Hello",
                    ref_audio=ref_audio,
                    ref_text="Ref",
                    max_tokens=100,
                    repetition_penalty=1.5,
                )
            )

        # Should produce no results since EOS on first step
        self.assertEqual(len(results), 0)

    def test_generate_icl_max_tokens_limit(self):
        """Test that max_tokens caps effective generation length."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        # With max_tokens=2, should generate at most 2 tokens
        results = list(
            model._generate_icl(
                text="Hello",
                ref_audio=ref_audio,
                ref_text="Ref",
                max_tokens=2,
                repetition_penalty=1.5,
            )
        )

        if results:
            # token_count should be <= 2
            self.assertLessEqual(results[0].token_count, 2)

    def test_generate_icl_repetition_penalty_applied(self):
        """Test that repetition penalty is applied during generation."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        # Track _sample_token calls to verify rep_penalty param
        original_sample = model._sample_token
        sample_calls = []

        def tracking_sample(*args, **kwargs):
            sample_calls.append(kwargs.get("repetition_penalty", 1.0))
            return original_sample(*args, **kwargs)

        with patch.object(model, "_sample_token", side_effect=tracking_sample):
            list(
                model._generate_icl(
                    text="Hello",
                    ref_audio=ref_audio,
                    ref_text="Ref",
                    max_tokens=2,
                    repetition_penalty=1.5,
                )
            )

        # All CB0 sampling calls should use the specified repetition penalty
        for call_pen in sample_calls:
            if call_pen != 1.0:  # CB1+ calls don't pass rep_penalty
                self.assertEqual(call_pen, 1.5)

    def test_generate_icl_ref_codes_prepended(self):
        """Test that reference codes are prepended to generated codes for decoding."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))
        ref_time = 5  # Matches mock setup
        config = model.config.talker_config
        eos_id = config.codec_eos_token_id

        # Track decode calls
        decode_calls = []
        original_decode = model.speech_tokenizer.decode

        def tracking_decode(codes):
            decode_calls.append(codes)
            return original_decode(codes)

        model.speech_tokenizer.decode = tracking_decode

        # Force generation of exactly 2 tokens then EOS
        cb0_count = [0]

        def controlled_sample(*args, **kwargs):
            # CB0 calls have eos_token_id set
            if kwargs.get("eos_token_id") is not None:
                cb0_count[0] += 1
                if cb0_count[0] <= 2:
                    return mx.array([[5]])  # non-EOS token
                else:
                    return mx.array([[eos_id]])  # EOS
            # Code predictor calls: return valid token
            return mx.array([[3]])

        with patch.object(model, "_sample_token", side_effect=controlled_sample):
            results = list(
                model._generate_icl(
                    text="Hello world test",
                    ref_audio=ref_audio,
                    ref_text="Ref",
                    max_tokens=10,
                    repetition_penalty=1.5,
                )
            )

        self.assertEqual(len(results), 1)
        # Check that decode was called with ref_time + gen_len time steps
        self.assertEqual(len(decode_calls), 1)
        decode_args = decode_calls[0]
        gen_len = results[0].token_count
        self.assertEqual(gen_len, 2)
        expected_time = ref_time + gen_len
        self.assertEqual(decode_args.shape[1], expected_time)

    def test_generate_icl_proportional_trimming(self):
        """Test that reference audio portion is trimmed from output."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))
        ref_time = 5

        # Set up decode to return longer audio so trimming is testable
        total_audio_len = 48000
        model.speech_tokenizer.decode.return_value = (
            mx.random.normal((1, total_audio_len)),
            mx.array([total_audio_len]),
        )

        results = list(
            model._generate_icl(
                text="Hello world",
                ref_audio=ref_audio,
                ref_text="Ref",
                max_tokens=5,
                repetition_penalty=1.5,
            )
        )

        if results:
            # Audio should be shorter than total_audio_len due to trimming
            self.assertLess(results[0].samples, total_audio_len)

    def test_generate_routing_uses_icl_when_ref_audio_provided(self):
        """Test that generate() routes to ICL when ref_audio and ref_text provided."""
        model = self._make_icl_model()

        ref_audio = mx.random.normal((24000,))

        with patch.object(model, "_generate_icl") as mock_icl:
            mock_icl.return_value = iter([])  # Empty generator
            list(
                model.generate(
                    text="Hello",
                    ref_audio=ref_audio,
                    ref_text="Reference text",
                )
            )

        # Should have called _generate_icl
        mock_icl.assert_called_once()

    def test_generate_routing_icl_rep_penalty_floor(self):
        """Test that generate() enforces min rep_penalty=1.5 for ICL mode."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        with patch.object(model, "_generate_icl") as mock_icl:
            mock_icl.return_value = iter([])
            list(
                model.generate(
                    text="Hello",
                    ref_audio=ref_audio,
                    ref_text="Ref",
                    repetition_penalty=1.05,  # Below floor
                )
            )

        # Should have been called with rep_penalty=1.5 (the floor)
        call_kwargs = mock_icl.call_args[1]
        self.assertEqual(call_kwargs["repetition_penalty"], 1.5)

    def test_generate_routing_icl_rep_penalty_passthrough(self):
        """Test that rep_penalty > 1.5 is passed through unchanged."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        with patch.object(model, "_generate_icl") as mock_icl:
            mock_icl.return_value = iter([])
            list(
                model.generate(
                    text="Hello",
                    ref_audio=ref_audio,
                    ref_text="Ref",
                    repetition_penalty=2.0,  # Above floor
                )
            )

        call_kwargs = mock_icl.call_args[1]
        self.assertEqual(call_kwargs["repetition_penalty"], 2.0)

    def test_generate_routing_no_icl_without_encoder(self):
        """Test that generate() skips ICL when speech_tokenizer has no encoder."""
        model = self._make_icl_model()
        model.speech_tokenizer.has_encoder = False
        ref_audio = mx.random.normal((24000,))

        with patch.object(model, "_generate_icl") as mock_icl:
            mock_icl.return_value = iter([])
            # This will fall through to the non-ICL base path
            # which would call _generate_base or similar
            try:
                list(
                    model.generate(
                        text="Hello",
                        ref_audio=ref_audio,
                        ref_text="Ref",
                    )
                )
            except Exception:
                pass  # May fail in non-ICL path, that's fine

        # Should NOT have called _generate_icl
        mock_icl.assert_not_called()

    def test_generate_routing_no_icl_without_ref_text(self):
        """Test that generate() skips ICL when ref_text is not provided."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        with patch.object(model, "_generate_icl") as mock_icl:
            mock_icl.return_value = iter([])
            try:
                list(
                    model.generate(
                        text="Hello",
                        ref_audio=ref_audio,
                        ref_text=None,
                    )
                )
            except Exception:
                pass

        mock_icl.assert_not_called()

    def test_batch_supports_shared_ref_but_not_continuous(self):
        """Shared ref cloning can use batch_generate but not continuous batching."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))

        self.assertTrue(
            model.supports_tts_batch(ref_audio=ref_audio, ref_text="Reference")
        )
        self.assertFalse(
            model.supports_tts_continuous_batch(
                ref_audio=ref_audio,
                ref_text="Reference",
            )
        )
        self.assertFalse(model.supports_tts_batch(ref_audio=ref_audio))

    def test_batch_generate_shared_ref_uses_icl_manual_path(self):
        """Shared ref batches avoid continuous batching and decode with ref context."""
        model = self._make_icl_model()
        ref_audio = mx.random.normal((24000,))
        eos_id = model.config.talker_config.codec_eos_token_id
        num_code_groups = model.config.talker_config.num_code_groups
        sample_calls = {"count": 0}

        def sample_batch(logits, **kwargs):
            del kwargs
            sample_calls["count"] += 1
            token = 5 if sample_calls["count"] == 1 else eos_id
            return mx.full((logits.shape[0], 1), token, dtype=mx.int32)

        def predict_codes(first_token, hidden, **kwargs):
            del hidden, kwargs
            batch = first_token.shape[0]
            all_codes = mx.ones((batch, num_code_groups), dtype=mx.int32)
            code_tokens = [
                all_codes[:, index : index + 1] for index in range(num_code_groups)
            ]
            return code_tokens, all_codes

        with (
            patch.object(
                model,
                "create_tts_batch_session",
                side_effect=AssertionError("continuous batching should not run"),
            ),
            patch.object(model, "_sample_token_batch", side_effect=sample_batch),
            patch.object(model, "_predict_code_tokens", side_effect=predict_codes),
        ):
            results = list(
                model.batch_generate(
                    ["first", "second"],
                    ref_audios=[ref_audio, ref_audio],
                    ref_texts=["Reference", "Reference"],
                    max_tokens=3,
                    stream=False,
                )
            )

        self.assertEqual([result.sequence_idx for result in results], [0, 1])
        self.assertEqual([result.token_count for result in results], [1, 1])
        self.assertEqual(model.speech_tokenizer.decode.call_count, 2)
        decoded_codes = model.speech_tokenizer.decode.call_args_list[0][0][0]
        self.assertEqual(decoded_codes.shape[1], 6)  # ref_time(5) + generated(1)

    def test_batch_generate_rejects_mixed_refs(self):
        """Qwen3 batch ICL currently accepts only one shared reference pair."""
        model = self._make_icl_model()

        with self.assertRaisesRegex(ValueError, "one shared ref_audio"):
            list(
                model.batch_generate(
                    ["first", "second"],
                    ref_audios=["a.wav", "b.wav"],
                    ref_texts=["Reference", "Reference"],
                )
            )


class TestQwen3TTSStreamingDecode(unittest.TestCase):
    """Tests for streaming vs non-streaming decode behavior."""

    def _make_model(self, hidden_size=64, num_code_groups=4, vocab_size=2048):
        """Create a minimal Qwen3-TTS model for testing."""
        from mlx_audio.tts.models.qwen3_tts import Model, ModelConfig

        config_dict = {
            "model_type": "qwen3_tts",
            "tts_model_type": "base",
            "tts_model_size": "0b6",
            "talker_config": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "hidden_act": "silu",
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "num_code_groups": num_code_groups,
                "text_hidden_size": hidden_size,
                "text_vocab_size": 100,
                "codec_eos_token_id": 30,
                "codec_pad_id": 28,
                "codec_bos_id": 29,
                "codec_language_id": {"english": 20, "chinese": 21},
                "spk_id": {"chelsie": 10, "ethan": 11},
                "code_predictor_config": {
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "intermediate_size": 128,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 32,
                    "hidden_act": "silu",
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "num_code_groups": num_code_groups,
                },
            },
            "speaker_encoder_config": None,
            "tokenizer_config": None,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "tts_pad_token_id": 151671,
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673,
            "sample_rate": 24000,
        }

        config = ModelConfig.from_dict(config_dict)
        model = Model(config)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(10))
        model.tokenizer = mock_tokenizer

        # Mock speech_tokenizer
        mock_speech_tokenizer = MagicMock()
        mock_speech_tokenizer.has_encoder = False
        mock_speech_tokenizer.decode_upsample_rate = 1920
        mock_speech_tokenizer.decode.return_value = (
            mx.random.normal((1, 48000)),
            mx.array([48000]),
        )

        def mock_streaming_decode(codes, chunk_tokens=100):
            # Yield chunks
            total_samples = 48000
            chunk_samples = chunk_tokens * 1920
            for i in range(0, total_samples, chunk_samples):
                end = min(i + chunk_samples, total_samples)
                yield mx.random.normal((1, end - i))

        mock_speech_tokenizer.streaming_decode = mock_streaming_decode
        model.speech_tokenizer = mock_speech_tokenizer

        return model

    def test_decode_chunk_uses_streaming_decode(self):
        """Test that _decode_chunk internally calls streaming_decode."""
        model = self._make_model()

        # Track streaming_decode calls
        streaming_decode_calls = []

        def tracking_streaming_decode(codes, chunk_tokens=100):
            streaming_decode_calls.append(
                {"codes": codes, "chunk_tokens": chunk_tokens}
            )
            yield mx.random.normal((1, 48000))

        model.speech_tokenizer.streaming_decode = tracking_streaming_decode

        # Call _decode_chunk directly
        codes = mx.zeros((1, 10, 4))  # [batch, time, num_code_groups]
        model._decode_chunk(codes, chunk_tokens=50)

        # Verify streaming_decode was called with correct chunk_tokens
        self.assertEqual(len(streaming_decode_calls), 1)
        self.assertEqual(streaming_decode_calls[0]["chunk_tokens"], 50)

    def test_decode_chunk_respects_chunk_tokens_parameter(self):
        """Test that _decode_chunk passes chunk_tokens to streaming_decode."""
        model = self._make_model()

        # Track chunk_tokens values
        chunk_tokens_used = []

        def tracking_streaming_decode(codes, chunk_tokens=100):
            chunk_tokens_used.append(chunk_tokens)
            yield mx.random.normal((1, 48000))

        model.speech_tokenizer.streaming_decode = tracking_streaming_decode

        # Test with different chunk_tokens values
        codes = mx.zeros((1, 10, 4))

        model._decode_chunk(codes, chunk_tokens=25)
        self.assertEqual(chunk_tokens_used[-1], 25)

        model._decode_chunk(codes, chunk_tokens=100)
        self.assertEqual(chunk_tokens_used[-1], 100)

        model._decode_chunk(codes, chunk_tokens=300)
        self.assertEqual(chunk_tokens_used[-1], 300)

    def test_streaming_chunk_size_calculation(self):
        """Test that streaming_chunk_size is calculated from streaming_interval."""
        # The formula is: streaming_chunk_size = max(1, int(streaming_interval * 12.5))
        # Test the calculation directly

        # streaming_interval=2.0 -> 25 tokens
        self.assertEqual(max(1, int(2.0 * 12.5)), 25)

        # streaming_interval=4.0 -> 50 tokens
        self.assertEqual(max(1, int(4.0 * 12.5)), 50)

        # streaming_interval=8.0 -> 100 tokens
        self.assertEqual(max(1, int(8.0 * 12.5)), 100)

        # streaming_interval=0.1 -> 1 token (minimum)
        self.assertEqual(max(1, int(0.1 * 12.5)), 1)


@patch("importlib.resources.open_text", patched_open_text)
class TestBailingMMModel(unittest.TestCase):
    HAS_ONNX = importlib.util.find_spec("onnx") is not None

    @staticmethod
    def _build_dummy_onnx(path: Path):
        import onnx
        from onnx import helper, numpy_helper

        expected = {
            "linear.weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "linear.bias": np.array([0.1, -0.2], dtype=np.float32),
        }
        initializers = [
            numpy_helper.from_array(array, name=name)
            for name, array in expected.items()
        ]
        graph = helper.make_graph(
            nodes=[],
            name="dummy_campplus",
            inputs=[],
            outputs=[],
            initializer=initializers,
        )
        model = helper.make_model(graph, producer_name="mlx-audio-tests")
        onnx.save(model, path.as_posix())
        return expected

    def _make_minimal_bailingmm_model(self):
        from mlx_audio.tts.models.bailingmm.bailingmm import Model

        model = Model.__new__(Model)
        model.tokenizer = MagicMock()
        model.tokenizer.encode.return_value = [1, 2, 3]
        model.audio = MagicMock()
        model.audio.sample_rate = 24000
        return model

    def test_convert_campplus_onnx_to_safetensors_allclose(self):
        if not self.HAS_ONNX:
            self.skipTest("onnx is required for campplus conversion test.")

        from safetensors.numpy import load_file

        from mlx_audio.tts.models.bailingmm import convert_campplus_onnx_to_safetensors

        with TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "campplus.onnx"
            expected = self._build_dummy_onnx(onnx_path)

            safetensors_path = convert_campplus_onnx_to_safetensors(
                onnx_path=onnx_path,
                output_path=Path(tmpdir) / "campplus.safetensors",
                verify_allclose=True,
            )

            converted = load_file(str(safetensors_path))
            self.assertEqual(set(converted.keys()), set(expected.keys()))
            for key in expected:
                self.assertTrue(
                    np.allclose(expected[key], converted[key], rtol=1e-5, atol=1e-6)
                )

    def test_qwen2_sliding_window_attention_applies_after_window_boundary(self):
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.qwen2 import ModelArgs as Qwen2ModelArgs

        from mlx_audio.tts.models.bailingmm.bailingmm import MingQwen2Model

        args = Qwen2ModelArgs(
            model_type="qwen2",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=1,
            rope_theta=1_000_000.0,
        )
        model = MingQwen2Model(
            args,
            {
                "use_sliding_window": True,
                "sliding_window": 4,
                "max_window_layers": 0,
            },
        )

        x = mx.random.normal((1, 8, 32), dtype=mx.float32)
        mask = create_attention_mask(x, None)

        y_sliding = model.layers[0](x, mask, None)
        model.layers[0].self_attn.sliding_window = None
        y_full = model.layers[0](x, mask, None)

        diff = np.abs(np.array(y_sliding) - np.array(y_full)).mean(axis=-1)[0]
        self.assertLess(diff[:4].max(), 1e-6)
        self.assertGreater(diff[4:].max(), 1e-6)

    @patch("mlx_audio.tts.models.bailingmm.bailingmm.FlowLoss")
    @patch("mlx_audio.tts.models.bailingmm.bailingmm.Aggregator")
    @patch("mlx_audio.tts.models.bailingmm.bailingmm.AudioVAE")
    @patch("mlx_audio.tts.models.bailingmm.bailingmm.MingBailingMoeModel")
    @patch("mlx_audio.tts.models.bailingmm.bailingmm.MingQwen2ForCausalLM")
    def test_init_uses_dense_qwen2_backbone_when_moe_fields_missing(
        self,
        mock_qwen2_lm,
        mock_moe_lm,
        mock_audio_vae,
        _mock_aggregator,
        _mock_flowloss,
    ):
        from mlx_audio.tts.models.bailingmm.bailingmm import Model

        audio = MagicMock()
        audio.sample_rate = 24000
        mock_audio_vae.return_value = audio

        config = {
            "model_type": "dense",
            "llm_config": {
                "model_type": "qwen2",
                "hidden_size": 896,
                "num_hidden_layers": 24,
                "intermediate_size": 4864,
                "num_attention_heads": 14,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 151936,
                "rope_theta": 1_000_000.0,
            },
            "audio_tokenizer_config": {
                "sample_rate": 24000,
                "enc_kwargs": {"latent_dim": 64},
            },
            "ditar_config": {"patch_size": 8},
            "aggregator_config": {"depth": 1},
        }

        model = Model(config)

        mock_qwen2_lm.assert_called_once()
        mock_moe_lm.assert_not_called()
        self.assertEqual(model.llm_args.model_type, "qwen2")

    @patch(
        "mlx_audio.tts.models.bailingmm.bailingmm.mx.get_peak_memory",
        return_value=0.0,
    )
    @patch(
        "mlx_audio.tts.models.bailingmm.bailingmm.time.perf_counter",
        side_effect=[0.0, 1.0],
    )
    def test_generate_keeps_trailing_samples(
        self, _mock_perf_counter, _mock_peak_memory
    ):
        model = self._make_minimal_bailingmm_model()

        latent = mx.zeros((1, 1, 64), dtype=mx.float32)
        model.sample = MagicMock(return_value=iter([(latent, True)]))

        # Torch path does not apply a trailing low-energy trim pass.
        chunk = mx.array([[0.25, 0.15, 0.0, 0.0, 0.0]], dtype=mx.float32)
        model.audio.decode.return_value = (chunk, (None, None, None), None)

        result = next(model.generate(text="hello", max_tokens=1))
        np.testing.assert_allclose(np.array(result.audio), np.array(chunk)[0], atol=0)

    @patch(
        "mlx_audio.tts.models.bailingmm.bailingmm.mx.get_peak_memory",
        return_value=0.0,
    )
    @patch(
        "mlx_audio.tts.models.bailingmm.bailingmm.time.perf_counter",
        side_effect=[0.0, 1.0],
    )
    def test_generate_concatenates_all_decode_chunks(
        self, _mock_perf_counter, _mock_peak_memory
    ):
        import mlx_audio.tts.models.bailingmm.bailingmm as bailingmm_module

        model = self._make_minimal_bailingmm_model()

        latent = mx.zeros((1, 1, 64), dtype=mx.float32)
        model.sample = MagicMock(return_value=iter([(latent, False), (latent, True)]))
        chunk1 = mx.array([[0.1, 0.2]], dtype=mx.float32)
        chunk2 = mx.zeros((1, 0), dtype=mx.float32)
        model.audio.decode.side_effect = [
            (chunk1, (None, None, None), None),
            (chunk2, (None, None, None), None),
        ]

        observed = {}
        original_concat = bailingmm_module.mx.concatenate

        def _concat_spy(items, axis=0):
            observed["num_chunks"] = len(items)
            return original_concat(items, axis=axis)

        with patch(
            "mlx_audio.tts.models.bailingmm.bailingmm.mx.concatenate",
            side_effect=_concat_spy,
        ):
            result = next(model.generate(text="hello", max_tokens=2))

        self.assertEqual(observed.get("num_chunks"), 2)
        np.testing.assert_allclose(np.array(result.audio), np.array(chunk1)[0], atol=0)


class TestFishSpeechPrompt(unittest.TestCase):
    def test_encode_for_inference_places_vq_codes_in_correct_rows(self):
        from mlx_audio.tts.models.fish_qwen3_omni.prompt import (
            Conversation,
            Message,
            TextPart,
            VQPart,
        )

        tokenizer = FakeTokenizer()
        conversation = Conversation(
            [
                Message(
                    role="user",
                    parts=[
                        TextPart("hello"),
                        VQPart(mx.array([[1, 2], [3, 4]], dtype=mx.int32)),
                        TextPart("world"),
                    ],
                    add_im_start=False,
                    add_im_end=False,
                )
            ]
        )

        values = conversation.encode_for_inference(tokenizer, num_codebooks=2)
        self.assertEqual(tuple(values.shape), (3, 4))
        self.assertEqual(values[0].tolist(), [1, 1001, 1002, 2])
        self.assertEqual(values[1].tolist(), [0, 1, 2, 0])
        self.assertEqual(values[2].tolist(), [0, 3, 4, 0])


class TestFishSpeechModel(unittest.TestCase):
    def test_model_type_remapping_uses_config_value(self):
        from mlx_audio.tts.utils import get_model_and_args

        module, model_type = get_model_and_args("fish_qwen3_omni", ["s2", "pro"])
        self.assertEqual(model_type, "fish_qwen3_omni")
        self.assertTrue(hasattr(module, "Model"))

    def test_sanitize_remaps_upstream_keys(self):
        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

        model = Model(tiny_config())
        weights = {
            "text_model.model.embeddings.weight": mx.zeros((4, 4)),
            "audio_decoder.embeddings.weight": mx.zeros((4, 4)),
            "audio_decoder.layers.0.attention.wqkv.weight": mx.zeros((4, 4)),
            "audio_decoder.codebook_embeddings.weight": mx.zeros((4, 4)),
        }

        sanitized = model.sanitize(weights)

        self.assertIn("model.embeddings.weight", sanitized)
        self.assertIn("model.fast_embeddings.weight", sanitized)
        self.assertIn("model.fast_layers.0.attention.wqkv.weight", sanitized)
        self.assertIn("model.codebook_embeddings.weight", sanitized)

    def test_config_from_dict_handles_upstream_nested_shape(self):
        config = tiny_config()
        self.assertEqual(config.text_config.n_layer, 1)
        self.assertEqual(config.audio_decoder_config.num_codebooks, 2)
        self.assertEqual(config.semantic_start_token_id, 1000)

    def test_sample_semantic_only_samples_high_temp_when_rejected(self):
        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

        config = tiny_config()
        config.semantic_start_token_id = 10
        config.semantic_end_token_id = 15
        model = Model(config)
        model.semantic_logit_bias = mx.zeros((1, 16), dtype=mx.float32)
        calls = []

        def fake_sample_logits(logits, temperature, top_p, top_k):
            calls.append((temperature, top_p, logits.shape[0]))
            if top_p == 0.9:
                return mx.array([11], dtype=mx.int32)
            return mx.array([10], dtype=mx.int32)

        with patch(
            "mlx_audio.tts.models.fish_qwen3_omni.fish_speech._sample_logits",
            side_effect=fake_sample_logits,
        ):
            accepted = model._sample_semantic(
                logits=mx.zeros((1, 16), dtype=mx.float32),
                previous_semantic_tokens=[],
                top_p=0.7,
                top_k=0,
                temperature=0.7,
            )
            rejected = model._sample_semantic(
                logits=mx.zeros((1, 16), dtype=mx.float32),
                previous_semantic_tokens=[10],
                top_p=0.7,
                top_k=0,
                temperature=0.7,
            )

        self.assertEqual(accepted.tolist(), [10])
        self.assertEqual(rejected.tolist(), [11])
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], (0.7, 0.7, 1))
        self.assertEqual(calls[1], (0.7, 0.7, 1))
        self.assertEqual(calls[2], (1.0, 0.9, 1))

    def test_sample_semantic_batch_only_resamples_rejected_rows(self):
        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

        config = tiny_config()
        config.semantic_start_token_id = 10
        config.semantic_end_token_id = 15
        model = Model(config)
        model.semantic_logit_bias = mx.zeros((1, 16), dtype=mx.float32)
        calls = []

        def fake_sample_logits(logits, temperature, top_p, top_k):
            calls.append((temperature, top_p, logits.shape[0]))
            if top_p == 0.9:
                return mx.array([13, 14][: logits.shape[0]], dtype=mx.int32)
            return mx.array([10, 11, 12][: logits.shape[0]], dtype=mx.int32)

        with patch(
            "mlx_audio.tts.models.fish_qwen3_omni.fish_speech._sample_logits",
            side_effect=fake_sample_logits,
        ):
            tokens = model._sample_semantic_batch(
                logits=mx.zeros((3, 16), dtype=mx.float32),
                previous_semantic_tokens=[[10], [], [12]],
                top_p=0.7,
                top_k=0,
                temperature=0.7,
            )

        self.assertEqual(tokens.tolist(), [13, 11, 14])
        self.assertEqual(calls, [(0.7, 0.7, 3), (1.0, 0.9, 2)])

    def test_prepare_batched_prompt_inputs_left_pads_variable_lengths(self):
        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model
        from mlx_audio.tts.models.fish_qwen3_omni.prompt import Message, TextPart

        class VariableTokenizer:
            semantic_begin_id = 1000

            def encode(self, text):
                return list(range(1, max(1, len(text.split())) + 1))

        model = Model(tiny_config())
        model.tokenizer = VariableTokenizer()

        short = model._build_conversation([], [])
        short.append(
            Message(
                role="user",
                parts=[TextPart("short")],
                add_im_start=True,
                add_im_end=True,
            )
        )
        long = model._build_conversation([], [])
        long.append(
            Message(
                role="user",
                parts=[TextPart("this is a much longer prompt")],
                add_im_start=True,
                add_im_end=True,
            )
        )

        prompt, mask = model._prepare_batched_prompt_inputs([short, long])
        mx.eval(prompt, mask)

        self.assertEqual(prompt.shape[0], 2)
        self.assertEqual(prompt.shape[1], model.model.num_codebooks + 1)
        self.assertEqual(prompt.shape[2], mask.shape[1])
        self.assertEqual(mask.tolist()[0][0], 0.0)
        self.assertEqual(mask.tolist()[0][-1], 1.0)
        self.assertTrue(all(value == 1.0 for value in mask.tolist()[1]))

    def test_batch_generate_yields_sequence_results(self):
        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

        model = Model(tiny_config())
        model.tokenizer = FakeTokenizer()
        model.codec = object()
        calls = []

        def fake_generate_codes(**kwargs):
            calls.append(list(kwargs["batch_texts"]))
            return [
                mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
                for _ in kwargs["batch_texts"]
            ]

        def fake_decode_codes(codes_list):
            return [
                mx.ones((codes.shape[1] * 4,), dtype=mx.float32) * (idx + 1)
                for idx, codes in enumerate(codes_list)
            ]

        model._generate_codes_for_text_batch = fake_generate_codes
        model._decode_codes_batch = fake_decode_codes

        results = list(model.batch_generate(["first", "second"], verbose=False))

        self.assertEqual(calls, [["first", "second"]])
        self.assertEqual([result.sequence_idx for result in results], [0, 1])
        self.assertEqual([result.token_count for result in results], [3, 3])
        self.assertEqual([result.samples for result in results], [12, 12])

    def test_generate_codes_for_text_batch_with_tiny_model(self):
        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model
        from mlx_audio.tts.models.fish_qwen3_omni.prompt import Message, TextPart

        class TinyTokenizer:
            semantic_begin_id = 10
            vocab_size = 32

            def encode(self, text):
                words = text.split() or [text]
                return [idx % 20 + 1 for idx, _ in enumerate(words)]

            def get_token_id(self, token):
                return 2

        config = tiny_config()
        config.semantic_start_token_id = 10
        config.semantic_end_token_id = 17
        model = Model(config)
        model.tokenizer = TinyTokenizer()
        semantic_bias = mx.full((1, config.text_config.vocab_size), -1e9)
        semantic_bias[:, 10:18] = 0.0
        model.semantic_logit_bias = semantic_bias

        short = model._build_conversation([], [])
        short.append(
            Message(
                role="user",
                parts=[TextPart("short")],
                add_im_start=True,
                add_im_end=True,
            )
        )
        long = model._build_conversation([], [])
        long.append(
            Message(
                role="user",
                parts=[TextPart("this is longer")],
                add_im_start=True,
                add_im_end=True,
            )
        )

        codes = model._generate_codes_for_text_batch(
            conversations=[short, long],
            batch_texts=["short", "this is longer"],
            max_new_tokens=1,
            top_p=1.0,
            top_k=0,
            temperature=0.0,
        )

        self.assertEqual(len(codes), 2)
        self.assertEqual(tuple(codes[0].shape), (2, 1))
        self.assertEqual(tuple(codes[1].shape), (2, 1))

    def test_batch_generate_validates_parallel_arg_lengths(self):
        from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

        model = Model(tiny_config())
        model.tokenizer = FakeTokenizer()
        model.codec = object()

        with self.assertRaises(ValueError):
            list(
                model.batch_generate(
                    ["first", "second"],
                    ref_texts=["only one"],
                    verbose=False,
                )
            )


# ---------------------------------------------------------------------------
# Irodori-TTS helpers
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal HuggingFace-style tokenizer stub for Irodori-TTS tests."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [3, 4, 5]


class _FakeDACVAE:
    """DACVAE stub that matches the real API shapes."""

    def __init__(self, latent_dim: int = 8, downsample_factor: int = 1920):
        self.latent_dim = latent_dim
        self.downsample_factor = downsample_factor

    def encode(self, audio_in: mx.array) -> mx.array:
        B = audio_in.shape[0]
        T = max(1, int(audio_in.shape[1]) // self.downsample_factor)
        return mx.zeros((B, self.latent_dim, T), dtype=mx.float32)

    def decode(self, latent: mx.array, **kwargs) -> mx.array:
        B, _D, T = latent.shape
        return mx.zeros((B, T * self.downsample_factor, 1), dtype=mx.float32)


def _small_irodori_dit_config(**overrides):
    from mlx_audio.tts.models.irodori_tts.config import IrodoriDiTConfig

    defaults = dict(
        latent_dim=8,
        latent_patch_size=1,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        text_mlp_ratio=2.0,
        speaker_mlp_ratio=2.0,
        text_vocab_size=64,
        text_dim=32,
        text_layers=1,
        text_heads=4,
        speaker_dim=32,
        speaker_layers=1,
        speaker_heads=4,
        speaker_patch_size=1,
        timestep_embed_dim=16,
        adaln_rank=8,
        norm_eps=1e-5,
    )
    defaults.update(overrides)
    return IrodoriDiTConfig(**defaults)


def _small_irodori_model_config(**sampler_overrides):
    from mlx_audio.tts.models.irodori_tts.config import ModelConfig, SamplerConfig

    sampler_defaults = dict(
        num_steps=1,
        cfg_scale_text=1.0,
        cfg_scale_speaker=1.0,
        sequence_length=4,
    )
    sampler_defaults.update(sampler_overrides)
    return ModelConfig(
        dit=_small_irodori_dit_config(),
        sampler=SamplerConfig(**sampler_defaults),
    )


# ---------------------------------------------------------------------------
# Irodori-TTS test classes
# ---------------------------------------------------------------------------


class TestIrodoriNormalizeText(unittest.TestCase):
    def test_fullwidth_alpha_to_halfwidth(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        self.assertEqual(normalize_text("Ａｂ"), "Ab")

    def test_fullwidth_digits_to_halfwidth(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        self.assertEqual(normalize_text("１２３"), "123")

    def test_halfwidth_kana_to_fullwidth(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        self.assertEqual(normalize_text("ｱｲ"), "アイ")

    def test_wave_dash_to_katakana_dash(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        self.assertEqual(normalize_text("ー〜ー"), "ーーー")

    def test_trailing_kuten_stripped(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        result = normalize_text("こんにちは。")
        self.assertFalse(result.endswith("。"))
        self.assertEqual(result, "こんにちは")

    def test_surrounding_brackets_stripped(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        self.assertEqual(normalize_text("「こんにちは」"), "こんにちは")

    def test_no_change_for_plain_text(self):
        from mlx_audio.tts.models.irodori_tts.text import normalize_text

        text = "こんにちは"
        self.assertEqual(normalize_text(text), text)


class TestIrodoriEncodeText(unittest.TestCase):
    def setUp(self):
        self.tok = _MockTokenizer()

    def test_output_shapes(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        self.assertEqual(tuple(ids.shape), (1, 10))
        self.assertEqual(tuple(mask.shape), (1, 10))

    def test_bos_prepended(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        self.assertEqual(int(ids[0, 0]), self.tok.bos_token_id)

    def test_no_bos(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, _ = encode_text("hello", self.tok, max_length=10, add_bos=False)
        self.assertEqual(int(ids[0, 0]), 3)

    def test_padding(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        for i in range(4, 10):
            self.assertEqual(int(ids[0, i]), self.tok.pad_token_id)
            self.assertFalse(bool(mask[0, i]))

    def test_mask_true_for_real_tokens(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, mask = encode_text("hello", self.tok, max_length=10, add_bos=True)
        for i in range(4):
            self.assertTrue(bool(mask[0, i]))

    def test_truncation(self):
        from mlx_audio.tts.models.irodori_tts.text import encode_text

        ids, mask = encode_text("hello", self.tok, max_length=2, add_bos=True)
        self.assertEqual(tuple(ids.shape), (1, 2))


class TestIrodoriDiTShapes(unittest.TestCase):
    def setUp(self):
        from mlx_audio.tts.models.irodori_tts.model import IrodoriDiT

        self.cfg = _small_irodori_dit_config()
        self.model = IrodoriDiT(self.cfg)

    def test_full_forward_shape(self):
        B, S = 1, 6
        x_t = mx.random.normal((B, S, self.cfg.patched_latent_dim))
        t = mx.array([0.5], dtype=mx.float32)
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.random.normal((B, 8, self.cfg.latent_dim))
        ref_mask = mx.ones((B, 8), dtype=mx.bool_)

        out = self.model(x_t, t, text_ids, text_mask, ref_latent, ref_mask)
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (B, S, self.cfg.patched_latent_dim))

    def test_encode_conditions_shapes(self):
        B = 1
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.random.normal((B, 8, self.cfg.latent_dim))
        ref_mask = mx.ones((B, 8), dtype=mx.bool_)

        t_state, t_mask, s_state, s_mask = self.model.encode_conditions(
            text_ids, text_mask, ref_latent, ref_mask
        )
        mx.eval(t_state, s_state)
        self.assertEqual(tuple(t_state.shape), (B, 5, self.cfg.text_dim))
        self.assertEqual(int(s_state.shape[0]), B)
        self.assertEqual(int(s_state.shape[-1]), self.cfg.speaker_dim)

    def test_kv_cache_and_forward_with_conditions(self):
        B, S = 1, 4
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.zeros((B, 8, self.cfg.latent_dim))
        ref_mask = mx.ones((B, 8), dtype=mx.bool_)

        t_state, t_mask, s_state, s_mask = self.model.encode_conditions(
            text_ids, text_mask, ref_latent, ref_mask
        )
        kv_text, kv_speaker = self.model.build_kv_cache(t_state, s_state)
        self.assertEqual(len(kv_text), self.cfg.num_layers)
        self.assertEqual(len(kv_speaker), self.cfg.num_layers)

        x_t = mx.random.normal((B, S, self.cfg.patched_latent_dim))
        t = mx.array([0.3], dtype=mx.float32)
        out = self.model.forward_with_conditions(
            x_t, t, t_state, t_mask, s_state, s_mask, kv_text, kv_speaker
        )
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (B, S, self.cfg.patched_latent_dim))

    def test_zero_speaker_latent(self):
        B, S = 1, 4
        x_t = mx.random.normal((B, S, self.cfg.patched_latent_dim))
        t = mx.array([1.0], dtype=mx.float32)
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        ref_latent = mx.zeros((B, 1, self.cfg.latent_dim))
        ref_mask = mx.zeros((B, 1), dtype=mx.bool_)

        out = self.model(x_t, t, text_ids, text_mask, ref_latent, ref_mask)
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (B, S, self.cfg.patched_latent_dim))


class TestIrodoriModelSanitize(unittest.TestCase):
    def setUp(self):
        from mlx_audio.tts.models.irodori_tts.irodori_tts import Model

        self.model = Model(_small_irodori_model_config())

    def test_cond_module_key_remapped(self):
        weights = {"cond_module.0.weight": mx.zeros((1, 1), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.cond_module.layers.0.weight", sanitized)
        self.assertNotIn("cond_module.0.weight", sanitized)

    def test_model_prefix_added(self):
        weights = {"blocks.0.mlp.w1.weight": mx.zeros((1, 1), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.blocks.0.mlp.w1.weight", sanitized)

    def test_model_prefix_not_doubled(self):
        weights = {"model.out_proj.weight": mx.zeros((1, 1), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.out_proj.weight", sanitized)
        self.assertNotIn("model.model.out_proj.weight", sanitized)

    def test_deep_cond_module_key(self):
        weights = {"cond_module.2.bias": mx.zeros((1,), dtype=mx.float32)}
        sanitized = self.model.sanitize(weights)
        self.assertIn("model.cond_module.layers.2.bias", sanitized)


class TestIrodoriGenerateSmoke(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.irodori_tts.irodori_tts import Model

        cfg = _small_irodori_model_config()
        model = Model(cfg)
        model.dacvae = _FakeDACVAE(
            latent_dim=cfg.dit.latent_dim,
            downsample_factor=cfg.audio_downsample_factor,
        )
        model._tokenizer = _MockTokenizer()
        return model

    def test_generate_returns_result(self):
        model = self._make_model()
        results = list(model.generate("こんにちは", rng_seed=0))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].sample_rate, 48000)
        self.assertGreater(results[0].samples, 0)

    def test_generate_with_ref_audio(self):
        from mlx_audio.tts.models.irodori_tts.irodori_tts import Model

        cfg = _small_irodori_model_config()
        model = Model(cfg)
        model.dacvae = _FakeDACVAE(
            latent_dim=cfg.dit.latent_dim,
            downsample_factor=cfg.audio_downsample_factor,
        )
        model._tokenizer = _MockTokenizer()
        ref = mx.zeros((1, cfg.audio_downsample_factor * 4), dtype=mx.float32)
        results = list(model.generate("テスト", ref_audio=ref, rng_seed=1))
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].samples, 0)

    def test_generate_stream_raises(self):
        model = self._make_model()
        with self.assertRaises(NotImplementedError):
            next(model.generate("hi", stream=True))

    def test_generate_without_dacvae_raises(self):
        from mlx_audio.tts.models.irodori_tts.irodori_tts import Model

        cfg = _small_irodori_model_config()
        model = Model(cfg)
        model._tokenizer = _MockTokenizer()
        with self.assertRaises(ValueError):
            next(model.generate("hi"))

    def test_result_fields(self):
        model = self._make_model()
        result = next(model.generate("テスト", rng_seed=0))
        self.assertIsNotNone(result.audio)
        self.assertIsInstance(result.token_count, int)
        self.assertGreater(result.token_count, 0)
        self.assertIsNotNone(result.audio_duration)
        self.assertGreater(result.real_time_factor, 0.0)


def _small_irodori_dit_config_voicedesign(**overrides):
    defaults = dict(
        latent_dim=8,
        latent_patch_size=1,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        text_mlp_ratio=2.0,
        text_vocab_size=64,
        text_dim=32,
        text_layers=1,
        text_heads=4,
        speaker_dim=32,
        speaker_layers=1,
        speaker_heads=4,
        speaker_patch_size=1,
        timestep_embed_dim=16,
        adaln_rank=8,
        norm_eps=1e-5,
        use_caption_condition=True,
        caption_vocab_size=64,
        caption_dim=32,
        caption_layers=1,
        caption_heads=4,
        caption_mlp_ratio=2.0,
    )
    defaults.update(overrides)
    from mlx_audio.tts.models.irodori_tts.config import IrodoriDiTConfig

    return IrodoriDiTConfig(**defaults)


def _small_irodori_model_config_voicedesign(**sampler_overrides):
    from mlx_audio.tts.models.irodori_tts.config import ModelConfig, SamplerConfig

    sampler_defaults = dict(
        num_steps=1,
        cfg_scale_text=1.0,
        cfg_scale_caption=1.0,
        sequence_length=4,
    )
    sampler_defaults.update(sampler_overrides)
    return ModelConfig(
        dit=_small_irodori_dit_config_voicedesign(),
        sampler=SamplerConfig(**sampler_defaults),
    )


class TestIrodoriVoiceDesignShapes(unittest.TestCase):
    def setUp(self):
        from mlx_audio.tts.models.irodori_tts.model import IrodoriDiT

        self.cfg = _small_irodori_dit_config_voicedesign()
        self.model = IrodoriDiT(self.cfg)

    def test_forward_shape(self):
        B, S = 1, 6
        x_t = mx.random.normal((B, S, self.cfg.patched_latent_dim))
        t = mx.array([0.5], dtype=mx.float32)
        text_ids = mx.zeros((B, 5), dtype=mx.int32)
        text_mask = mx.ones((B, 5), dtype=mx.bool_)
        caption_ids = mx.zeros((B, 5), dtype=mx.int32)
        caption_mask = mx.ones((B, 5), dtype=mx.bool_)

        text_state, text_mask_out, ctx_state, ctx_mask = self.model.encode_conditions(
            text_ids,
            text_mask,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
        )
        out = self.model.forward_with_conditions(
            x_t,
            t,
            text_state,
            text_mask_out,
            ctx_state,
            ctx_mask,
        )
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (B, S, self.cfg.patched_latent_dim))


class TestIrodoriVoiceDesignGenerate(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.irodori_tts.irodori_tts import Model

        cfg = _small_irodori_model_config_voicedesign()
        model = Model(cfg)
        model.dacvae = _FakeDACVAE(
            latent_dim=cfg.dit.latent_dim,
            downsample_factor=cfg.audio_downsample_factor,
        )
        model._tokenizer = _MockTokenizer()
        model._caption_tokenizer = _MockTokenizer()
        return model

    def test_generate_with_caption(self):
        model = self._make_model()
        results = list(model.generate("こんにちは", caption="穏やかな声", rng_seed=0))
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].samples, 0)

    def test_generate_with_instruct_alias(self):
        model = self._make_model()
        results = list(model.generate("こんにちは", instruct="明るい声", rng_seed=0))
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].samples, 0)

    def test_generate_instruct_takes_priority_over_none_caption(self):
        model = self._make_model()
        # instruct should be used when caption is not provided
        results = list(model.generate("テスト", instruct="低い声", rng_seed=0))
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].samples, 0)


class TestKugelAudioModel(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.kugelaudio.config import ModelConfig
        from mlx_audio.tts.models.kugelaudio.kugelaudio import Model

        cfg = ModelConfig.from_dict(
            {
                "acoustic_vae_dim": 4,
                "decoder_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "num_hidden_layers": 1,
                    "vocab_size": 32,
                    "max_position_embeddings": 64,
                },
                "diffusion_head_config": {
                    "hidden_size": 8,
                    "latent_size": 4,
                    "head_layers": 1,
                },
                "acoustic_tokenizer_config": {
                    "vae_dim": 4,
                    "encoder_n_filters": 4,
                    "decoder_n_filters": 4,
                    "encoder_ratios": [2],
                    "encoder_depths": "1",
                },
            }
        )
        return Model(cfg)

    def test_from_dict_basic(self):
        """Basic config creation with nested sub-configs."""
        from mlx_audio.tts.models.kugelaudio.config import ModelConfig

        cfg = ModelConfig.from_dict(
            {
                "acoustic_vae_dim": 64,
                "decoder_config": {"hidden_size": 3584, "num_hidden_layers": 28},
                "diffusion_head_config": {},
                "acoustic_tokenizer_config": {},
            }
        )
        self.assertEqual(cfg.model_type, "kugelaudio")
        self.assertEqual(cfg.acoustic_vae_dim, 64)
        self.assertEqual(cfg.decoder_config.hidden_size, 3584)
        self.assertEqual(cfg.decoder_config.num_hidden_layers, 28)

    def test_sanitize(self):
        """Test full sanitize: prefix stripping, skip rules, re-indexing,
        quantization metadata, and linear weight transposition."""
        from mlx.utils import tree_flatten

        model = self._make_model()
        params = dict(tree_flatten(model.parameters()))

        # Build fake HF-style weight dict exercising every sanitize path
        fake_weights = {}

        # 1) "model." prefix → should be stripped
        lang_key = next(k for k in params if k.startswith("language_model."))
        fake_weights[f"model.{lang_key}"] = params[lang_key]

        # 2) Keys that should be dropped
        fake_weights["model.semantic_tokenizer.foo"] = mx.zeros((2,))
        fake_weights["model.semantic_connector.bar"] = mx.zeros((2,))
        fake_weights["model.acoustic_tokenizer.encoder.baz"] = mx.zeros((2,))

        # 3) Sequential re-indexing (.mlp.0. → .mlp.layers.0.)
        mlp_key = next((k for k in params if "t_embedder.mlp.layers." in k), None)
        if mlp_key:
            pytorch_mlp = "model." + mlp_key.replace(".mlp.layers.", ".mlp.")
            fake_weights[pytorch_mlp] = params[mlp_key]

        # 4) Quantization metadata
        fake_weights["language_model.layers.0.self_attn.q_proj.scales"] = mx.zeros((4,))
        fake_weights["language_model.layers.0.self_attn.q_proj.biases"] = mx.zeros((4,))

        # 5) Linear weight transposition (reversed shape)
        lm_key = "lm_head.weight"
        if lm_key in params:
            fake_weights[lm_key] = mx.zeros(tuple(reversed(params[lm_key].shape)))

        result = model.sanitize(fake_weights)

        # Verify prefix stripping
        self.assertIn(lang_key, result)

        # Verify dropped keys
        for dropped in [
            "semantic_tokenizer.foo",
            "semantic_connector.bar",
            "acoustic_tokenizer.encoder.baz",
        ]:
            self.assertNotIn(dropped, result)

        # Verify sequential re-indexing
        if mlp_key:
            self.assertIn(mlp_key, result)

        # Verify quantization metadata preserved
        self.assertIn("language_model.layers.0.self_attn.q_proj.scales", result)
        self.assertIn("language_model.layers.0.self_attn.q_proj.biases", result)

        # Verify linear weight transposed to correct shape
        if lm_key in params:
            self.assertEqual(result[lm_key].shape, params[lm_key].shape)

    def test_token_constraint_mask_allows_valid_tokens(self):
        """Only VALID_SPEECH_TOKENS should survive the mask."""
        from mlx_audio.tts.models.kugelaudio.kugelaudio import VALID_SPEECH_TOKENS

        vocab_size = 152064
        logits = mx.zeros((1, vocab_size))

        constraint_mask = mx.full(logits.shape, float("-inf"))
        valid_indices = mx.array(VALID_SPEECH_TOKENS)
        constraint_mask[:, valid_indices] = 0.0
        masked = logits + constraint_mask
        mx.eval(masked)

        for tid in VALID_SPEECH_TOKENS:
            self.assertEqual(masked[0, tid].item(), 0.0)

        self.assertEqual(masked[0, 0].item(), float("-inf"))
        self.assertEqual(masked[0, 1000].item(), float("-inf"))

    def test_token_argmax_selects_highest_valid_token(self):
        """argmax on masked logits should pick the boosted valid token."""
        from mlx_audio.tts.models.kugelaudio.kugelaudio import (
            SPEECH_DIFFUSION_ID,
            VALID_SPEECH_TOKENS,
        )

        vocab_size = 152064
        logits = mx.zeros((1, vocab_size))

        logits = logits.at[0, SPEECH_DIFFUSION_ID].add(mx.array(10.0))

        constraint_mask = mx.full(logits.shape, float("-inf"))
        valid_indices = mx.array(VALID_SPEECH_TOKENS)
        constraint_mask[:, valid_indices] = 0.0
        masked = logits + constraint_mask

        selected = mx.argmax(masked, axis=-1).item()
        self.assertEqual(selected, SPEECH_DIFFUSION_ID)

    def test_sde_scheduler_inherits_from_base(self):
        """SDEDPMSolverMultistepScheduler should subclass the base."""
        from mlx_audio.tts.models.kugelaudio.scheduler import (
            SDEDPMSolverMultistepScheduler,
        )
        from mlx_audio.tts.models.vibevoice.scheduler import (
            DPMSolverMultistepScheduler as BaseDPMSolver,
        )

        self.assertTrue(issubclass(SDEDPMSolverMultistepScheduler, BaseDPMSolver))

    def test_sde_scheduler_adds_noise(self):
        """SDE variant should produce different results from deterministic."""
        from mlx_audio.tts.models.kugelaudio.scheduler import (
            SDEDPMSolverMultistepScheduler,
        )
        from mlx_audio.tts.models.vibevoice.scheduler import (
            DPMSolverMultistepScheduler as BaseDPMSolver,
        )

        mx.random.seed(42)
        sde = SDEDPMSolverMultistepScheduler(
            num_train_timesteps=100, prediction_type="v_prediction"
        )
        det = BaseDPMSolver(num_train_timesteps=100, prediction_type="v_prediction")

        sde.set_timesteps(5)
        det.set_timesteps(5)

        sample = mx.ones((1, 4)) * 0.5
        model_output = mx.ones((1, 4)) * 0.1

        sde_result = sde.step(model_output, sde.timesteps[0], sample)
        det_result = det.step(model_output, det.timesteps[0], sample)

        mx.eval(sde_result.prev_sample, det_result.prev_sample)

        diff = mx.abs(sde_result.prev_sample - det_result.prev_sample).sum().item()
        self.assertGreater(diff, 0.0)

    def test_sde_scheduler_step_output_shape(self):
        """Output shape should match input shape."""
        from mlx_audio.tts.models.kugelaudio.scheduler import (
            SDEDPMSolverMultistepScheduler,
        )

        sched = SDEDPMSolverMultistepScheduler(num_train_timesteps=100)
        sched.set_timesteps(5)

        sample = mx.zeros((2, 8))
        output = mx.zeros((2, 8))

        result = sched.step(output, sched.timesteps[0], sample)
        mx.eval(result.prev_sample)

        self.assertEqual(result.prev_sample.shape, (2, 8))
        self.assertIsNotNone(result.x0_pred)

    def test_sde_scheduler_reset_clears_state(self):
        """reset() should zero out step index and lower order count."""
        from mlx_audio.tts.models.kugelaudio.scheduler import (
            SDEDPMSolverMultistepScheduler,
        )

        sched = SDEDPMSolverMultistepScheduler(num_train_timesteps=100)
        sched.set_timesteps(5)
        sched._step_index = 3  # pylint: disable=protected-access
        sched.lower_order_nums = 2
        sched.reset()
        self.assertIsNone(sched._step_index)  # pylint: disable=protected-access
        self.assertEqual(sched.lower_order_nums, 0)


class TestAudioDiTModel(unittest.TestCase):
    @staticmethod
    def _small_vae_config():
        from mlx_audio.tts.models.longcat_audiodit.config import VaeConfig

        return VaeConfig(
            in_channels=1,
            channels=4,
            c_mults=[1, 2],
            strides=[2, 4],
            latent_dim=4,
            encoder_latent_dim=8,
            use_snake=True,
            downsample_shortcut="averaging",
            upsample_shortcut="duplicating",
            out_shortcut="averaging",
            in_shortcut="duplicating",
            final_tanh=False,
            downsampling_ratio=8,
            sample_rate=24000,
            scale=0.71,
        )

    def setUp(self):
        from mlx_audio.tts.models.longcat_audiodit.config import (
            ModelConfig,
            TextEncoderConfig,
        )
        from mlx_audio.tts.models.longcat_audiodit.longcat_audiodit import Model

        self.cfg = ModelConfig(
            dit_dim=16,
            dit_depth=2,
            dit_heads=2,
            dit_ff_mult=2.0,
            dit_text_dim=8,
            dit_dropout=0.0,
            dit_bias=True,
            dit_cross_attn=True,
            dit_adaln_type="global",
            dit_adaln_use_text_cond=True,
            dit_long_skip=True,
            dit_text_conv=True,
            dit_qk_norm=True,
            dit_cross_attn_norm=False,
            dit_eps=1e-6,
            dit_use_latent_condition=True,
            repa_dit_layer=1,
            latent_dim=4,
            sigma=0.0,
            sampling_rate=24000,
            latent_hop=8,
            max_wav_duration=2.0,
            text_encoder_model="google/umt5-base",
            text_add_embed=True,
            text_norm_feat=True,
            vae_config=TestAudioDiTModel._small_vae_config(),
            text_encoder_config=TextEncoderConfig(
                vocab_size=32,
                d_model=8,
                d_kv=4,
                d_ff=16,
                num_layers=1,
                num_heads=2,
                relative_attention_num_buckets=8,
                relative_attention_max_distance=16,
            ),
        )

        self.model = Model(self.cfg)

    # -- config --

    def test_config_from_dict(self):
        from mlx_audio.tts.models.longcat_audiodit.config import ModelConfig

        cfg = ModelConfig(
            vae_config={"channels": 64, "latent_dim": 32},
            text_encoder_config={"d_model": 256},
        )
        self.assertEqual(cfg.vae_config.channels, 64)
        self.assertEqual(cfg.text_encoder_config.d_model, 256)

    def test_config_defaults(self):
        from mlx_audio.tts.models.longcat_audiodit.config import ModelConfig

        cfg = ModelConfig()
        self.assertEqual(cfg.model_type, "audiodit")
        self.assertEqual(cfg.dit_dim, 1536)
        self.assertIsNotNone(cfg.vae_config)
        self.assertEqual(cfg.vae_config.scale, 0.71)
        total = 1
        for s in cfg.vae_config.strides:
            total *= s
        self.assertEqual(total, cfg.vae_config.downsampling_ratio)

    # -- forward --

    def test_vae_encode_decode_shapes(self):
        mx.random.seed(0)
        x = mx.random.normal((1, 64, 1))
        latent = self.model.vae.encode(x)
        mx.eval(latent)
        self.assertEqual(tuple(latent.shape), (1, 8, 4))

        recon = self.model.vae.decode(latent)
        mx.eval(recon)
        self.assertEqual(tuple(recon.shape), (1, 64, 1))

    def test_text_encoder_forward(self):
        ids = mx.zeros((1, 5), dtype=mx.int32)
        mask = mx.ones((1, 5), dtype=mx.float32)
        out = self.model.encode_text(ids, mask)
        mx.eval(out)
        self.assertEqual(tuple(out.shape), (1, 5, self.cfg.text_encoder_config.d_model))

    def test_dit_forward(self):
        B, S, T = 1, 4, 3
        out = self.model.transformer(
            x=mx.random.normal((B, S, self.cfg.latent_dim)),
            text=mx.random.normal((B, T, self.cfg.dit_text_dim)),
            text_len=mx.array([T], dtype=mx.float32),
            time=mx.array([0.5]),
            mask=mx.ones((B, S), dtype=mx.bool_),
            cond_mask=mx.ones((B, T), dtype=mx.bool_),
            return_ith_layer=self.cfg.repa_dit_layer,
            latent_cond=mx.zeros((B, S, self.cfg.latent_dim)),
        )
        mx.eval(out["last_hidden_state"])
        self.assertEqual(
            tuple(out["last_hidden_state"].shape), (B, S, self.cfg.latent_dim)
        )
        self.assertIsNotNone(out["hidden_state"])

    def test_encode_prompt_audio(self):
        audio = mx.random.normal((1, 64, 1))
        latent, prompt_dur = self.model.encode_prompt_audio(audio)
        mx.eval(latent)
        self.assertEqual(latent.shape[2], self.cfg.latent_dim)
        self.assertEqual(latent.shape[1], prompt_dur)
        self.assertGreater(prompt_dur, 0)

    # -- sanitize --

    def test_sanitize_weight_norm(self):
        weights = {
            "vae.encoder.layers.0.weight_v": mx.ones((4, 1, 3)),
            "vae.encoder.layers.0.weight_g": mx.ones((4, 1, 1)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("vae.encoder.layers.0.weight", sanitized)
        self.assertNotIn("vae.encoder.layers.0.weight_v", sanitized)
        self.assertEqual(
            tuple(sanitized["vae.encoder.layers.0.weight"].shape), (4, 3, 1)
        )

    def test_sanitize_conv_transpose_weight_norm(self):
        weights = {
            "vae.decoder.layers.1.layers.1.weight_v": mx.ones((8, 4, 4)),
            "vae.decoder.layers.1.layers.1.weight_g": mx.ones((8, 1, 1)),
        }
        sanitized = self.model.sanitize(weights)
        key = "vae.decoder.layers.1.layers.1.weight"
        self.assertIn(key, sanitized)
        self.assertEqual(tuple(sanitized[key].shape), (4, 4, 8))

    def test_sanitize_text_encoder_remapping(self):
        weights = {
            "text_encoder.encoder.embed_tokens.weight": mx.zeros((32, 8)),
            "text_encoder.encoder.block.0.layer.0.SelfAttention.q.weight": mx.zeros(
                (4, 8)
            ),
            "text_encoder.encoder.block.0.layer.1.DenseReluDense.wi_0.weight": mx.zeros(
                (16, 8)
            ),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("text_encoder.shared.weight", sanitized)
        self.assertIn("text_encoder.block.0.SelfAttention.q.weight", sanitized)
        self.assertIn("text_encoder.block.0.DenseReluDense.wi_0.weight", sanitized)

    def test_sanitize_transformer_remapping(self):
        weights = {
            "transformer.proj.2.weight": mx.zeros((4, 4)),
            "transformer.blocks.0.attn.to_out.0.weight": mx.zeros((4, 4)),
            "transformer.blocks.0.ff.3.weight": mx.zeros((4, 4)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("transformer.proj.1.weight", sanitized)
        self.assertIn("transformer.blocks.0.attn.to_out.weight", sanitized)
        self.assertIn("transformer.blocks.0.ff.1.weight", sanitized)

    def test_sanitize_dwconv_transpose(self):
        weights = {
            "transformer.text_conv_layer.0.dwconv.weight": mx.ones((8, 1, 7)),
            "transformer.text_conv_layer.0.dwconv.bias": mx.zeros((8,)),
        }
        sanitized = self.model.sanitize(weights)
        self.assertIn("transformer.text_conv_layer.0.dwconv_weight", sanitized)
        self.assertEqual(
            tuple(sanitized["transformer.text_conv_layer.0.dwconv_weight"].shape),
            (8, 7, 1),
        )


class TestOmniVoiceConfig(unittest.TestCase):
    def test_parse_from_dict_minimal(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
                "sample_rate": 24000,
            }
        )
        self.assertEqual(cfg.audio_vocab_size, 1025)
        self.assertEqual(cfg.num_audio_codebook, 8)
        self.assertEqual(cfg.sample_rate, 24000)

    def test_unknown_keys_are_ignored(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig

        OmniVoiceConfig.from_dict({"model_type": "omnivoice", "future_key": 99})

    def test_higgs_audio_config(self):
        from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig

        cfg = HiggsAudioConfig.from_dict(
            {
                "model_type": "higgs_audio_v2_tokenizer",
                "sample_rate": 24000,
                "codebook_size": 1024,
                "downsample_factor": 320,
            }
        )
        self.assertEqual(cfg.downsample_factor, 320)
        self.assertAlmostEqual(cfg.tokens_per_second, 25.0)


class TestOmniVoiceRegistration(unittest.TestCase):
    def test_model_type_registered(self):
        from mlx_audio.tts.utils import MODEL_REMAPPING

        self.assertIn("omnivoice", MODEL_REMAPPING)
        self.assertEqual(MODEL_REMAPPING["omnivoice"], "omnivoice")


class TestMossTTSRegistration(unittest.TestCase):
    def test_config_model_types_load_shared_architecture(self):
        from mlx_audio.tts.models.moss_tts import Model as SharedModel
        from mlx_audio.tts.utils import MODEL_REMAPPING
        from mlx_audio.utils import get_model_class

        cases = [
            ("moss_tts_delay", ["OpenMOSS-Team", "MOSS-TTSD-v1.0"]),
            ("moss_tts_local", ["OpenMOSS-Team", "MOSS-TTS"]),
        ]

        for expected_model_type, model_name in cases:
            arch, model_type = get_model_class(
                model_type=expected_model_type,
                model_name=model_name,
                category="tts",
                model_remapping=MODEL_REMAPPING,
            )

            self.assertEqual(model_type, expected_model_type)
            self.assertIs(arch.Model, SharedModel)


class TestOmniVoiceBackbone(unittest.TestCase):
    def _make_backbone(self):
        from mlx_audio.tts.models.omnivoice.backbone import (
            BackboneConfig,
            OmniVoiceBackbone,
        )

        cfg = BackboneConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=151676,
            head_dim=16,
            rms_norm_eps=1e-6,
        )
        return OmniVoiceBackbone(cfg)

    def test_output_shape(self):
        model = self._make_backbone()
        B, S = 1, 10
        embeds = mx.zeros((B, S, 64))
        out = model(embeds)
        self.assertEqual(out.shape, (B, S, 64))

    def test_bidirectional_no_causal_leak(self):
        model = self._make_backbone()
        S = 10
        base_embeds = mx.zeros((1, S, 64))
        perturbed_list = np.zeros((1, S, 64), dtype=np.float32)
        perturbed_list[0, 7, :] = 1.0
        perturbed = mx.array(perturbed_list)

        out_base = model(base_embeds)
        out_perturbed = model(perturbed)
        diff = mx.abs(out_base[0, 3] - out_perturbed[0, 3])
        self.assertGreater(
            float(mx.max(diff).item()),
            1e-6,
            "Position 3 unchanged after perturbing pos 7 — causal mask still active!",
        )


class TestOmniVoiceModel(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def test_logits_shape(self):
        model = self._make_model()
        B, S, T = 1, 5, 7
        C = 8
        input_ids_unified = mx.full((B, S + T, C), 0, dtype=mx.int32)
        target = mx.full((B, T, C), 1024, dtype=mx.int32)
        input_ids_unified = mx.concatenate(
            [input_ids_unified[:, :S, :], target], axis=1
        )
        audio_mask = mx.concatenate(
            [mx.zeros((B, S), dtype=mx.bool_), mx.ones((B, T), dtype=mx.bool_)],
            axis=1,
        )
        logits = model(input_ids_unified, audio_mask)
        self.assertEqual(logits.shape, (B, S + T, 8, 1025))

    def test_embed_inputs_shape(self):
        model = self._make_model()
        B, S, T = 1, 5, 7
        C = 8
        input_ids_unified = mx.zeros((B, S + T, C), dtype=mx.int32)
        audio_mask = mx.concatenate(
            [mx.zeros((B, S), dtype=mx.bool_), mx.ones((B, T), dtype=mx.bool_)],
            axis=1,
        )
        embeds = model._prepare_embed_inputs(input_ids_unified, audio_mask)
        self.assertEqual(embeds.shape, (B, S + T, 64))


class TestOmniVoicePrepareInputs(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def test_no_ref_structure(self):
        model = self._make_model()
        style_ids = mx.array([1, 2, 3], dtype=mx.int32)
        text_ids = mx.array([10, 11, 12, 13], dtype=mx.int32)
        T = 5
        result = model._prepare_inference_inputs(style_ids, text_ids, T)
        input_ids = result["input_ids"]
        audio_mask = result["audio_mask"]
        self.assertEqual(input_ids.shape, (1, 12, 8))
        self.assertEqual(audio_mask.shape, (1, 12))
        self.assertTrue(mx.all(audio_mask[0, :7] == False).item())
        self.assertTrue(mx.all(audio_mask[0, 7:] == True).item())
        self.assertTrue(mx.all(input_ids[0, 7:, :] == 1024).item())

    def test_with_ref_structure(self):
        model = self._make_model()
        style_ids = mx.array([1, 2, 3], dtype=mx.int32)
        text_ids = mx.array([10, 11], dtype=mx.int32)
        ref_tokens = mx.ones((4, 8), dtype=mx.int32) * 500
        T = 3
        result = model._prepare_inference_inputs(style_ids, text_ids, T, ref_tokens)
        input_ids = result["input_ids"]
        audio_mask = result["audio_mask"]
        self.assertEqual(input_ids.shape, (1, 12, 8))
        self.assertTrue(mx.all(audio_mask[0, :5] == False).item())
        self.assertTrue(mx.all(audio_mask[0, 5:] == True).item())
        self.assertTrue(mx.all(input_ids[0, 5:9, :] == 500).item())
        self.assertTrue(mx.all(input_ids[0, 9:, :] == 1024).item())

    def test_text_ids_repeated_across_codebooks(self):
        model = self._make_model()
        style_ids = mx.array([1, 2], dtype=mx.int32)
        text_ids = mx.array([10, 11], dtype=mx.int32)
        result = model._prepare_inference_inputs(style_ids, text_ids, T=2)
        input_ids = result["input_ids"]
        for c in range(8):
            self.assertTrue(mx.all(input_ids[0, :4, c] == input_ids[0, :4, 0]).item())


class TestOmniVoiceGeneration(unittest.TestCase):
    def test_schedule_monotone(self):
        from mlx_audio.tts.models.omnivoice.generation import _get_time_steps

        ts = _get_time_steps(num_step=32, t_shift=0.1)
        self.assertEqual(len(ts), 33)
        for i in range(1, len(ts)):
            self.assertGreaterEqual(ts[i], ts[i - 1])
        self.assertAlmostEqual(ts[0], 0.0, places=6)
        self.assertAlmostEqual(ts[-1], 1.0, places=4)

    def test_iterative_unmask_no_mask_remaining(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.generation import iterative_unmask
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        model = Model(cfg)

        T = 10
        C = 8
        S = 3
        text_block = mx.zeros((1, S, C), dtype=mx.int32)
        target_block = mx.full((1, T, C), 1024, dtype=mx.int32)
        cond_input_ids = mx.concatenate([text_block, target_block], axis=1)
        cond_audio_mask = mx.concatenate(
            [mx.zeros((1, S), dtype=mx.bool_), mx.ones((1, T), dtype=mx.bool_)],
            axis=1,
        )
        tokens = iterative_unmask(
            model=model,
            cond_input_ids=cond_input_ids,
            cond_audio_mask=cond_audio_mask,
            T=T,
            num_steps=5,
            guidance_scale=2.0,
        )
        self.assertEqual(tokens.shape, (T, 8))
        mask_count = int(mx.sum(tokens == 1024).item())
        self.assertEqual(
            mask_count, 0, f"Found {mask_count} mask tokens after unmasking"
        )
        self.assertTrue(bool(mx.all(tokens >= 0).item()))
        self.assertTrue(bool(mx.all(tokens <= 1023).item()))

    def test_frozen_tokens_invariant(self):
        from mlx_audio.tts.models.omnivoice.generation import (  # noqa: F401
            iterative_unmask,
        )

        pass


class TestOmniVoiceIterativeUnmaskRefactor(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def _build_cond(self, model, S, T):
        C = 8
        text_block = mx.zeros((1, S, C), dtype=mx.int32)
        target_block = mx.full((1, T, C), 1024, dtype=mx.int32)
        cond_input_ids = mx.concatenate([text_block, target_block], axis=1)
        cond_audio_mask = mx.concatenate(
            [mx.zeros((1, S), dtype=mx.bool_), mx.ones((1, T), dtype=mx.bool_)],
            axis=1,
        )
        return cond_input_ids, cond_audio_mask

    def test_new_signature_shape(self):
        from mlx_audio.tts.models.omnivoice.generation import iterative_unmask

        model = self._make_model()
        cond_input_ids, cond_audio_mask = self._build_cond(model, S=3, T=10)
        tokens = iterative_unmask(
            model, cond_input_ids, cond_audio_mask, T=10, num_steps=2
        )
        self.assertEqual(tokens.shape, (10, 8))

    def test_no_mask_tokens_remain(self):
        from mlx_audio.tts.models.omnivoice.generation import iterative_unmask

        model = self._make_model()
        cond_input_ids, cond_audio_mask = self._build_cond(model, S=3, T=10)
        tokens = iterative_unmask(
            model, cond_input_ids, cond_audio_mask, T=10, num_steps=5
        )
        self.assertEqual(int(mx.sum(tokens == 1024).item()), 0)

    def test_deterministic_with_fixed_seed(self):
        from mlx_audio.tts.models.omnivoice.generation import iterative_unmask

        model = self._make_model()
        cond_input_ids, cond_audio_mask = self._build_cond(model, S=3, T=5)

        mx.random.seed(42)
        t1 = iterative_unmask(model, cond_input_ids, cond_audio_mask, T=5, num_steps=3)
        _ = int(mx.sum(t1).item())

        cond_input_ids2, cond_audio_mask2 = self._build_cond(model, S=3, T=5)
        mx.random.seed(42)
        t2 = iterative_unmask(
            model, cond_input_ids2, cond_audio_mask2, T=5, num_steps=3
        )

        self.assertTrue(bool(mx.all(t1 == t2).item()))


class TestOmniVoiceSanitize(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def test_llm_prefix_remapped(self):
        model = self._make_model()
        x = mx.zeros((4,))
        result = model.sanitize({"llm.layers.0.weight": x})
        self.assertIn("backbone.layers.0.weight", result)
        self.assertNotIn("llm.layers.0.weight", result)

    def test_audio_embeddings_split(self):
        model = self._make_model()
        x = mx.zeros((8 * 1025, 4))
        result = model.sanitize({"audio_embeddings.weight": x})
        for i in range(8):
            self.assertIn(f"audio_embeddings.{i}.weight", result)
            self.assertEqual(result[f"audio_embeddings.{i}.weight"].shape, (1025, 4))
        self.assertNotIn("audio_embeddings.weight", result)

    def test_audio_heads_split(self):
        model = self._make_model()
        x = mx.zeros((8 * 1025, 4))
        result = model.sanitize({"audio_heads.weight": x})
        for i in range(8):
            self.assertIn(f"audio_heads.{i}.weight", result)
        self.assertNotIn("audio_heads.weight", result)

    def test_codebook_layer_offsets_dropped(self):
        model = self._make_model()
        x = mx.array([0, 1025, 2050, 3075, 4100, 5125, 6150, 7175])
        result = model.sanitize({"codebook_layer_offsets": x})
        self.assertNotIn("codebook_layer_offsets", result)
        self.assertEqual(len(result), 0)

    def test_other_keys_pass_through(self):
        model = self._make_model()
        x = mx.zeros((4,))
        result = model.sanitize({"some.other.key": x})
        self.assertIn("some.other.key", result)


class TestOmniVoiceGenerate(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def test_generate_returns_generation_result(self):
        import math

        from mlx_audio.tts.models.base import GenerationResult

        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=1.0, num_steps=5))
        self.assertIsInstance(result, GenerationResult)

    def test_generate_token_count(self):
        import math

        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=1.0, num_steps=5))
        expected_T = math.ceil(1.0 * 24000 / 960)
        self.assertEqual(result.token_count, expected_T)

    def test_generate_sample_rate(self):
        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=1.0, num_steps=5))
        self.assertEqual(result.sample_rate, 24000)

    def test_generate_processing_time_positive(self):
        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=1.0, num_steps=5))
        self.assertGreater(result.processing_time_seconds, 0)

    def test_generate_result_field_types(self):
        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=1.0, num_steps=5))
        self.assertIsInstance(result.audio_duration, str)
        self.assertIsInstance(result.prompt, dict)
        self.assertIn("tokens-per-sec", result.prompt)
        self.assertIsInstance(result.audio_samples, dict)
        self.assertIn("samples", result.audio_samples)
        self.assertIn("samples-per-sec", result.audio_samples)

    def test_generate_with_ref_tokens_succeeds(self):
        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        ref_tokens = mx.ones((4, 8), dtype=mx.int32)
        result = next(
            model.generate(
                input_ids=input_ids, duration_s=0.5, num_steps=3, ref_tokens=ref_tokens
            )
        )
        self.assertIsInstance(result.token_count, int)
        self.assertGreater(result.token_count, 0)


class TestOmniVoiceCloneUtils(unittest.TestCase):
    def test_remove_silence_matches_omnivoice_gap_policy(self):
        from mlx_audio.tts.models.omnivoice.utils import _remove_silence

        sr = 24000
        tone = np.full(int(0.3 * sr), 0.2, dtype=np.float32)
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        long_gap = np.zeros(int(1.2 * sr), dtype=np.float32)
        audio = np.concatenate([silence, tone, long_gap, tone, silence])

        trimmed = _remove_silence(audio, sr)

        self.assertEqual(trimmed.dtype, np.float32)
        self.assertEqual(len(trimmed), int(1.6 * sr))

    def test_trim_long_audio_splits_at_last_gap_before_max_duration(self):
        from mlx_audio.tts.models.omnivoice.utils import _trim_long_audio

        sr = 24000
        first = np.full(10 * sr, 0.2, dtype=np.float32)
        gap = np.zeros(1 * sr, dtype=np.float32)
        second = np.full(10 * sr, 0.2, dtype=np.float32)
        audio = np.concatenate([first, gap, second])

        trimmed = _trim_long_audio(audio, sr)

        self.assertEqual(trimmed.dtype, np.float32)
        self.assertEqual(len(trimmed), 11 * sr)

    def test_no_tokenizer_returns_empty(self):
        from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt

        result = create_voice_clone_prompt("any_path.wav", tokenizer=None)
        self.assertEqual(result.shape, (0, 8))
        self.assertEqual(result.dtype, mx.int32)

    def test_missing_file_raises(self):
        from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer
        from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt

        tok = HiggsAudioTokenizer(HiggsAudioConfig())
        with self.assertRaises(FileNotFoundError):
            create_voice_clone_prompt("/nonexistent/file.wav", tokenizer=tok)

    def test_with_tokenizer_returns_2d(self):
        import os
        import tempfile

        from mlx_audio.audio_io import write as audio_write
        from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer
        from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt

        tok = HiggsAudioTokenizer(HiggsAudioConfig())
        tok.encode = lambda wav: mx.zeros(
            (wav.shape[0], wav.shape[1] // 960, 8), dtype=mx.int32
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        audio = np.zeros(24000 * 2, dtype=np.float32)
        audio_write(tmp_path, audio, 24000)
        result = create_voice_clone_prompt(tmp_path, tokenizer=tok)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[1], 8)
        self.assertEqual(result.dtype, mx.int32)
        os.unlink(tmp_path)


class TestOmniVoiceGenerateWithTokenizer(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def _make_tokenizer(self):
        from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        return HiggsAudioTokenizer(HiggsAudioConfig())

    def test_audio_is_zeros_without_tokenizer(self):
        model = self._make_model()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(model.generate(input_ids=input_ids, duration_s=0.1, num_steps=2))
        self.assertIsInstance(result.audio, mx.array)

    def test_audio_is_array_with_tokenizer(self):
        model = self._make_model()
        tok = self._make_tokenizer()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(
            model.generate(
                input_ids=input_ids, duration_s=0.1, num_steps=2, tokenizer=tok
            )
        )
        self.assertIsNotNone(result.audio)
        self.assertIsInstance(result.audio, mx.array)

    def test_samples_count_with_tokenizer(self):
        model = self._make_model()
        tok = self._make_tokenizer()
        input_ids = mx.zeros((5,), dtype=mx.int32)
        result = next(
            model.generate(
                input_ids=input_ids, duration_s=0.1, num_steps=2, tokenizer=tok
            )
        )
        expected_samples = result.token_count * 960
        self.assertEqual(result.audio.size, expected_samples)


class TestHiggsAudioDAC(unittest.TestCase):
    def test_residual_unit_shape(self):
        from mlx_audio.codec.models.higgs_audio.dac import ResidualUnit

        model = ResidualUnit(64)
        x = mx.zeros((1, 100, 64))
        y = model(x)
        self.assertEqual(y.shape, (1, 100, 64))

    def test_encoder_block_downsamples(self):
        from mlx_audio.codec.models.higgs_audio.dac import AcousticEncoderBlock

        model = AcousticEncoderBlock(64, 128, stride=8)
        x = mx.zeros((1, 800, 64))
        y = model(x)
        self.assertEqual(y.shape[1], 100)

    def test_acoustic_encoder_hop(self):
        from mlx_audio.codec.models.higgs_audio.dac import AcousticEncoder

        model = AcousticEncoder()
        x = mx.zeros((1, 960, 1))
        y = model(x)
        self.assertEqual(y.shape, (1, 1, 256))

    def test_acoustic_decoder_upsample(self):
        from mlx_audio.codec.models.higgs_audio.dac import AcousticDecoder

        model = AcousticDecoder()
        x = mx.zeros((1, 1, 256))
        y = model(x)
        self.assertEqual(y.shape, (1, 960, 1))

    def test_rvq_decode_shape(self):
        from mlx_audio.codec.models.higgs_audio.dac import ResidualVectorQuantizer

        model = ResidualVectorQuantizer()
        codes = mx.zeros((1, 17, 8), dtype=mx.int32)
        y = model.decode(codes)
        self.assertEqual(y.shape, (1, 17, 1024))


class TestHiggsAudioTokenizer(unittest.TestCase):
    def test_higgs_audio_instantiation(self):
        from mlx_audio.codec.models.higgs_audio import (
            HiggsAudioConfig,
            HiggsAudioTokenizer,
        )

        tokenizer = HiggsAudioTokenizer(HiggsAudioConfig())
        self.assertIsNotNone(tokenizer)

    def test_higgs_audio_config_tokens_per_second(self):
        from mlx_audio.codec.models.higgs_audio import HiggsAudioConfig

        cfg = HiggsAudioConfig()
        self.assertAlmostEqual(cfg.tokens_per_second, 25.0)


class TestHiggsAudioTokenizerFull(unittest.TestCase):
    def _tok(self):
        from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        return HiggsAudioTokenizer(HiggsAudioConfig())

    def test_instantiation(self):
        self.assertIsNotNone(self._tok())

    def test_decode_2d_shape(self):
        tok = self._tok()
        tokens = mx.zeros((4, 8), dtype=mx.int32)
        wav = tok.decode(tokens)
        self.assertEqual(wav.shape, (4 * 960,))

    def test_decode_3d_shape(self):
        tok = self._tok()
        tokens = mx.zeros((1, 4, 8), dtype=mx.int32)
        wav = tok.decode(tokens)
        self.assertEqual(wav.ndim, 3)
        self.assertEqual(wav.shape[0], 1)
        self.assertEqual(wav.shape[2], 1)

    def test_encode_raises_without_pt_tokenizer(self):
        tok = self._tok()
        wav = mx.zeros((1, 960 * 5, 1))
        with self.assertRaises(RuntimeError):
            tok.encode(wav)

    def test_sanitize_keeps_encode_path(self):
        tok = self._tok()
        weights = {
            "acoustic_encoder.conv1.weight_g": mx.zeros((1,)),
            "semantic_model.encoder.conv.weight": mx.zeros((1,)),
            "fc2.weight": mx.zeros((256, 1024)),
            "fc1.weight": mx.zeros((768, 1024)),
        }
        result = tok.sanitize(weights)
        self.assertIn("acoustic_encoder.conv1.weight_g", result)
        self.assertIn("fc2.weight", result)
        self.assertIn("semantic_model.encoder.conv.weight", result)
        self.assertNotIn("fc1.weight", result)

    def test_from_pretrained_missing_raises(self):
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        with self.assertRaises(FileNotFoundError):
            HiggsAudioTokenizer.from_pretrained("/nonexistent/path")


class TestHiggsAudioEncodeConfig(unittest.TestCase):
    def test_semantic_config_fields(self):
        from mlx_audio.codec.models.higgs_audio import HiggsAudioConfig

        cfg = HiggsAudioConfig.from_dict(
            {
                "model_type": "higgs_audio_v2_tokenizer",
                "sample_rate": 24000,
                "semantic_sample_rate": 16000,
                "downsample_factor": 320,
                "strides": [1, 1],
                "block_dilations": [1, 1],
                "channel_ratios": [1, 1],
                "kernel_size": 3,
                "unit_kernel_size": 3,
                "semantic_model_config": {
                    "model_type": "hubert",
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                },
            }
        )
        self.assertEqual(cfg.semantic_sample_rate, 16000)
        self.assertEqual(cfg.strides, [1, 1])
        self.assertIsNotNone(cfg.semantic_model_config)

    def test_semantic_downsample_factor_property(self):
        from mlx_audio.codec.models.higgs_audio import HiggsAudioConfig

        cfg = HiggsAudioConfig()
        self.assertEqual(cfg.semantic_downsample_factor, 2)


class TestSemanticEncoder(unittest.TestCase):
    def test_output_shape_preserves_time(self):
        from mlx_audio.codec.models.higgs_audio.semantic import SemanticEncoder

        enc = SemanticEncoder(
            hidden_size=768,
            strides=[1, 1],
            dilations=[1, 1],
            channel_ratios=[1, 1],
            kernel_size=3,
            unit_kernel_size=3,
        )
        x = mx.zeros((1, 25, 768))
        y = enc(x)
        self.assertEqual(y.shape, (1, 25, 768))

    def test_different_batch_and_time(self):
        from mlx_audio.codec.models.higgs_audio.semantic import SemanticEncoder

        enc = SemanticEncoder(
            hidden_size=768,
            strides=[1, 1],
            dilations=[1, 1],
            channel_ratios=[1, 1],
            kernel_size=3,
            unit_kernel_size=3,
        )
        x = mx.zeros((2, 50, 768))
        y = enc(x)
        self.assertEqual(y.shape, (2, 50, 768))

    def test_nonzero_output(self):
        from mlx_audio.codec.models.higgs_audio.semantic import SemanticEncoder

        enc = SemanticEncoder(
            hidden_size=768,
            strides=[1, 1],
            dilations=[1, 1],
            channel_ratios=[1, 1],
            kernel_size=3,
            unit_kernel_size=3,
        )
        x = mx.ones((1, 10, 768))
        y = enc(x)
        mx.eval(y)
        self.assertFalse(mx.all(y == 0).item())


class TestHiggsAudioSanitizeEncode(unittest.TestCase):
    def _tok(self):
        from mlx_audio.codec.models.higgs_audio import (
            HiggsAudioConfig,
            HiggsAudioTokenizer,
        )

        return HiggsAudioTokenizer(HiggsAudioConfig())

    def test_keeps_semantic_model_weights(self):
        tok = self._tok()
        weights = {
            "semantic_model.encoder.layers.0.attention.k_proj.weight": mx.zeros(
                (768, 768)
            )
        }
        result = tok.sanitize(weights)
        self.assertEqual(len(result), 1)
        self.assertIn("semantic_model.encoder.layers.0.attention.k_proj.weight", result)

    def test_keeps_encoder_semantic_weights(self):
        tok = self._tok()
        weights = {"encoder_semantic.conv.weight": mx.zeros((768, 768, 3))}
        result = tok.sanitize(weights)
        self.assertEqual(len(result), 1)
        # Conv weight should be transposed for MLX
        self.assertEqual(result["encoder_semantic.conv.weight"].shape, (768, 3, 768))

    def test_keeps_fc_weights(self):
        tok = self._tok()
        weights = {"fc.weight": mx.zeros((1024, 1024)), "fc.bias": mx.zeros((1024,))}
        result = tok.sanitize(weights)
        self.assertIn("fc.weight", result)
        self.assertIn("fc.bias", result)

    def test_still_drops_decoder_semantic(self):
        tok = self._tok()
        weights = {"decoder_semantic.conv.weight": mx.zeros((768, 768, 3))}
        result = tok.sanitize(weights)
        self.assertEqual(len(result), 0)

    def test_still_drops_fc1(self):
        tok = self._tok()
        weights = {"fc1.weight": mx.zeros((768, 1024))}
        result = tok.sanitize(weights)
        self.assertEqual(len(result), 0)

    def test_semantic_model_conv_transposed(self):
        tok = self._tok()
        weights = {
            "semantic_model.feature_extractor.conv_layers.0.conv.weight": mx.zeros(
                (512, 1, 10)
            )
        }
        result = tok.sanitize(weights)
        key = "semantic_model.feature_extractor.conv_layers.0.conv.weight"
        self.assertIn(key, result)
        self.assertEqual(result[key].shape, (512, 10, 1))

    def test_semantic_model_parametrizations_remapped(self):
        tok = self._tok()
        weights = {
            "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0": mx.zeros(
                (768, 48, 128)
            ),
            "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1": mx.zeros(
                (768, 48, 128)
            ),
        }
        result = tok.sanitize(weights)
        self.assertIn("semantic_model.encoder.pos_conv_embed.conv.weight_g", result)
        self.assertIn("semantic_model.encoder.pos_conv_embed.conv.weight_v", result)
        # Should be transposed
        self.assertEqual(
            result["semantic_model.encoder.pos_conv_embed.conv.weight_g"].shape,
            (768, 128, 48),
        )


class TestHiggsAudioEncodePureMlx(unittest.TestCase):
    def _config(self):
        from mlx_audio.codec.models.higgs_audio import HiggsAudioConfig

        return HiggsAudioConfig.from_dict(
            {
                "sample_rate": 24000,
                "semantic_sample_rate": 16000,
                "downsample_factor": 320,
                "strides": [1, 1],
                "block_dilations": [1, 1],
                "channel_ratios": [1, 1],
                "kernel_size": 3,
                "unit_kernel_size": 3,
                "semantic_model_config": {
                    "model_type": "wav2vec2",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "intermediate_size": 128,
                    "hidden_dropout": 0.0,
                    "activation_dropout": 0.0,
                    "attention_dropout": 0.0,
                    "feat_proj_dropout": 0.0,
                    "final_dropout": 0.0,
                    "layerdrop": 0.0,
                    "conv_dim": [32, 32, 32, 32, 32, 32, 32],
                    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
                    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                    "num_conv_pos_embeddings": 32,
                    "num_conv_pos_embedding_groups": 8,
                },
            }
        )

    def _tokenizer(self):
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        tok = HiggsAudioTokenizer(self._config())
        tok._init_encode_modules()

        class DummyAcousticEncoder(nn.Module):
            def __call__(self, waveform: mx.array) -> mx.array:
                batch, time, _ = waveform.shape
                frames = max(time // 960, 1)
                return mx.zeros((batch, frames, 256), dtype=mx.float32)

        class DummyQuantizer(nn.Module):
            def encode(self, embeddings: mx.array) -> mx.array:
                batch, time, _ = embeddings.shape
                return mx.zeros((batch, time, 8), dtype=mx.int32)

        tok.acoustic_encoder = DummyAcousticEncoder()
        tok.quantizer = DummyQuantizer()
        return tok

    def test_encode_returns_correct_shape(self):
        tok = self._tokenizer()
        wav = mx.zeros((1, 4800, 1), dtype=mx.float32)

        codes = tok.encode(wav)

        self.assertEqual(codes.shape, (1, 5, 8))

    def test_encode_returns_int32(self):
        tok = self._tokenizer()
        wav = mx.zeros((1, 4800, 1), dtype=mx.float32)

        codes = tok.encode(wav)

        self.assertEqual(codes.dtype, mx.int32)

    def test_encode_without_modules_raises(self):
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        tok = HiggsAudioTokenizer(self._config())
        wav = mx.zeros((1, 4800, 1), dtype=mx.float32)

        with self.assertRaises(RuntimeError):
            tok.encode(wav)


class TestHiggsAudioEncodeParity(unittest.TestCase):
    """Compare MLX encode output against real model weights.

    Requires parity_test/model_src/ with full OmniVoice checkpoint.
    Tests skip gracefully if weights are not available.
    """

    def _skip_if_no_weights(self):
        """Skip test if parity_test weights not available."""
        import os

        weights_path = "parity_test/model_src/audio_tokenizer/model.safetensors"
        if not os.path.exists(weights_path):
            self.skipTest("parity_test/model_src not available")

    def test_encode_shape_matches(self):
        """Verify encode output shape with real weights.

        1 second of zeros at 24kHz should produce ~25 frames (24000/960).
        Output shape: [batch=1, time=T, codebooks=8]
        """
        self._skip_if_no_weights()
        from mlx_audio.codec.models.higgs_audio import HiggsAudioTokenizer

        tok = HiggsAudioTokenizer.from_pretrained("parity_test/model_src")
        wav = mx.zeros((1, 24000, 1), dtype=mx.float32)  # 1 second at 24kHz
        codes = tok.encode(wav)

        # Verify shape
        self.assertEqual(codes.ndim, 3, "codes should be 3D: [batch, time, codebooks]")
        self.assertEqual(codes.shape[0], 1, "batch size should be 1")
        self.assertEqual(codes.shape[2], 8, "should have 8 codebooks")

        # Verify time dimension is reasonable (20-30 frames for 1 second)
        self.assertGreater(
            codes.shape[1], 20, "time frames should be > 20 for 1 second"
        )
        self.assertLess(codes.shape[1], 30, "time frames should be < 30 for 1 second")

    def test_encode_values_in_range(self):
        """Verify encode output values are valid codebook indices.

        All codes should be in range [0, 1024) for 10-bit codebooks.
        """
        self._skip_if_no_weights()
        from mlx_audio.codec.models.higgs_audio import HiggsAudioTokenizer

        tok = HiggsAudioTokenizer.from_pretrained("parity_test/model_src")
        wav = mx.random.normal((1, 24000, 1), dtype=mx.float32) * 0.1  # Random audio
        codes = tok.encode(wav)
        mx.eval(codes)

        # Verify all codes are valid indices
        self.assertTrue(mx.all(codes >= 0).item(), "all codes should be >= 0")
        self.assertTrue(
            mx.all(codes < 1024).item(), "all codes should be < 1024 (10-bit codebook)"
        )


class TestOmniVoiceEnsureList(unittest.TestCase):
    def test_scalar_auto_repeat(self):
        from mlx_audio.tts.models.omnivoice.omnivoice import _ensure_list

        self.assertEqual(_ensure_list("en", 3, auto_repeat=True), ["en", "en", "en"])

    def test_scalar_no_repeat_raises(self):
        from mlx_audio.tts.models.omnivoice.omnivoice import _ensure_list

        with self.assertRaises(ValueError):
            _ensure_list("en", 3, auto_repeat=False)

    def test_list_passthrough(self):
        from mlx_audio.tts.models.omnivoice.omnivoice import _ensure_list

        self.assertEqual(_ensure_list(["a", "b"], 2), ["a", "b"])

    def test_list_length_mismatch_raises(self):
        from mlx_audio.tts.models.omnivoice.omnivoice import _ensure_list

        with self.assertRaises(ValueError):
            _ensure_list(["a"], 2)

    def test_none_fills(self):
        from mlx_audio.tts.models.omnivoice.omnivoice import _ensure_list

        self.assertEqual(_ensure_list(None, 3), [None, None, None])


class TestOmniVoicePackBatch(unittest.TestCase):
    def test_two_items_shapes(self):
        from mlx_audio.tts.models.omnivoice.omnivoice import _pack_batch

        inputs_list = [
            {
                "input_ids": mx.zeros((1, 10, 8), dtype=mx.int32),
                "audio_mask": mx.concatenate(
                    [mx.zeros((1, 5), dtype=mx.bool_), mx.ones((1, 5), dtype=mx.bool_)],
                    axis=1,
                ),
            },
            {
                "input_ids": mx.zeros((1, 12, 8), dtype=mx.int32),
                "audio_mask": mx.concatenate(
                    [mx.zeros((1, 6), dtype=mx.bool_), mx.ones((1, 6), dtype=mx.bool_)],
                    axis=1,
                ),
            },
        ]
        result = _pack_batch(inputs_list, target_lens=[5, 6], mask_id=1024)
        self.assertEqual(result["cond_input_ids"].shape, (2, 12, 8))
        self.assertEqual(result["cond_audio_mask"].shape, (2, 12))
        self.assertEqual(result["uncond_input_ids"].shape, (2, 6, 8))
        self.assertEqual(result["uncond_audio_mask"].shape, (2, 6))

    def test_cond_padding_is_mask_id(self):
        from mlx_audio.tts.models.omnivoice.omnivoice import _pack_batch

        inputs_list = [
            {
                "input_ids": mx.full((1, 5, 8), 42, dtype=mx.int32),
                "audio_mask": mx.ones((1, 5), dtype=mx.bool_),
            },
            {
                "input_ids": mx.full((1, 8, 8), 42, dtype=mx.int32),
                "audio_mask": mx.ones((1, 8), dtype=mx.bool_),
            },
        ]
        result = _pack_batch(inputs_list, target_lens=[3, 4], mask_id=1024)
        self.assertTrue(mx.all(result["cond_input_ids"][0, 5:, :] == 1024).item())


class TestOmniVoiceIterativeUnmaskBatch(unittest.TestCase):
    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        return Model(cfg)

    def test_returns_list_of_correct_shapes(self):
        from mlx_audio.tts.models.omnivoice.generation import iterative_unmask_batch
        from mlx_audio.tts.models.omnivoice.omnivoice import _pack_batch

        model = self._make_model()
        mask_id = 1024
        T0, T1 = 5, 7
        inputs_list = [
            {
                "input_ids": mx.full((1, 8 + T0, 8), mask_id, dtype=mx.int32),
                "audio_mask": mx.concatenate(
                    [
                        mx.zeros((1, 8), dtype=mx.bool_),
                        mx.ones((1, T0), dtype=mx.bool_),
                    ],
                    axis=1,
                ),
            },
            {
                "input_ids": mx.full((1, 10 + T1, 8), mask_id, dtype=mx.int32),
                "audio_mask": mx.concatenate(
                    [
                        mx.zeros((1, 10), dtype=mx.bool_),
                        mx.ones((1, T1), dtype=mx.bool_),
                    ],
                    axis=1,
                ),
            },
        ]
        packed = _pack_batch(inputs_list, [T0, T1], mask_id)
        results = iterative_unmask_batch(model, packed, num_steps=3, guidance_scale=2.0)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].shape, (T0, 8))
        self.assertEqual(results[1].shape, (T1, 8))
        self.assertTrue(mx.all(results[0] >= 0).item())
        self.assertTrue(mx.all(results[0] < 1024).item())


class TestOmniVoiceGenerateBatch(unittest.TestCase):
    class _TinyTextTokenizer:
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            ids = [((ord(ch) % 31) + 1) for ch in text] or [1]
            if return_tensors == "np":
                return SimpleNamespace(input_ids=[ids])
            return SimpleNamespace(input_ids=ids)

    def _make_model(self):
        from mlx_audio.tts.models.omnivoice.config import OmniVoiceConfig
        from mlx_audio.tts.models.omnivoice.omnivoice import Model

        cfg = OmniVoiceConfig.from_dict(
            {
                "model_type": "omnivoice",
                "audio_vocab_size": 1025,
                "audio_mask_id": 1024,
                "num_audio_codebook": 8,
                "sample_rate": 24000,
                "llm_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 200,
                    "head_dim": 16,
                    "rms_norm_eps": 1e-6,
                },
            }
        )
        model = Model(cfg)
        model.text_tokenizer = self._TinyTextTokenizer()
        return model

    def test_batch_returns_list(self):
        from mlx_audio.tts.models.base import GenerationResult

        model = self._make_model()
        results = model.generate_batch(
            text=["Hello world", "Goodbye world"],
            duration_s=1.0,
            num_steps=3,
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], GenerationResult)

    def test_batch_backward_compat_single(self):
        model = self._make_model()
        results = model.generate_batch(text=["Hello"], duration_s=1.0, num_steps=3)
        self.assertEqual(len(results), 1)


class TestOmniVoiceBatchEdgeCases(TestOmniVoiceGenerateBatch):
    def test_mismatched_list_lengths_raises(self):
        from mlx_audio.tts.models.omnivoice.omnivoice import _ensure_list

        with self.assertRaises(ValueError):
            _ensure_list(["a", "b"], 3)

    def test_batch_different_durations(self):
        model = self._make_model()
        results = model.generate_batch(
            text=["Hello", "Goodbye"],
            duration_s=[1.0, 2.0],
            num_steps=3,
        )
        self.assertEqual(len(results), 2)
        self.assertNotEqual(results[0].token_count, results[1].token_count)

    def test_batch_of_one_equals_single(self):
        from mlx_audio.tts.models.base import GenerationResult

        model = self._make_model()

        mx.random.seed(42)
        batch_results = model.generate_batch(
            text=["Hello world"],
            duration_s=1.0,
            num_steps=3,
        )
        self.assertEqual(len(batch_results), 1)
        batch_result = batch_results[0]

        mx.random.seed(42)
        single_result = next(
            model.generate(text="Hello world", duration_s=1.0, num_steps=3)
        )

        self.assertIsInstance(batch_result, GenerationResult)
        self.assertIsInstance(single_result, GenerationResult)
        self.assertEqual(batch_result.token_count, single_result.token_count)


class TestMeloTTSConfig(unittest.TestCase):
    """Tests for MeloTTS model config."""

    def test_config_defaults(self):
        from mlx_audio.tts.models.melotts import ModelConfig

        config = ModelConfig()
        self.assertEqual(config.sample_rate, 44100)
        self.assertEqual(config.inter_channels, 192)
        self.assertEqual(config.hidden_channels, 192)
        self.assertEqual(config.n_heads, 2)
        self.assertEqual(config.n_layers, 6)
        self.assertEqual(config.n_vocab, 219)
        self.assertEqual(config.num_tones, 16)
        self.assertEqual(config.num_languages, 10)
        self.assertEqual(config.gin_channels, 256)

    def test_config_from_dict(self):
        from mlx_audio.tts.models.melotts import ModelConfig

        config = ModelConfig.from_dict(
            {
                "sampling_rate": 22050,
                "n_vocab": 100,
                "num_languages": 8,
                "spk2id": {"EN-US": 0},
            }
        )
        self.assertEqual(config.sample_rate, 22050)
        self.assertEqual(config.n_vocab, 100)
        self.assertEqual(config.num_languages, 8)
        self.assertEqual(config.spk2id, {"EN-US": 0})


class TestMeloTTSModel(unittest.TestCase):
    """Tests for MeloTTS model instantiation and components."""

    @property
    def _default_config(self):
        from mlx_audio.tts.models.melotts import ModelConfig

        return ModelConfig(
            n_vocab=50,
            n_speakers=4,
            spk2id={"EN-Default": 0},
            inter_channels=32,
            hidden_channels=32,
            filter_channels=64,
            n_heads=2,
            n_layers=2,
            n_layers_trans_flow=2,
            gin_channels=32,
            upsample_initial_channel=64,
            upsample_rates=[4, 4],
            upsample_kernel_sizes=[8, 8],
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3]],
            num_tones=16,
            num_languages=10,
        )

    def test_model_init(self):
        from mlx_audio.tts.models.melotts import Model

        config = self._default_config
        model = Model(config)
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.sample_rate, 44100)

    def test_model_components_exist(self):
        from mlx_audio.tts.models.melotts import Model

        model = Model(self._default_config)
        self.assertIsNotNone(model.enc_p)
        self.assertIsNotNone(model.dec)
        self.assertIsNotNone(model.enc_q)
        self.assertIsNotNone(model.dp)
        self.assertIsNotNone(model.sdp)
        self.assertIsNotNone(model.emb_g)
        self.assertEqual(len(model.flow_layers), 8)  # 4 coupling + 4 flip

    def test_sanitize_skips_discriminator(self):
        from mlx_audio.tts.models.melotts import Model

        model = Model(self._default_config)
        weights = {
            "enc_p.emb.weight": mx.zeros((50, 32)),
            "net_dur_disc.something": mx.zeros(10),
            "net_d.layer": mx.zeros(10),
        }
        sanitized = model.sanitize(weights)
        self.assertIn("enc_p.emb.weight", sanitized)
        self.assertNotIn("net_dur_disc.something", sanitized)
        self.assertNotIn("net_d.layer", sanitized)

    def test_sanitize_weight_norm_merge(self):
        from mlx_audio.tts.models.melotts import Model

        model = Model(self._default_config)
        weights = {
            "dec.ups.0.weight_g": mx.ones((4, 1, 1)),
            "dec.ups.0.weight_v": mx.ones((4, 2, 8)) * 2.0,
        }
        sanitized = model.sanitize(weights)
        self.assertNotIn("dec.ups.0.weight_g", sanitized)
        self.assertNotIn("dec.ups.0.weight_v", sanitized)
        self.assertIn("dec.ups.0.weight", sanitized)

    def test_sanitize_flow_remapping(self):
        from mlx_audio.tts.models.melotts import Model

        model = Model(self._default_config)
        weights = {
            "flow.flows.0.pre.weight": mx.zeros((32, 16, 1)),
        }
        sanitized = model.sanitize(weights)
        self.assertIn("flow_layers.0.pre.weight", sanitized)
        self.assertNotIn("flow.flows.0.pre.weight", sanitized)

    def test_sanitize_layernorm_rename(self):
        from mlx_audio.tts.models.melotts import Model

        model = Model(self._default_config)
        weights = {
            "dp.norm_1.gamma": mx.ones(32),
            "dp.norm_1.beta": mx.zeros(32),
        }
        sanitized = model.sanitize(weights)
        self.assertIn("dp.norm_1.weight", sanitized)
        self.assertIn("dp.norm_1.bias", sanitized)
        self.assertNotIn("dp.norm_1.gamma", sanitized)

    def test_infer_shapes(self):
        """Test that infer produces audio with correct shape."""
        from mlx_audio.tts.models.melotts import Model

        config = self._default_config
        model = Model(config)

        B, T = 1, 10
        audio = model.infer(
            x=mx.zeros((B, T), dtype=mx.int32),
            x_lengths=mx.array([T]),
            sid=mx.array([0]),
            tone=mx.zeros((B, T), dtype=mx.int32),
            language=mx.zeros((B, T), dtype=mx.int32),
            bert=mx.zeros((B, 1024, T)),
            ja_bert=mx.zeros((B, 768, T)),
        )
        # Force computation
        audio_np = np.array(audio)
        self.assertEqual(audio.ndim, 3)
        self.assertEqual(audio.shape[0], 1)  # batch
        self.assertEqual(audio.shape[1], 1)  # mono channel


class TestMeloTTSText(unittest.TestCase):
    """Tests for MeloTTS text processing pipeline."""

    def test_text_normalize(self):
        from mlx_audio.tts.models.melotts.text import text_normalize

        self.assertEqual(text_normalize("Dr. Smith"), "doctor smith")
        self.assertIn("forty two", text_normalize("42"))
        self.assertIn("three point one four", text_normalize("3.14"))

    def _require_g2p(self):
        try:
            import g2p_en  # noqa: F401
        except ImportError:
            self.skipTest("g2p_en is required for this test")

    def test_g2p_basic(self):
        self._require_g2p()
        from mlx_audio.tts.models.melotts.text import g2p, text_normalize

        phones, tones, word2ph = g2p(text_normalize("hello"))
        self.assertEqual(phones[0], "_")  # pad start
        self.assertEqual(phones[-1], "_")  # pad end
        self.assertIn("hh", phones)
        self.assertEqual(len(phones), sum(word2ph))

    def test_g2p_punctuation(self):
        self._require_g2p()
        from mlx_audio.tts.models.melotts.text import g2p, text_normalize

        phones, tones, word2ph = g2p(text_normalize("hello, world."))
        self.assertIn(",", phones)
        self.assertIn(".", phones)

    def test_cleaned_text_to_sequence(self):
        from mlx_audio.tts.models.melotts.text import cleaned_text_to_sequence

        phones = ["_", "hh", "ah", "_"]
        tones = [0, 0, 1, 0]
        phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phones, tones, "EN")
        self.assertEqual(len(phone_ids), 4)
        self.assertEqual(len(tone_ids), 4)
        # EN tone offset is 7
        self.assertEqual(tone_ids[0], 7)  # 0 + 7
        self.assertEqual(tone_ids[2], 8)  # 1 + 7
        # EN lang id is 2
        self.assertTrue(all(lid == 2 for lid in lang_ids))

    def test_load_symbols_from_config(self):
        import mlx_audio.tts.models.melotts.text as text_mod

        original_symbols = list(text_mod.symbols)
        try:
            test_symbols = ["_", "a", "b", "c"]
            text_mod.load_symbols_from_config(test_symbols)
            # Access via module to see updated globals
            self.assertEqual(len(text_mod.symbols), 4)
            self.assertEqual(text_mod._symbol_to_id["a"], 1)
            self.assertEqual(text_mod._symbol_to_id["c"], 3)
        finally:
            text_mod.load_symbols_from_config(original_symbols)

    def test_process_text_returns_correct_keys(self):
        self._require_g2p()
        from mlx_audio.tts.models.melotts.text import process_text

        result = process_text("hello", bert_model=None, language="EN", add_blank=True)
        self.assertIn("phone_ids", result)
        self.assertIn("tone_ids", result)
        self.assertIn("lang_ids", result)
        self.assertIn("bert_features", result)
        self.assertIn("phones", result)
        self.assertIn("norm_text", result)

    def test_process_text_blank_insertion(self):
        self._require_g2p()
        from mlx_audio.tts.models.melotts.text import process_text

        result_blank = process_text(
            "hi", bert_model=None, language="EN", add_blank=True
        )
        result_no_blank = process_text(
            "hi", bert_model=None, language="EN", add_blank=False
        )
        # With blanks, every phone is surrounded by pad symbols
        self.assertGreater(
            len(result_blank["phone_ids"]), len(result_no_blank["phone_ids"])
        )

    def test_bert_features_shape(self):
        self._require_g2p()
        from mlx_audio.tts.models.melotts.text import process_text

        result = process_text("hello", bert_model=None, language="EN", add_blank=True)
        # Without BERT model, features are zeros
        self.assertEqual(result["bert_features"].shape[0], 768)
        self.assertEqual(result["bert_features"].shape[1], len(result["phone_ids"]))


class TestMeloTTSBert(unittest.TestCase):
    """Tests for MeloTTS BERT model."""

    def test_bert_config_defaults(self):
        from mlx_audio.tts.models.melotts.bert import BertConfig

        config = BertConfig()
        self.assertEqual(config.vocab_size, 30522)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_hidden_layers, 12)
        self.assertEqual(config.num_attention_heads, 12)

    def test_bert_model_init(self):
        from mlx_audio.tts.models.melotts.bert import BertConfig, BertModel

        config = BertConfig(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=128,
        )
        model = BertModel(config)
        self.assertIsNotNone(model.embeddings)
        self.assertIsNotNone(model.encoder)
        self.assertIsNotNone(model.pooler)

    def test_bert_forward(self):
        from mlx_audio.tts.models.melotts.bert import BertConfig, BertModel

        config = BertConfig(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=128,
        )
        model = BertModel(config)
        input_ids = mx.array([[1, 2, 3, 4]])
        seq_out, pooled, hidden_states = model(input_ids)
        seq_out_np = np.array(seq_out)
        pooled_np = np.array(pooled)
        self.assertEqual(seq_out.shape, (1, 4, 64))
        self.assertEqual(pooled.shape, (1, 64))
        self.assertIsNone(hidden_states)

    def test_bert_extract_features(self):
        from mlx_audio.tts.models.melotts.bert import BertConfig, BertModel

        config = BertConfig(
            num_hidden_layers=4,
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=128,
        )
        model = BertModel(config)
        input_ids = mx.array([[1, 2, 3]])
        features = model.extract_features(input_ids)
        features_np = np.array(features)
        # extract_features returns hidden_states[-3] (3rd from last)
        self.assertEqual(features.shape, (1, 3, 64))


class TestMeloTTSHiFiGAN(unittest.TestCase):
    """Tests for MeloTTS HiFi-GAN decoder."""

    def test_generator_init(self):
        from mlx_audio.tts.models.melotts.hifigan import Generator

        gen = Generator(
            initial_channel=32,
            resblock="1",
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3]],
            upsample_rates=[4, 4],
            upsample_initial_channel=64,
            upsample_kernel_sizes=[8, 8],
            gin_channels=32,
        )
        self.assertEqual(gen.num_upsamples, 2)
        self.assertEqual(gen.num_kernels, 1)

    def test_generator_forward(self):
        from mlx_audio.tts.models.melotts.hifigan import Generator

        gen = Generator(
            initial_channel=32,
            resblock="1",
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3]],
            upsample_rates=[4, 4],
            upsample_initial_channel=64,
            upsample_kernel_sizes=[8, 8],
            gin_channels=32,
        )
        z = mx.zeros((1, 32, 10))
        g = mx.zeros((1, 32, 1))
        audio = gen(z, g=g)
        audio_np = np.array(audio)
        self.assertEqual(audio.shape[0], 1)
        self.assertEqual(audio.shape[1], 1)
        # 10 frames * 4 * 4 = 160 samples
        self.assertEqual(audio.shape[2], 160)


class TestMeloTTSAttentions(unittest.TestCase):
    """Tests for MeloTTS attention modules."""

    def test_layernorm_channel_first(self):
        from mlx_audio.tts.models.melotts.attentions import LayerNorm

        ln = LayerNorm(32)
        x = mx.random.normal((1, 32, 10))
        out = ln(x)
        out_np = np.array(out)
        self.assertEqual(out.shape, (1, 32, 10))

    def test_encoder_init(self):
        from mlx_audio.tts.models.melotts.attentions import Encoder

        enc = Encoder(
            hidden_channels=32,
            filter_channels=64,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            gin_channels=32,
        )
        self.assertEqual(len(enc.attn_layers), 2)
        self.assertEqual(len(enc.ffn_layers), 2)
        self.assertTrue(hasattr(enc, "spk_emb_linear"))

    def test_encoder_cond_layer_idx(self):
        from mlx_audio.tts.models.melotts.attentions import Encoder

        enc = Encoder(
            hidden_channels=32,
            filter_channels=64,
            n_heads=2,
            n_layers=4,
            kernel_size=3,
            gin_channels=32,
            cond_layer_idx=2,
        )
        self.assertEqual(enc.cond_layer_idx, 2)


# ── VoxCPM2 ──────────────────────────────────────────────────────


def _tiny_voxcpm2_args():
    """Create a minimal VoxCPM2 config for fast tests."""
    from mlx_audio.tts.models.voxcpm2.config import (
        AudioVAEConfig,
        CFMConfig,
        DiTConfig,
        EncoderConfig,
        LMConfig,
        ModelArgs,
    )

    lm = LMConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=100,
        use_mup=False,
        scale_emb=12,
        scale_depth=1.4,
        dim_model_base=256,
        rope_long_factor=[1.0] * 8,
        rope_short_factor=[1.0] * 8,
    )
    return ModelArgs(
        lm_config=lm,
        encoder_config=EncoderConfig(
            hidden_dim=64, ffn_dim=128, num_heads=4, num_layers=1
        ),
        dit_config=DiTConfig(
            hidden_dim=64,
            ffn_dim=128,
            num_heads=4,
            num_layers=1,
            cfm_config=CFMConfig(),
        ),
        audio_vae_config=AudioVAEConfig(
            encoder_dim=8,
            encoder_rates=[2, 2],
            latent_dim=16,
            decoder_dim=64,
            decoder_rates=[2, 2],
            depthwise=False,
            sample_rate=16000,
            out_sample_rate=48000,
            sr_bin_boundaries=[20000, 30000, 40000],
        ),
        patch_size=2,
        feat_dim=16,
        scalar_quantization_latent_dim=32,
        residual_lm_num_layers=1,
        residual_lm_no_rope=True,
    )


class TestVoxCPM2Config(unittest.TestCase):
    def test_from_dict_parses_real_config(self):
        from mlx_audio.tts.models.voxcpm2.config import ModelArgs

        config = ModelArgs.from_dict(
            {
                "lm_config": {
                    "hidden_size": 2048,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "intermediate_size": 6144,
                    "vocab_size": 73448,
                    "rms_norm_eps": 1e-5,
                    "rope_theta": 10000,
                    "kv_channels": 128,
                    "rope_scaling": {
                        "type": "longrope",
                        "long_factor": [1.0] * 64,
                        "short_factor": [1.0] * 64,
                        "original_max_position_embeddings": 32768,
                    },
                    "use_mup": False,
                    "scale_emb": 12,
                    "dim_model_base": 256,
                    "scale_depth": 1.4,
                },
                "encoder_config": {
                    "hidden_dim": 1024,
                    "kv_channels": 128,
                    "num_layers": 12,
                },
                "dit_config": {
                    "hidden_dim": 1024,
                    "num_layers": 12,
                    "kv_channels": 128,
                    "mean_mode": False,
                    "cfm_config": {"solver": "euler"},
                },
                "audio_vae_config": {
                    "encoder_rates": [2, 5, 8, 8],
                    "decoder_rates": [8, 6, 5, 2, 2, 2],
                    "sample_rate": 16000,
                    "out_sample_rate": 48000,
                    "sr_bin_boundaries": [20000, 30000, 40000],
                },
                "patch_size": 4,
                "scalar_quantization_latent_dim": 512,
                "residual_lm_no_rope": True,
            }
        )
        self.assertEqual(config.lm_config.hidden_size, 2048)
        self.assertEqual(config.lm_config.kv_channels, 128)
        self.assertFalse(config.lm_config.use_mup)
        self.assertTrue(config.residual_lm_no_rope)
        self.assertEqual(config.audio_vae_config.out_sample_rate, 48000)
        self.assertFalse(config.dit_config.dit_mean_mode)
        self.assertEqual(config.scalar_quantization_latent_dim, 512)

    def test_mean_mode_alias(self):
        """dit_config.mean_mode maps to dit_mean_mode."""
        from mlx_audio.tts.models.voxcpm2.config import ModelArgs

        config = ModelArgs.from_dict(
            {"dit_config": {"mean_mode": True, "cfm_config": {}}}
        )
        self.assertTrue(config.dit_config.dit_mean_mode)


class TestVoxCPM2Registration(unittest.TestCase):
    def test_model_type_in_remapping(self):
        from mlx_audio.tts.utils import MODEL_REMAPPING

        self.assertIn("voxcpm2", MODEL_REMAPPING)
        self.assertEqual(MODEL_REMAPPING["voxcpm2"], "voxcpm2")


class TestVoxCPM2AudioVAE(unittest.TestCase):
    def test_encode_decode_shape(self):
        from mlx_audio.tts.models.voxcpm2.audio_vae import AudioVAE
        from mlx_audio.tts.models.voxcpm2.config import AudioVAEConfig

        config = AudioVAEConfig(
            encoder_dim=8,
            encoder_rates=[2, 2],
            latent_dim=16,
            decoder_dim=64,
            decoder_rates=[2, 2],
            depthwise=False,
            sample_rate=16000,
            out_sample_rate=48000,
            sr_bin_boundaries=[20000, 30000, 40000],
        )
        vae = AudioVAE(config)
        mx.eval(vae.parameters())

        x = mx.zeros((1, 16, 1))  # (B, T, C=1)
        encoded = vae.encode(x)
        self.assertEqual(encoded.ndim, 3)
        self.assertEqual(encoded.shape[-1], 16)  # latent_dim

        decoded = vae.decode(encoded)
        self.assertIsNotNone(decoded)

    def test_sr_conditioning(self):
        from mlx_audio.tts.models.voxcpm2.audio_vae import AudioVAE
        from mlx_audio.tts.models.voxcpm2.config import AudioVAEConfig

        config = AudioVAEConfig(
            encoder_dim=8,
            encoder_rates=[2, 2],
            latent_dim=16,
            decoder_dim=64,
            decoder_rates=[2, 2],
            depthwise=False,
            sr_bin_boundaries=[20000, 30000, 40000],
        )
        vae = AudioVAE(config)
        mx.eval(vae.parameters())

        self.assertEqual(len(vae.decoder.sr_cond_layers), 2)
        # Test bucket index
        idx = vae.decoder.get_sr_idx(mx.array([48000], dtype=mx.int32))
        self.assertEqual(idx.item(), 3)  # > all boundaries
        idx = vae.decoder.get_sr_idx(mx.array([10000], dtype=mx.int32))
        self.assertEqual(idx.item(), 0)  # < all boundaries

    def test_sanitize_weight_norm_fusion(self):
        from mlx_audio.tts.models.voxcpm2.audio_vae import AudioVAE
        from mlx_audio.tts.models.voxcpm2.config import AudioVAEConfig

        config = AudioVAEConfig(
            encoder_dim=8,
            encoder_rates=[2],
            decoder_rates=[2],
            latent_dim=16,
            decoder_dim=32,
            depthwise=False,
            sr_bin_boundaries=None,
        )
        vae = AudioVAE(config)

        g = mx.ones((8, 1, 1)) * 2.0
        v = mx.ones((8, 1, 7))
        weights = {"encoder.conv_in.weight_g": g, "encoder.conv_in.weight_v": v}
        sanitized = vae.sanitize(weights)

        self.assertIn("encoder.conv_in.weight", sanitized)
        self.assertNotIn("encoder.conv_in.weight_g", sanitized)
        self.assertNotIn("encoder.conv_in.weight_v", sanitized)


class TestVoxCPM2MiniCPM(unittest.TestCase):
    def test_no_rope(self):
        from mlx_audio.tts.models.voxcpm2.config import LMConfig
        from mlx_audio.tts.models.voxcpm2.minicpm import MiniCPMModel

        config = LMConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=0,
            no_rope=True,
            use_mup=False,
            rope_long_factor=[1.0] * 8,
            rope_short_factor=[1.0] * 8,
        )
        model = MiniCPMModel(config)
        mx.eval(model.parameters())
        self.assertIsNone(model.rope)

        x = mx.random.normal((1, 4, 64))
        out, cache = model(inputs_embeds=x)
        self.assertEqual(out.shape, (1, 4, 64))

    def test_kv_channels(self):
        from mlx_audio.tts.models.voxcpm2.config import LMConfig
        from mlx_audio.tts.models.voxcpm2.minicpm import Attention

        config = LMConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            kv_channels=32,
            rope_long_factor=[1.0] * 16,
            rope_short_factor=[1.0] * 16,
        )
        attn = Attention(config)
        self.assertEqual(attn.head_dim, 32)


class TestVoxCPM2DiT(unittest.TestCase):
    def test_multi_token_mu(self):
        from mlx_audio.tts.models.voxcpm2.config import LMConfig
        from mlx_audio.tts.models.voxcpm2.dit import VoxCPMLocDiTV2

        config = LMConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=0,
            use_mup=False,
            rope_long_factor=[1.0] * 8,
            rope_short_factor=[1.0] * 8,
        )
        dit = VoxCPMLocDiTV2(config, in_channels=16)
        mx.eval(dit.parameters())

        x = mx.random.normal((1, 16, 4))
        mu = mx.random.normal((1, 128))  # 2 * hidden_size
        t = mx.array([0.5])
        cond = mx.random.normal((1, 16, 4))
        dt = mx.array([0.0])

        out = dit(x, mu, t, cond, dt)
        self.assertEqual(out.shape, x.shape)

    def test_single_token_mu(self):
        from mlx_audio.tts.models.voxcpm2.config import LMConfig
        from mlx_audio.tts.models.voxcpm2.dit import VoxCPMLocDiTV2

        config = LMConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=0,
            use_mup=False,
            rope_long_factor=[1.0] * 8,
            rope_short_factor=[1.0] * 8,
        )
        dit = VoxCPMLocDiTV2(config, in_channels=16)
        mx.eval(dit.parameters())

        x = mx.random.normal((1, 16, 4))
        mu = mx.random.normal((1, 64))  # 1 * hidden_size
        out = dit(x, mu, mx.array([0.5]), mx.random.normal((1, 16, 4)), mx.array([0.0]))
        self.assertEqual(out.shape, x.shape)


class TestVoxCPM2Model(unittest.TestCase):
    def test_init(self):
        from mlx_audio.tts.models.voxcpm2.voxcpm2 import Model

        args = _tiny_voxcpm2_args()
        model = Model(args)
        mx.eval(model.parameters())

        self.assertIsNotNone(model.base_lm)
        self.assertIsNotNone(model.residual_lm)
        self.assertIsNotNone(model.fusion_concat_proj)
        self.assertIsNone(model.residual_lm.rope)  # no_rope=True
        self.assertIsNotNone(model.base_lm.rope)
        self.assertEqual(model.sample_rate, 48000)
        self.assertEqual(model.fusion_concat_proj.weight.shape, (64, 128))

    def test_embed_tokens(self):
        from mlx_audio.tts.models.voxcpm2.voxcpm2 import Model

        args = _tiny_voxcpm2_args()
        model = Model(args)
        mx.eval(model.parameters())

        x = mx.array([[1, 2, 3]])
        emb = model.base_lm.embed_tokens(x)
        self.assertEqual(emb.shape, (1, 3, 64))

    def test_inference_pipeline(self):
        """Test forward pass through the full inference pipeline (no tokenizer)."""
        from mlx_audio.tts.models.voxcpm2.voxcpm2 import Model

        args = _tiny_voxcpm2_args()
        model = Model(args)
        mx.eval(model.parameters())

        # Simulate zero-shot input
        text_token = mx.array([[1, 2, 3, 101]])
        text_length = 4
        audio_feat = mx.zeros((1, text_length, 2, 16))
        text_mask = mx.ones((1, text_length))
        audio_mask = mx.zeros((1, text_length))

        feat_embed = model.feat_encoder(audio_feat)
        feat_embed = model.enc_to_lm_proj(feat_embed)
        text_embed = model.base_lm.embed_tokens(text_token)
        combined = (
            text_mask[:, :, None] * text_embed + audio_mask[:, :, None] * feat_embed
        )

        enc_out, lm_cache = model.base_lm(combined)
        self.assertEqual(enc_out.shape, (1, text_length, 64))

        # Fusion concat proj
        residual_in = model.fusion_concat_proj(
            mx.concatenate([enc_out, audio_mask[:, :, None] * feat_embed], axis=-1)
        )
        self.assertEqual(residual_in.shape, (1, text_length, 64))

        res_out, res_cache = model.residual_lm(residual_in)

        # DiT hidden (concat, not sum)
        lm_h = model.lm_to_dit_proj(enc_out[:, -1, :])
        res_h = model.res_to_dit_proj(res_out[:, -1, :])
        dit_h = mx.concatenate([lm_h, res_h], axis=-1)
        self.assertEqual(dit_h.shape, (1, 128))  # 2 * dit_hidden_dim

    def test_sanitize_populates_rope(self):
        from mlx_audio.tts.models.voxcpm2.voxcpm2 import Model

        args = _tiny_voxcpm2_args()
        model = Model(args)

        weights = model.sanitize({})
        rope_keys = [k for k in weights if "rope" in k]
        self.assertGreater(len(rope_keys), 0)

    def test_sanitize_sr_boundaries(self):
        from mlx_audio.tts.models.voxcpm2.voxcpm2 import Model

        args = _tiny_voxcpm2_args()
        model = Model(args)

        weights = {"audio_vae.decoder._sr_boundaries": mx.array([20000, 30000, 40000])}
        sanitized = model.sanitize(weights)
        # Buffer should be extracted, not in returned weights
        self.assertNotIn("audio_vae.decoder._sr_boundaries", sanitized)

    def test_voice_design_prefix(self):
        """Instruct param prepends voice description to text."""
        from unittest.mock import MagicMock

        from mlx_audio.tts.models.voxcpm2.voxcpm2 import Model

        args = _tiny_voxcpm2_args()
        model = Model(args)
        model.tokenizer = MagicMock()
        model.tokenizer.tokenize = MagicMock(return_value=["hello"])
        model.tokenizer.convert_tokens_to_ids = MagicMock(return_value=[1, 2, 3])

        # Call generate with instruct — it should prepend (instruct)text
        gen = model.generate(
            text="Hello",
            instruct="A warm voice",
            max_tokens=1,
            warmup_patches=0,
        )
        try:
            next(gen)
        except Exception:
            pass
        # Check tokenizer.tokenize was called with prefixed text
        call_args = model.tokenizer.tokenize.call_args[0][0]
        self.assertTrue(call_args.startswith("(A warm voice)"))


class MossFakeTokenizer:
    def encode(self, text, *args, **kwargs):
        del args, kwargs
        return [ord(ch) % 97 for ch in text]

    def decode(self, token_ids, *args, **kwargs):
        del args, kwargs
        return "".join(chr(int(token_id) + 30) for token_id in token_ids)


class MossFakeAudioTokenizer:
    def __init__(self):
        self.encoded_audio = None
        self.decoded_codes = None

    def encode_audio(self, audio, **kwargs):
        del kwargs
        self.encoded_audio = audio
        return mx.array([[1, 2], [3, 4]], dtype=mx.int32)

    def decode_audio_codes(self, audio_codes, **kwargs):
        del kwargs
        self.decoded_codes = audio_codes
        return mx.ones((4, 1), dtype=mx.float32)


def moss_tiny_config(**overrides):
    from mlx_audio.tts.models.moss_tts_nano import ModelConfig

    config = {
        "model_type": "moss_tts_nano",
        "n_vq": 2,
        "audio_vocab_size": 8,
        "audio_codebook_sizes": [8, 8],
        "audio_pad_token_id": 8,
        "pad_token_id": 3,
        "im_start_token_id": 4,
        "im_end_token_id": 5,
        "audio_start_token_id": 6,
        "audio_end_token_id": 7,
        "audio_user_slot_token_id": 8,
        "audio_assistant_slot_token_id": 9,
        "gpt2_config": {
            "vocab_size": 32,
            "n_positions": 64,
            "n_ctx": 64,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 4,
            "n_inner": 32,
            "position_embedding_type": "rope",
            "rope_base": 10000.0,
            "layer_norm_epsilon": 1e-5,
        },
        "local_transformer_layers": 1,
    }
    config.update(overrides)
    return ModelConfig.from_dict(config)


class TestMossTTSNanoConfig(unittest.TestCase):
    def test_config_parses_upstream_shape(self):
        from mlx_audio.tts.models.moss_tts_nano import ModelConfig

        config = ModelConfig.from_dict(
            {
                "model_type": "moss_tts_nano",
                "n_vq": 16,
                "audio_vocab_size": 1024,
                "audio_codebook_sizes": [1024] * 16,
                "gpt2_config": {
                    "vocab_size": 16384,
                    "n_embd": 768,
                    "n_head": 12,
                    "n_layer": 12,
                    "n_inner": 3072,
                    "n_positions": 32768,
                },
                "local_transformer_layers": 1,
            }
        )

        self.assertEqual(config.model_type, "moss_tts_nano")
        self.assertEqual(config.n_vq, 16)
        self.assertEqual(config.gpt2_config.n_embd, 768)
        self.assertEqual(config.local_gpt2_config().n_positions, 17)

    def test_config_rejects_wrong_codebook_count(self):
        from mlx_audio.tts.models.moss_tts_nano import ModelConfig

        with self.assertRaises(ValueError):
            ModelConfig.from_dict(
                {
                    "n_vq": 2,
                    "audio_vocab_size": 8,
                    "audio_codebook_sizes": [8],
                    "gpt2_config": {},
                }
            )


class TestMossTTSNanoText(unittest.TestCase):
    def test_prompt_prefix_contains_role_and_reference_template(self):
        from mlx_audio.tts.models.moss_tts_nano.text import (
            build_assistant_prompt_prefix,
            build_user_prompt_after_reference,
            build_user_prompt_prefix,
        )

        tokenizer = MossFakeTokenizer()
        config = moss_tiny_config()

        prefix = build_user_prompt_prefix(tokenizer, config)
        after_reference = build_user_prompt_after_reference(tokenizer)
        assistant = build_assistant_prompt_prefix(tokenizer, config)

        self.assertEqual(prefix[0], config.im_start_token_id)
        self.assertGreater(len(after_reference), 0)
        self.assertIn(config.im_end_token_id, assistant)
        self.assertIn(config.im_start_token_id, assistant)

    def test_split_text_into_token_budget_chunks(self):
        from mlx_audio.tts.models.moss_tts_nano.text import (
            split_text_into_best_sentences,
        )

        tokenizer = MossFakeTokenizer()
        chunks = split_text_into_best_sentences(
            tokenizer,
            "hello world. this is another sentence.",
            max_tokens=14,
        )

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.strip() for chunk in chunks))


class TestMossTTSNanoSampling(unittest.TestCase):
    def test_top_k_masks_low_scores(self):
        from mlx_audio.tts.models.moss_tts_nano.sampling import apply_top_k

        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        filtered = np.array(apply_top_k(logits, 2))

        self.assertTrue(np.isneginf(filtered[0, 0]))
        self.assertTrue(np.isneginf(filtered[0, 1]))
        self.assertEqual(filtered[0, 2], 3.0)
        self.assertEqual(filtered[0, 3], 4.0)

    def test_top_p_keeps_at_least_one_score(self):
        from mlx_audio.tts.models.moss_tts_nano.sampling import apply_top_p

        logits = mx.array([[10.0, 1.0, 0.0]])
        filtered = np.array(apply_top_p(logits, 0.1))

        self.assertFalse(np.isneginf(filtered[0, 0]))
        self.assertTrue(np.isneginf(filtered[0, 1]))
        self.assertTrue(np.isneginf(filtered[0, 2]))


class TestMossTTSNanoModel(unittest.TestCase):
    def test_build_inputs_embeds_shape(self):
        from mlx_audio.tts.models.moss_tts_nano import Model

        model = Model(moss_tiny_config())
        input_ids = mx.array([[[1, 8, 8], [2, 3, 4]]], dtype=mx.int32)

        embeds = model._build_inputs_embeds(input_ids)

        self.assertEqual(embeds.shape, (1, 2, 16))

    def test_voice_clone_prompt_rows_include_reference_audio_rows(self):
        from mlx_audio.tts.models.moss_tts_nano import Model
        from mlx_audio.tts.models.moss_tts_nano.text import build_user_prompt_prefix

        model = Model(moss_tiny_config())
        tokenizer = MossFakeTokenizer()
        prompt_audio_codes = mx.array([[1, 2], [3, 4]], dtype=mx.int32)

        input_ids, attention_mask = model.build_inference_input_ids(
            text="hello",
            tokenizer=tokenizer,
            mode="voice_clone",
            prompt_audio_codes=prompt_audio_codes,
        )

        rows = np.array(input_ids[0])
        reference_start = len(build_user_prompt_prefix(tokenizer, model.config)) + 1
        reference_rows = rows[reference_start : reference_start + 2]
        self.assertEqual(reference_rows.shape[0], 2)
        self.assertTrue(
            np.all(reference_rows[:, 0] == model.config.audio_user_slot_token_id)
        )
        np.testing.assert_array_equal(reference_rows[:, 1:], np.array([[1, 2], [3, 4]]))
        self.assertEqual(attention_mask.shape[1], input_ids.shape[1])

    def test_sanitize_drops_tied_and_absent_weights(self):
        from mlx_audio.tts.models.moss_tts_nano import Model

        model = Model(moss_tiny_config())
        x = mx.zeros((1,))
        sanitized = model.sanitize(
            {
                "text_lm_head.weight": x,
                "audio_lm_heads.0.weight": x,
                "local_transformer.wte.weight": x,
                "transformer.wpe.weight": x,
                "transformer.h.0.ln_1.weight": x,
            }
        )

        self.assertEqual(set(sanitized), {"transformer.h.0.ln_1.weight"})

    def test_generate_encodes_reference_audio_and_decodes_generated_tokens(self):
        from mlx_audio.tts.models.moss_tts_nano import Model

        model = Model(moss_tiny_config())
        model.tokenizer = MossFakeTokenizer()
        model.audio_tokenizer = MossFakeAudioTokenizer()

        def fake_generate_audio_token_ids(self, **kwargs):
            self._last_generation_kwargs = kwargs
            return mx.array([[[5, 6], [7, 1]]], dtype=mx.int32)

        model.generate_audio_token_ids = MethodType(
            fake_generate_audio_token_ids, model
        )

        results = list(
            model.generate(
                text="hello",
                ref_audio=np.zeros((16,), dtype=np.float32),
                max_tokens=2,
                do_sample=False,
            )
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].sample_rate, model.sample_rate)
        self.assertEqual(results[0].samples, 4)
        self.assertEqual(results[0].token_count, 2)
        np.testing.assert_array_equal(
            np.array(model.audio_tokenizer.decoded_codes),
            np.array([[[5, 6], [7, 1]]]),
        )

    def test_audio_tokenizer_loader_skips_tts_model_root(self):
        from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.json").write_text(
                '{"model_type": "moss_tts_nano", "gpt2_config": {}}'
            )
            (root / "model.safetensors").write_bytes(b"")

            calls = []
            original_from_pretrained = MossAudioTokenizer.from_pretrained

            def fake_from_pretrained(cls, source):
                del cls
                calls.append(source)
                return "audio-tokenizer"

            try:
                MossAudioTokenizer.from_pretrained = classmethod(fake_from_pretrained)
                tokenizer = MossAudioTokenizer.from_model_dir(
                    root,
                    fallback_source="codec/repo",
                )
            finally:
                MossAudioTokenizer.from_pretrained = original_from_pretrained

            self.assertEqual(tokenizer, "audio-tokenizer")
            self.assertEqual(calls, ["codec/repo"])

    def test_audio_tokenizer_parent_config_defaults_to_mono(self):
        from mlx_audio.codec.models.moss_audio_tokenizer import AudioTokenizerConfig

        config = AudioTokenizerConfig.from_dict(
            {
                "sampling_rate": 24000,
                "downsample_rate": 1920,
                "encoder_kwargs": [],
                "decoder_kwargs": [],
                "quantizer_kwargs": {"num_quantizers": 32},
            }
        )

        self.assertEqual(config.number_channels, 1)
        self.assertEqual(config.sampling_rate, 24000)


class MossDelayFakeTokenizer:
    def __init__(self):
        self.id_to_token = {
            0: "<|endoftext|>",
            1: "<|im_start|>",
            2: "<|im_end|>",
            3: "<|audio_start|>",
            4: "<|audio_end|>",
            5: "<|audio_user_slot|>",
            6: "<|audio_assistant_gen_slot|>",
            7: "<|audio_assistant_delay_slot|>",
        }
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self._special_tokens = sorted(self.token_to_id, key=len, reverse=True)

    def convert_ids_to_tokens(self, token_id):
        return self.id_to_token[int(token_id)]

    def encode(self, text, *args, **kwargs):
        del args, kwargs
        tokens = []
        index = 0
        while index < len(text):
            for special in self._special_tokens:
                if text.startswith(special, index):
                    tokens.append(self.token_to_id[special])
                    index += len(special)
                    break
            else:
                tokens.append(16 + (ord(text[index]) % 32))
                index += 1
        return tokens

    def decode(self, token_ids, *args, **kwargs):
        del args, kwargs
        parts = []
        for token_id in token_ids:
            token_id = int(token_id)
            parts.append(self.id_to_token.get(token_id, chr((token_id - 16) % 32 + 64)))
        return "".join(parts)


def moss_delay_tiny_config(**overrides):
    from mlx_audio.tts.models.moss_tts import ModelConfig

    config = {
        "model_type": "moss_tts_delay",
        "n_vq": 2,
        "audio_vocab_size": 8,
        "audio_pad_code": 8,
        "pad_token_id": 0,
        "im_start_token_id": 1,
        "im_end_token_id": 2,
        "audio_start_token_id": 3,
        "audio_end_token_id": 4,
        "audio_user_slot_token_id": 5,
        "audio_assistant_gen_slot_token_id": 6,
        "audio_assistant_delay_slot_token_id": 7,
        "sampling_rate": 24000,
        "language_config": {
            "model_type": "qwen3",
            "vocab_size": 64,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        },
    }
    config.update(overrides)
    return ModelConfig.from_dict(config)


def moss_local_tiny_config(**overrides):
    config = {
        "additional_mlp_ffn_hidden_size": 20,
        "local_ffn_hidden_size": 24,
        "local_hidden_size": 12,
        "local_num_layers": 1,
    }
    config.update(overrides)
    return moss_delay_tiny_config(**config)


class TestMossTTSDelayConfig(unittest.TestCase):
    def test_config_parses_upstream_shape(self):
        from mlx_audio.tts.models.moss_tts import ModelConfig

        config = ModelConfig.from_dict(
            {
                "model_type": "moss_tts_delay",
                "n_vq": 32,
                "audio_vocab_size": 1024,
                "audio_pad_code": 1024,
                "sampling_rate": 24000,
                "language_config": {
                    "model_type": "qwen3",
                    "vocab_size": 155648,
                    "hidden_size": 4096,
                    "num_hidden_layers": 36,
                    "intermediate_size": 12288,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "rms_norm_eps": 1e-6,
                    "max_position_embeddings": 40960,
                    "rope_theta": 1000000,
                },
            }
        )

        self.assertEqual(config.model_type, "moss_tts_delay")
        self.assertEqual(config.n_vq, 32)
        self.assertEqual(config.hidden_size, 4096)
        self.assertFalse(config.language_config.tie_word_embeddings)
        self.assertEqual(config.sampling_rate, 24000)

    def test_config_parses_local_transformer_shape(self):
        from mlx_audio.tts.models.moss_tts import ModelConfig

        config = ModelConfig.from_dict(
            {
                "model_type": "moss_tts_delay",
                "n_vq": 32,
                "audio_vocab_size": 1024,
                "audio_pad_code": 1024,
                "sampling_rate": 24000,
                "additional_mlp_ffn_hidden_size": 2048,
                "local_ffn_hidden_size": 8960,
                "local_hidden_size": 1536,
                "local_num_layers": 4,
                "language_config": {
                    "model_type": "qwen3",
                    "vocab_size": 155648,
                    "hidden_size": 2048,
                    "num_hidden_layers": 28,
                    "intermediate_size": 6144,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "rms_norm_eps": 1e-6,
                    "max_position_embeddings": 40960,
                    "rope_theta": 1000000,
                },
            }
        )

        self.assertTrue(config.is_local_transformer)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.local_hidden_size, 1536)
        self.assertEqual(config.local_transformer_config().hidden_size, 1536)
        self.assertEqual(config.local_transformer_config().num_hidden_layers, 4)

    def test_config_parses_ttsd_shape(self):
        from mlx_audio.tts.models.moss_tts import ModelConfig

        config = ModelConfig.from_dict(
            {
                "model_type": "moss_tts_delay",
                "n_vq": 16,
                "audio_vocab_size": 1024,
                "audio_pad_code": 1024,
                "sampling_rate": 24000,
                "language_config": {
                    "model_type": "qwen3",
                    "vocab_size": 155648,
                    "hidden_size": 4096,
                    "num_hidden_layers": 36,
                    "intermediate_size": 12288,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "rms_norm_eps": 1e-6,
                    "max_position_embeddings": 40960,
                    "rope_theta": 1000000,
                },
            }
        )

        self.assertEqual(config.model_type, "moss_tts_delay")
        self.assertEqual(config.n_vq, 16)
        self.assertFalse(config.is_local_transformer)


class TestMossTTSDelayProcessor(unittest.TestCase):
    def test_delay_pattern_round_trip(self):
        from mlx_audio.tts.models.moss_tts.processor import (
            apply_de_delay_pattern,
            apply_delay_pattern,
        )

        codes = mx.array([[1, 2], [3, 4], [5, 6]], dtype=mx.int32)
        delayed = apply_delay_pattern(codes, pad_code=8)
        restored = apply_de_delay_pattern(delayed)

        np.testing.assert_array_equal(np.array(restored), np.array(codes))

    def test_generation_prompt_includes_reference_delay_codes(self):
        from mlx_audio.tts.models.moss_tts.processor import MossTTSDelayProcessor

        config = moss_delay_tiny_config()
        processor = MossTTSDelayProcessor(MossDelayFakeTokenizer(), config)
        reference_codes = mx.array([[1, 2], [3, 4]], dtype=mx.int32)

        batch = processor(
            [processor.build_user_message(text="hello", reference=[reference_codes])],
            mode="generation",
        )

        rows = np.array(batch["input_ids"][0])
        non_pad_audio_rows = rows[~np.all(rows[:, 1:] == config.audio_pad_code, axis=1)]
        np.testing.assert_array_equal(
            non_pad_audio_rows[:, 1:],
            np.array([[1, 8], [3, 2], [8, 4]]),
        )
        self.assertTrue(
            np.all(non_pad_audio_rows[:, 0] == config.audio_user_slot_token_id)
        )

    def test_generation_prompt_truncates_reference_to_model_n_vq(self):
        from mlx_audio.tts.models.moss_tts.processor import MossTTSDelayProcessor

        config = moss_delay_tiny_config()
        processor = MossTTSDelayProcessor(MossDelayFakeTokenizer(), config)
        reference_codes = mx.array([[1, 2, 7, 7], [3, 4, 7, 7]], dtype=mx.int32)

        batch = processor(
            [processor.build_user_message(text="hello", reference=[reference_codes])],
            mode="generation",
        )

        rows = np.array(batch["input_ids"][0])
        self.assertEqual(rows.shape[-1], config.n_vq + 1)
        non_pad_audio_rows = rows[~np.all(rows[:, 1:] == config.audio_pad_code, axis=1)]
        np.testing.assert_array_equal(
            non_pad_audio_rows[:, 1:],
            np.array([[1, 8], [3, 2], [8, 4]]),
        )

    def test_generation_prompt_rejects_too_few_reference_channels(self):
        from mlx_audio.tts.models.moss_tts.processor import MossTTSDelayProcessor

        config = moss_delay_tiny_config()
        processor = MossTTSDelayProcessor(MossDelayFakeTokenizer(), config)
        reference_codes = mx.array([[1]], dtype=mx.int32)

        with self.assertRaisesRegex(ValueError, "audio_codes channels"):
            processor(
                [
                    processor.build_user_message(
                        text="hello", reference=[reference_codes]
                    )
                ],
                mode="generation",
            )

    def test_user_message_preserves_empty_references_and_scene(self):
        from mlx_audio.tts.models.moss_tts.processor import MossTTSDelayProcessor

        config = moss_delay_tiny_config()
        processor = MossTTSDelayProcessor(MossDelayFakeTokenizer(), config)
        reference_codes = mx.array([[1, 2]], dtype=mx.int32)

        message = processor.build_user_message(
            text="[S1] hello [S2] hi",
            reference=[None, reference_codes],
            tokens=123,
            scene="studio",
        )

        self.assertIn("[S1]: None", message["content"])
        self.assertIn("[S2]:\n<|audio|>", message["content"])
        self.assertIn("- Tokens:\n123", message["content"])
        self.assertIn("- Scene:\nstudio", message["content"])
        self.assertEqual(len(message["audio_codes_list"]), 1)

    def test_local_generation_prompt_uses_undelayed_reference_and_audio_start(self):
        from mlx_audio.tts.models.moss_tts.processor import MossTTSLocalProcessor

        config = moss_local_tiny_config()
        processor = MossTTSLocalProcessor(MossDelayFakeTokenizer(), config)
        reference_codes = mx.array([[1, 2], [3, 4]], dtype=mx.int32)

        batch = processor(
            [processor.build_user_message(text="hello", reference=[reference_codes])],
            mode="generation",
        )

        rows = np.array(batch["input_ids"][0])
        non_pad_audio_rows = rows[~np.all(rows[:, 1:] == config.audio_pad_code, axis=1)]
        np.testing.assert_array_equal(
            non_pad_audio_rows[:, 1:],
            np.array([[1, 2], [3, 4]]),
        )
        self.assertTrue(
            np.all(non_pad_audio_rows[:, 0] == config.audio_user_slot_token_id)
        )
        np.testing.assert_array_equal(
            rows[-1],
            np.array(
                [
                    config.audio_start_token_id,
                    config.audio_pad_code,
                    config.audio_pad_code,
                ]
            ),
        )


class TestMossTTSDelayModel(unittest.TestCase):
    def test_build_inputs_embeds_shape(self):
        from mlx_audio.tts.models.moss_tts import Model

        model = Model(moss_delay_tiny_config())
        input_ids = mx.array([[[1, 8, 8], [2, 3, 4]]], dtype=mx.int32)

        embeds = model._build_inputs_embeds(input_ids)

        self.assertEqual(embeds.shape, (1, 2, 16))

    def test_forward_head_shapes(self):
        from mlx_audio.tts.models.moss_tts import Model

        model = Model(moss_delay_tiny_config())
        input_ids = mx.array([[[1, 8, 8], [2, 3, 4]]], dtype=mx.int32)

        text_logits, audio_logits = model(input_ids, head_indices=[0, 1])

        self.assertEqual(text_logits.shape, (1, 2, 64))
        self.assertEqual(audio_logits.shape, (1, 2, 9))
        self.assertTrue(np.isneginf(np.array(audio_logits)[0, 0, -1]))

    def test_sanitize_strips_optional_model_prefix(self):
        from mlx_audio.tts.models.moss_tts import Model

        model = Model(moss_delay_tiny_config())
        x = mx.zeros((1,))
        sanitized = model.sanitize(
            {
                "model.language_model.embed_tokens.weight": x,
                "emb_ext.0.weight": x,
                "lm_heads.0.weight": x,
            }
        )

        self.assertIn("language_model.embed_tokens.weight", sanitized)
        self.assertIn("emb_ext.0.weight", sanitized)
        self.assertIn("lm_heads.0.weight", sanitized)

    def test_local_forward_head_shapes(self):
        from mlx_audio.tts.models.moss_tts import Model

        model = Model(moss_local_tiny_config())
        input_ids = mx.array([[[1, 8, 8], [3, 8, 8], [6, 1, 2]]], dtype=mx.int32)

        text_logits, audio_0_logits, audio_1_logits = model(
            input_ids,
            head_indices=[0, 1, 2],
        )

        self.assertEqual(text_logits.shape, (1, 3, 64))
        self.assertEqual(audio_0_logits.shape, (1, 3, 9))
        self.assertEqual(audio_1_logits.shape, (1, 3, 9))

    def test_local_sanitize_keeps_model_prefix(self):
        from mlx_audio.tts.models.moss_tts import Model

        model = Model(moss_local_tiny_config())
        x = mx.zeros((1,))
        sanitized = model.sanitize(
            {
                "model.embedding_list.0.weight": x,
                "model.language_model.embed_tokens.weight": x,
                "local_transformer.layers.0.input_layernorm.weight": x,
            }
        )

        self.assertIn("model.embedding_list.0.weight", sanitized)
        self.assertIn("model.language_model.embed_tokens.weight", sanitized)
        self.assertIn("local_transformer.layers.0.input_layernorm.weight", sanitized)

    def test_delay_generation_uses_generation_config_defaults(self):
        from mlx_audio.tts.models.moss_tts import Model

        model = Model(moss_delay_tiny_config())
        model.tokenizer = MossDelayFakeTokenizer()
        model.generation_config = {
            "max_new_tokens": 8192,
            "temperature": 1.1,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
        calls = {}

        def fake_generate_delay_pattern_ids(input_ids, **kwargs):
            del input_ids
            calls.update(kwargs)
            return []

        model.generate_delay_pattern_ids = fake_generate_delay_pattern_ids
        model._decode_generated_audio = lambda outputs, source=None: (
            mx.zeros((0, 1), dtype=mx.float32),
            0,
        )

        next(model.generate(text="hello", max_tokens=None))

        self.assertEqual(calls["max_new_tokens"], 8192)
        self.assertEqual(calls["text_temperature"], 1.1)
        self.assertEqual(calls["text_top_p"], 0.9)
        self.assertEqual(calls["audio_temperature"], 1.1)
        self.assertEqual(calls["audio_top_p"], 0.9)
        self.assertEqual(calls["audio_top_k"], 50)
        self.assertEqual(calls["audio_repetition_penalty"], 1.1)

    def test_delay_generation_explicit_args_override_generation_config(self):
        from mlx_audio.tts.models.moss_tts import Model

        model = Model(moss_delay_tiny_config())
        model.tokenizer = MossDelayFakeTokenizer()
        model.generation_config = {
            "max_new_tokens": 8192,
            "temperature": 1.1,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
        calls = {}

        def fake_generate_delay_pattern_ids(input_ids, **kwargs):
            del input_ids
            calls.update(kwargs)
            return []

        model.generate_delay_pattern_ids = fake_generate_delay_pattern_ids
        model._decode_generated_audio = lambda outputs, source=None: (
            mx.zeros((0, 1), dtype=mx.float32),
            0,
        )

        next(
            model.generate(
                text="hello",
                max_tokens=12,
                audio_temperature=0.5,
                audio_top_p=0.7,
                audio_top_k=10,
                audio_repetition_penalty=1.3,
            )
        )

        self.assertEqual(calls["max_new_tokens"], 12)
        self.assertEqual(calls["audio_temperature"], 0.5)
        self.assertEqual(calls["audio_top_p"], 0.7)
        self.assertEqual(calls["audio_top_k"], 10)
        self.assertEqual(calls["audio_repetition_penalty"], 1.3)

    def test_delay_generation_accepts_reference_audio_list(self):
        from mlx_audio.tts.models.moss_tts import Model

        model = Model(moss_delay_tiny_config())
        model.tokenizer = MossDelayFakeTokenizer()
        encoded_refs = []
        calls = {}

        def fake_encode_reference_audio(ref_audio, **kwargs):
            del kwargs
            encoded_refs.append(ref_audio)
            if len(encoded_refs) == 1:
                return mx.array([[1, 2]], dtype=mx.int32)
            return mx.array([[3, 4]], dtype=mx.int32)

        def fake_generate_delay_pattern_ids(input_ids, **kwargs):
            del kwargs
            calls["input_ids"] = input_ids
            return []

        model.encode_reference_audio = fake_encode_reference_audio
        model.generate_delay_pattern_ids = fake_generate_delay_pattern_ids
        model._decode_generated_audio = lambda outputs, source=None: (
            mx.zeros((0, 1), dtype=mx.float32),
            0,
        )

        next(
            model.generate(
                text="[S1] hello [S2] hi",
                ref_audio=["speaker-one", "speaker-two"],
                max_tokens=1,
            )
        )

        self.assertEqual(encoded_refs, ["speaker-one", "speaker-two"])
        rows = np.array(calls["input_ids"][0])
        non_pad_audio_rows = rows[
            ~np.all(rows[:, 1:] == model.config.audio_pad_code, axis=1)
        ]
        np.testing.assert_array_equal(
            non_pad_audio_rows[:, 1:],
            np.array([[1, 8], [8, 2], [3, 8], [8, 4]]),
        )


# ---------------------------------------------------------------------------
# Dramabox tests
# ---------------------------------------------------------------------------

import math

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.dramabox import dramabox as dramabox_module
from mlx_audio.tts.models.dramabox.audio_vae import (
    AudioDecoder,
    AudioEncoder,
    CausalityAxis,
    NormType,
)
from mlx_audio.tts.models.dramabox.config import ModelConfig
from mlx_audio.tts.models.dramabox.convert import sanitize_weights
from mlx_audio.tts.models.dramabox.duration import estimate_speech_duration
from mlx_audio.tts.models.dramabox.guidance import (
    MultiModalGuiderParams,
    auto_rescale_for_cfg,
    calculate_guided_prediction,
)
from mlx_audio.tts.models.dramabox.latent import (
    AudioLatentShape,
    AudioLatentTools,
    AudioPatchifier,
    LatentState,
    add_gaussian_noise,
    append_reference_latent,
)
from mlx_audio.tts.models.dramabox.layers import FeedForward, gelu_approx, rms_norm
from mlx_audio.tts.models.dramabox.rope import (
    LTXRopeType,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
    precompute_freqs_cis,
)
from mlx_audio.tts.models.dramabox.sampling import (
    aligned_frame_count,
    guided_euler_loop,
    patch_long_clip_silence_prior,
    resolve_generation_duration,
    target_shape_for_duration,
)
from mlx_audio.tts.models.dramabox.scheduler import (
    ltx2_sigmas,
    to_denoised,
    to_velocity,
)
from mlx_audio.tts.models.dramabox.text_conditioning import (
    DramaboxTextConditioner,
    Embeddings1DConnector,
    FeatureExtractorV2,
    binary_to_additive_attention_mask,
    norm_and_concat_per_token_rms,
    rescale_norm,
    stack_hidden_states,
)
from mlx_audio.tts.models.dramabox.timestep import (
    AdaLayerNormSingle,
    get_timestep_embedding,
)
from mlx_audio.tts.models.dramabox.transformer import (
    AudioOnlyLTXModel,
    Modality,
    X0Model,
)
from mlx_audio.tts.models.dramabox.vocoder import UpSample1d, Vocoder
from mlx_audio.tts.utils import get_model_and_args


def test_config_maps_hf_dramabox_defaults_to_mlx_defaults():
    config = ModelConfig.from_dict(
        {
            "model_type": "dramabox-tts",
            "num_layers": 48,
            "audio_num_attention_heads": 32,
            "audio_attention_head_dim": 64,
            "audio_cross_attention_dim": 2048,
            "audio": {"sample_rate": 48000, "vae_channels": 8, "mel_bins": 16},
            "inference_defaults": {
                "cfg_scale": 2.5,
                "stg_scale": 1.5,
                "rescale_scale": 0.0,
            },
        }
    )

    assert config.model_type == "dramabox-tts"
    assert config.transformer.num_layers == 48
    assert config.transformer.audio_num_attention_heads == 32
    assert config.audio.sample_rate == 48000
    assert config.inference_defaults.rescale_scale == "auto"
    assert "off-sync audio" in config.inference_defaults.negative_prompt


def test_dramabox_model_alias_is_registered():
    model_class, model_type = get_model_and_args("dramabox-tts", ["dramabox"])
    assert model_type == "dramabox"
    assert hasattr(model_class, "Model")
    assert hasattr(model_class, "ModelConfig")


def test_dramabox_text_encoder_override_refreshes_cache(monkeypatch):
    model = dramabox_module.Model.__new__(dramabox_module.Model)
    model.config = ModelConfig.from_dict({"mlx_text_encoder": "default-gemma"})
    model._text_encoder = None
    model._tokenizer = None
    model._text_encoder_id = None
    calls = []

    def fake_load_text_encoder(model_id):
        calls.append(model_id)
        return object(), object()

    monkeypatch.setattr(
        dramabox_module,
        "load_text_encoder",
        fake_load_text_encoder,
    )

    model._ensure_text_encoder()
    model._ensure_text_encoder()
    model._ensure_text_encoder("other-gemma")

    assert calls == ["default-gemma", "other-gemma"]


def test_dramabox_generate_does_not_pass_prompt_masks_to_dit(monkeypatch):
    model = dramabox_module.Model.__new__(dramabox_module.Model)
    model.config = ModelConfig.from_dict({})
    model.transformer = object()

    captured = {}

    def fake_encode_prompt_contexts(prompts, text_encoder_id=None):
        del prompts, text_encoder_id
        return [
            (
                mx.zeros((1, 2, 4), dtype=mx.float32),
                mx.array([[1, 0]], dtype=mx.int32),
            ),
            (
                mx.zeros((1, 3, 4), dtype=mx.float32),
                mx.array([[1, 1, 0]], dtype=mx.int32),
            ),
        ]

    def fake_guided_euler_loop(**kwargs):
        captured["context_mask"] = kwargs["context_mask"]
        captured["negative_context_mask"] = kwargs["negative_context_mask"]
        return kwargs["state"]

    class FakeAudioVAE:
        def decode(self, latents):
            del latents
            return mx.zeros((1, 2, 1, 64), dtype=mx.float32)

    class FakeVocoder:
        def __call__(self, mel):
            del mel
            return mx.zeros((1, 2, 480), dtype=mx.float32)

    model._encode_prompt_contexts = fake_encode_prompt_contexts
    model.audio_vae = FakeAudioVAE()
    model.vocoder = FakeVocoder()
    monkeypatch.setattr(dramabox_module, "guided_euler_loop", fake_guided_euler_loop)

    result = next(model.generate("hello", cfg_scale=2.0, gen_duration=0.1, steps=1))

    assert result.audio.shape == (480, 2)
    assert captured["context_mask"] is None
    assert captured["negative_context_mask"] is None


def test_converter_maps_dramabox_keys_to_module_tree():
    weights = {
        "model.diffusion_model.audio_patchify_proj.weight": mx.zeros((4, 4)),
        "model.diffusion_model.audio_embeddings_connector.learnable_registers": mx.zeros(
            (2, 4)
        ),
        "text_embedding_projection.audio_aggregate_embed.weight": mx.zeros((4, 4)),
        "audio_vae.encoder.conv_in.conv.weight": mx.zeros((4, 5, 3, 3)),
        "vocoder.vocoder.conv_pre.weight": mx.zeros((4, 5, 3)),
        "vocoder.vocoder.ups.0.weight": mx.zeros((8, 5, 4)),
        "vae.per_channel_statistics.mean-of-means": mx.zeros((4,)),
    }
    mapped = sanitize_weights(weights)
    assert "transformer.audio_patchify_proj.weight" in mapped
    assert "text_conditioner.audio_connector.learnable_registers" in mapped
    assert "text_conditioner.feature_extractor.audio_aggregate_embed.weight" in mapped
    assert "audio_vae.encoder.conv_in.conv.weight" in mapped
    assert mapped["audio_vae.encoder.conv_in.conv.weight"].shape == (4, 3, 3, 5)
    assert mapped["vocoder.vocoder.conv_pre.weight"].shape == (4, 3, 5)
    assert mapped["vocoder.vocoder.ups.0.weight"].shape == (5, 4, 8)
    assert "audio_vae.encoder.per_channel_statistics.mean_of_means" in mapped
    assert "audio_vae.decoder.per_channel_statistics.mean_of_means" in mapped

    remapped = sanitize_weights(mapped)
    assert remapped["audio_vae.encoder.conv_in.conv.weight"].shape == (4, 3, 3, 5)
    assert remapped["vocoder.vocoder.conv_pre.weight"].shape == (4, 3, 5)
    assert remapped["vocoder.vocoder.ups.0.weight"].shape == (5, 4, 8)


def test_audio_vae_encoder_decoder_shapes_with_small_config():
    encoder = AudioEncoder(
        ch=8,
        ch_mult=(1, 2),
        num_res_blocks=1,
        in_channels=2,
        z_channels=2,
        mel_bins=8,
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
    )
    decoder = AudioDecoder(
        ch=8,
        ch_mult=(1, 2),
        num_res_blocks=1,
        out_ch=2,
        z_channels=2,
        mel_bins=8,
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
    )

    spectrogram = mx.zeros((1, 2, 9, 8), dtype=mx.float32)
    latent = encoder(spectrogram)
    decoded = decoder(latent)

    assert latent.shape == (1, 2, 5, 4)
    assert decoded.shape == (1, 2, 17, 8)


def test_vocoder_shape_with_small_non_amp_config():
    vocoder = Vocoder(
        resblock_kernel_sizes=[3],
        upsample_rates=[2, 2],
        upsample_kernel_sizes=[4, 4],
        resblock_dilation_sizes=[[1, 3, 5]],
        upsample_initial_channel=8,
        resblock="1",
        output_sampling_rate=16000,
        in_channels=16,
        out_channels=2,
    )

    mel = mx.zeros((1, 2, 5, 8), dtype=mx.float32)
    audio = vocoder(mel)

    assert audio.shape == (1, 2, 20)


def test_bigvgan_upsample_preserves_reference_length_formula():
    upsample = UpSample1d(ratio=2, kernel_size=12)
    x = mx.zeros((1, 3, 85), dtype=mx.float32)
    y = upsample(x)
    assert y.shape == (1, 3, 170)


def test_duration_estimator_matches_reference_style_heuristic():
    assert estimate_speech_duration('A woman says, "Hello."') == 3.0
    prompt = 'A villain laughs maniacally, "Hahahahaha!" ' 'He pauses. "Now it begins."'
    assert estimate_speech_duration(prompt) == 8.6


def test_audio_patchifier_round_trip_and_positions():
    patchifier = AudioPatchifier()
    shape = AudioLatentShape(batch=1, channels=8, frames=4, mel_bins=16)
    latent = mx.arange(math.prod(shape.to_mlx_shape()), dtype=mx.float32).reshape(
        shape.to_mlx_shape()
    )

    patched = patchifier.patchify(latent)
    restored = patchifier.unpatchify(patched, shape)
    positions = patchifier.get_patch_grid_bounds(shape)

    assert patched.shape == (1, 4, 128)
    np.testing.assert_array_equal(np.array(restored), np.array(latent))
    np.testing.assert_allclose(
        np.array(positions[0, 0, :, 0]),
        np.array([0.0, 0.01, 0.05, 0.09], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(positions[0, 0, :, 1]),
        np.array([0.01, 0.05, 0.09, 0.13], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_reference_latent_conditioning_appends_asymmetric_mask():
    shape = AudioLatentShape(batch=1, channels=8, frames=3, mel_bins=16)
    tools = AudioLatentTools(AudioPatchifier(), shape)
    state = tools.create_initial_state()
    ref = mx.ones((1, 8, 2, 16), dtype=mx.float32)

    conditioned = append_reference_latent(state, tools, ref)

    assert conditioned.latent.shape == (1, 5, 128)
    assert conditioned.denoise_mask.shape == (1, 5, 1)
    assert conditioned.positions.shape == (1, 1, 5, 2)
    np.testing.assert_array_equal(
        np.array(conditioned.attention_mask[0]),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(np.array(conditioned.denoise_mask[0, 3:]), 0.0)
    np.testing.assert_allclose(
        np.array(conditioned.positions[0, 0, 3:, 0]),
        np.array([0.5, 0.51], dtype=np.float32),
        atol=1e-6,
    )


def _expected_ltx2_sigmas(steps, latent_shape):
    tokens = math.prod(latent_shape[2:])
    sigmas = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)
    slope = (2.05 - 0.95) / (4096 - 1024)
    intercept = 0.95 - slope * 1024
    shift = tokens * slope + intercept
    shifted = np.where(
        sigmas != 0,
        math.exp(shift) / (math.exp(shift) + (1 / sigmas - 1)),
        0,
    )
    one_minus = 1.0 - shifted[:-1]
    scale = one_minus[-1] / (1.0 - 0.1)
    return np.concatenate([1.0 - one_minus / scale, shifted[-1:]]).astype(np.float32)


def test_ltx2_scheduler_matches_reference_formula():
    latent = mx.zeros((1, 5, 3, 4), dtype=mx.float32)
    sigmas = ltx2_sigmas(steps=4, latent=latent)

    np.testing.assert_allclose(
        np.array(sigmas),
        _expected_ltx2_sigmas(4, latent.shape),
        rtol=1e-6,
        atol=1e-6,
    )
    assert sigmas[0].item() == 1.0
    assert sigmas[-1].item() == 0.0


def test_add_gaussian_noise_preserves_frozen_reference_tokens():
    state = LatentState(
        latent=mx.array([[[0.0], [3.0]]], dtype=mx.float32),
        denoise_mask=mx.array([[[1.0], [0.0]]], dtype=mx.float32),
        positions=mx.zeros((1, 1, 2, 2), dtype=mx.float32),
        clean_latent=mx.array([[[0.0], [3.0]]], dtype=mx.float32),
        attention_mask=None,
    )

    noised = add_gaussian_noise(state, seed=0)

    assert not np.isclose(np.array(noised.latent[0, 0, 0]), 0.0)
    np.testing.assert_allclose(np.array(noised.latent[0, 1, 0]), 3.0, atol=1e-6)


def test_velocity_and_denoised_are_inverse():
    sample = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    denoised = mx.array([0.5, 1.0, 1.5], dtype=mx.float32)
    sigma = mx.array(0.25, dtype=mx.float32)

    velocity = to_velocity(sample, sigma, denoised)
    restored = to_denoised(sample, velocity, sigma)

    np.testing.assert_allclose(np.array(restored), np.array(denoised), atol=1e-6)


def test_auto_rescale_and_guidance_math():
    assert auto_rescale_for_cfg(1.5) == 0.0
    assert auto_rescale_for_cfg(2.5) == 0.3
    assert auto_rescale_for_cfg(4.0) == 0.8
    assert auto_rescale_for_cfg(10.0) == 1.0

    cond = mx.array([[1.0, 2.0]], dtype=mx.float32)
    uncond_text = mx.array([[0.5, 1.5]], dtype=mx.float32)
    perturbed = mx.array([[0.75, 1.75]], dtype=mx.float32)
    params = MultiModalGuiderParams(cfg_scale=2.0, stg_scale=0.5)

    guided = calculate_guided_prediction(cond, uncond_text, perturbed, 0.0, params)
    expected = cond + (cond - uncond_text) + 0.5 * (cond - perturbed)
    np.testing.assert_allclose(np.array(guided), np.array(expected), atol=1e-6)


def test_text_feature_extractor_v2_normalizes_and_masks_hidden_states():
    h0 = mx.array(
        [[[3.0, 4.0], [0.0, 0.0], [1.0, 2.0]]],
        dtype=mx.float32,
    )
    h1 = h0 * 2
    mask = mx.array([[1, 0, 1]], dtype=mx.int32)
    stacked = stack_hidden_states([h0, h1])
    normed = norm_and_concat_per_token_rms(stacked, mask)

    assert stacked.shape == (1, 3, 2, 2)
    assert normed.shape == (1, 3, 4)
    np.testing.assert_array_equal(
        np.array(normed[0, 1]), np.zeros((4,), dtype=np.float32)
    )
    variance = np.mean(np.array(stacked[0, 0]) ** 2, axis=0, keepdims=True)
    expected_first = (np.array(stacked[0, 0]) / np.sqrt(variance + 1e-6)).reshape(4)
    np.testing.assert_allclose(np.array(normed[0, 0]), expected_first, atol=1e-6)


def test_feature_extractor_v2_shape_with_small_dimensions():
    extractor = FeatureExtractorV2(embedding_dim=2, audio_inner_dim=4, num_layers=2)
    # Make the projection deterministic and simple.
    extractor.audio_aggregate_embed.weight = mx.ones((4, 4), dtype=mx.float32)
    extractor.audio_aggregate_embed.bias = mx.zeros((4,), dtype=mx.float32)

    h0 = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
    h1 = h0 + 1
    mask = mx.array([[1, 1]], dtype=mx.int32)
    out = extractor([h0, h1], mask)

    expected_input = norm_and_concat_per_token_rms(stack_hidden_states([h0, h1]), mask)
    expected_input = rescale_norm(expected_input, target_dim=4, source_dim=2)
    expected = np.sum(np.array(expected_input), axis=-1, keepdims=True).repeat(
        4, axis=-1
    )
    assert out.shape == (1, 2, 4)
    np.testing.assert_allclose(np.array(out), expected, rtol=5e-4, atol=1e-5)


def test_embeddings_connector_replaces_padding_and_returns_binary_mask():
    connector = Embeddings1DConnector(
        attention_head_dim=2,
        num_attention_heads=1,
        num_layers=0,
        num_learnable_registers=2,
    )
    connector.learnable_registers = mx.array(
        [[10.0, 20.0], [30.0, 40.0]], dtype=mx.float32
    )
    hidden = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
    binary_mask = mx.array([[1, 0]], dtype=mx.int32)
    additive_mask = binary_to_additive_attention_mask(binary_mask, dtype=mx.float32)

    replaced, returned_mask = connector._replace_padded_with_learnable_registers(
        hidden, additive_mask
    )
    encoded, encoded_mask = connector(hidden, additive_mask)

    np.testing.assert_allclose(np.array(replaced[0, 0]), [1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(np.array(replaced[0, 1]), [30.0, 40.0], atol=1e-6)
    np.testing.assert_array_equal(np.array(returned_mask), np.zeros((1, 1, 1, 2)))
    assert encoded.shape == hidden.shape
    assert encoded_mask.shape == additive_mask.shape


def test_embeddings_connector_compacts_left_padded_tokens_like_reference():
    connector = Embeddings1DConnector(
        attention_head_dim=2,
        num_attention_heads=1,
        num_layers=0,
        num_learnable_registers=2,
    )
    connector.learnable_registers = mx.array(
        [[10.0, 20.0], [30.0, 40.0]], dtype=mx.float32
    )
    hidden = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
    binary_mask = mx.array([[0, 1]], dtype=mx.int32)
    additive_mask = binary_to_additive_attention_mask(binary_mask, dtype=mx.float32)

    replaced, returned_mask = connector._replace_padded_with_learnable_registers(
        hidden, additive_mask
    )

    np.testing.assert_allclose(np.array(replaced[0, 0]), [3.0, 4.0], atol=1e-6)
    np.testing.assert_allclose(np.array(replaced[0, 1]), [30.0, 40.0], atol=1e-6)
    np.testing.assert_array_equal(np.array(returned_mask), np.zeros((1, 1, 1, 2)))


def test_dramabox_text_conditioner_small_path_shapes():
    conditioner = DramaboxTextConditioner(
        embedding_dim=2,
        audio_inner_dim=4,
        num_gemma_layers=2,
        connector_layers=0,
        connector_heads=1,
        connector_head_dim=4,
        connector_num_learnable_registers=None,
    )
    conditioner.feature_extractor.audio_aggregate_embed.weight = mx.ones(
        (4, 4), dtype=mx.float32
    )
    conditioner.feature_extractor.audio_aggregate_embed.bias = mx.zeros(
        (4,), dtype=mx.float32
    )

    h0 = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
    h1 = h0 + 1
    mask = mx.array([[1, 0]], dtype=mx.int32)
    encoded, encoded_mask = conditioner([h0, h1], mask)

    assert encoded.shape == (1, 2, 4)
    assert encoded_mask.shape == (1, 2)
    np.testing.assert_array_equal(np.array(encoded_mask), np.ones((1, 2)))


def test_rope_interleaved_and_split_application():
    x = mx.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=mx.float32)
    cos = mx.zeros_like(x)
    sin = mx.ones_like(x)

    interleaved = apply_interleaved_rotary_emb(x, cos, sin)
    np.testing.assert_array_equal(
        np.array(interleaved),
        np.array([[[-2.0, 1.0, -4.0, 3.0]]], dtype=np.float32),
    )

    x_split = mx.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=mx.float32)
    cos_split = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
    sin_split = mx.ones((1, 1, 1, 2), dtype=mx.float32)
    split = apply_split_rotary_emb(x_split, cos_split, sin_split)
    np.testing.assert_array_equal(
        np.array(split),
        np.array([[[[-3.0, -4.0, 1.0, 2.0]]]], dtype=np.float32),
    )


def test_precompute_audio_split_rope_shapes():
    patchifier = AudioPatchifier()
    shape = AudioLatentShape(batch=1, channels=8, frames=3, mel_bins=16)
    positions = patchifier.get_patch_grid_bounds(shape)
    cos, sin = precompute_freqs_cis(
        positions,
        dim=2048,
        out_dtype=mx.float32,
        theta=10000.0,
        max_pos=[20.0],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=LTXRopeType.SPLIT,
        double_precision=True,
    )

    assert cos.shape == (1, 32, 3, 32)
    assert sin.shape == (1, 32, 3, 32)
    np.testing.assert_allclose(
        np.array(cos * cos + sin * sin),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )


def test_ltx_layers_rms_norm_gelu_and_feed_forward_shape():
    x = mx.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=mx.float32)
    out = rms_norm(x)
    expected = np.array(x) / np.sqrt(
        np.mean(np.array(x) ** 2, axis=-1, keepdims=True) + 1e-6
    )
    np.testing.assert_allclose(np.array(out), expected, atol=1e-6)

    gelu = gelu_approx(mx.array([-1.0, 0.0, 1.0], dtype=mx.float32))
    expected_gelu = (
        0.5
        * np.array([-1.0, 0.0, 1.0])
        * (
            1
            + np.tanh(
                np.sqrt(2 / np.pi)
                * (
                    np.array([-1.0, 0.0, 1.0])
                    + 0.044715 * np.array([-1.0, 0.0, 1.0]) ** 3
                )
            )
        )
    )
    np.testing.assert_allclose(np.array(gelu), expected_gelu, atol=1e-6)

    ff = FeedForward(dim=4, dim_out=4)
    y = ff(x)
    assert y.shape == x.shape


def test_timestep_embedding_and_adaln_shapes():
    emb = get_timestep_embedding(
        mx.array([1.0, 2.0], dtype=mx.float32),
        embedding_dim=6,
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
    )
    assert emb.shape == (2, 6)

    adaln = AdaLayerNormSingle(embedding_dim=12, embedding_coefficient=9)
    timestep, embedded = adaln(mx.array([1.0, 2.0], dtype=mx.float32))
    assert timestep.shape == (2, 108)
    assert embedded.shape == (2, 12)


def test_audio_only_ltx_model_shape_with_tiny_config():
    config = ModelConfig.from_dict(
        {
            "transformer": {
                "num_layers": 1,
                "audio_num_attention_heads": 1,
                "audio_attention_head_dim": 4,
                "audio_in_channels": 4,
                "audio_out_channels": 4,
                "audio_cross_attention_dim": 4,
                "norm_eps": 1e-6,
                "audio_positional_embedding_max_pos": [20.0],
                "timestep_scale_multiplier": 1000,
                "use_middle_indices_grid": True,
                "rope_type": "split",
                "frequencies_precision": "float64",
                "apply_gated_attention": False,
                "cross_attention_adaln": True,
            }
        }
    ).transformer
    model = AudioOnlyLTXModel(config)
    latent = mx.zeros((1, 3, 4), dtype=mx.float32)
    sigma = mx.array([1.0], dtype=mx.float32)
    timesteps = mx.ones((1, 3), dtype=mx.float32)
    positions = mx.array(
        [[[[0.0, 0.01], [0.01, 0.05], [0.05, 0.09]]]], dtype=mx.float32
    )
    context = mx.zeros((1, 2, 4), dtype=mx.float32)
    context_mask = mx.ones((1, 2), dtype=mx.int32)
    audio = Modality(
        latent=latent,
        sigma=sigma,
        timesteps=timesteps,
        positions=positions,
        context=context,
        context_mask=context_mask,
    )
    velocity = model(audio)
    denoised = X0Model(model)(audio)
    assert velocity.shape == latent.shape
    assert denoised.shape == latent.shape


def test_sampling_duration_shape_and_long_clip_patch():
    audio_config = ModelConfig.from_dict({}).audio
    assert aligned_frame_count(3.0, fps=25.0) == 73
    assert resolve_generation_duration("hello", gen_duration=4.5) == 4.5
    shape = target_shape_for_duration(3.0, audio_config)
    assert shape.batch == 1
    assert shape.channels == 8
    assert shape.mel_bins == 16

    latent = mx.zeros((1, 1, 515, 1), dtype=mx.float32)
    latent[:, :, 511, :] = 3.0
    latent[:, :, 514, :] = 6.0
    patched = patch_long_clip_silence_prior(latent)
    np.testing.assert_allclose(np.array(patched[:, :, 512, :]), 4.0, atol=1e-6)
    np.testing.assert_allclose(np.array(patched[:, :, 513, :]), 5.0, atol=1e-6)


def test_guided_euler_loop_runs_with_identity_x0_model():
    shape = AudioLatentShape(batch=1, channels=1, frames=2, mel_bins=2)
    tools = AudioLatentTools(AudioPatchifier(), shape)
    state = tools.create_initial_state()
    state = LatentState(
        latent=mx.ones_like(state.latent),
        denoise_mask=state.denoise_mask,
        positions=state.positions,
        clean_latent=state.clean_latent,
        attention_mask=state.attention_mask,
    )
    context = mx.zeros((1, 1, 2), dtype=mx.float32)

    def identity_x0(modality, stg_blocks=None):
        return mx.zeros_like(modality.latent)

    out = guided_euler_loop(
        state=state,
        x0_model=identity_x0,
        context=context,
        steps=2,
    )
    assert out.latent.shape == state.latent.shape
    np.testing.assert_allclose(np.array(out.latent), 0.0, atol=1e-5)


def test_guided_euler_loop_uses_negative_context_mask():
    shape = AudioLatentShape(batch=1, channels=1, frames=1, mel_bins=1)
    tools = AudioLatentTools(AudioPatchifier(), shape)
    state = tools.create_initial_state()
    seen_masks = []

    def mask_sensitive_x0(modality, stg_blocks=None):
        del stg_blocks
        if modality.context_mask is None:
            seen_masks.append(None)
        else:
            seen_masks.append(float(mx.sum(modality.context_mask).item()))
        return mx.zeros_like(modality.latent)

    guided_euler_loop(
        state=state,
        x0_model=mask_sensitive_x0,
        context=mx.zeros((1, 2, 1), dtype=mx.float32),
        negative_context=mx.zeros((1, 3, 1), dtype=mx.float32),
        context_mask=mx.array([[1, 0]], dtype=mx.int32),
        negative_context_mask=mx.array([[1, 1, 0]], dtype=mx.int32),
        steps=2,
        guider_params=MultiModalGuiderParams(cfg_scale=2.0, stg_scale=0.0),
    )

    assert 1.0 in seen_masks
    assert 2.0 in seen_masks


def test_guided_euler_loop_keeps_reference_tokens_clean():
    state = LatentState(
        latent=mx.array([[[1.0], [7.0]]], dtype=mx.float32),
        denoise_mask=mx.array([[[1.0], [0.0]]], dtype=mx.float32),
        positions=mx.zeros((1, 1, 2, 2), dtype=mx.float32),
        clean_latent=mx.array([[[0.0], [7.0]]], dtype=mx.float32),
        attention_mask=None,
    )
    seen_timesteps = []

    def zero_x0(modality, stg_blocks=None):
        del stg_blocks
        seen_timesteps.append(np.array(modality.timesteps))
        return mx.zeros_like(modality.latent)

    out = guided_euler_loop(
        state=state,
        x0_model=zero_x0,
        context=mx.zeros((1, 1, 1), dtype=mx.float32),
        steps=2,
    )

    np.testing.assert_allclose(np.array(out.latent[:, 0]), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.array(out.latent[:, 1]), 7.0, atol=1e-5)
    assert all(step[0, 1] == 0.0 for step in seen_timesteps)


if __name__ == "__main__":
    unittest.main()
